"""Dunhuang dance motion model: a unified predictor / interpolator.


"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bvh_utils import NUM_JOINTS, PARENTS, LIMB_GROUPS, NUM_FAMILIES


# ---------------------------------------------------------------------------
# Graph structures
# ---------------------------------------------------------------------------

def build_joint_adjacency():
    """Symmetric normalized adjacency over the 27 joints using PARENTS."""
    A = np.eye(NUM_JOINTS, dtype=np.float32)
    for j, p in enumerate(PARENTS):
        if p >= 0:
            A[j, p] = 1.0
            A[p, j] = 1.0
    d = A.sum(axis=1)
    d_inv = 1.0 / np.sqrt(np.maximum(d, 1e-6))
    A = A * d_inv[:, None] * d_inv[None, :]
    return torch.from_numpy(A)  # (J, J)


def build_limb_assignment():
    """(J, L) one-hot membership matrix over L=6 limb groups."""
    limbs = list(LIMB_GROUPS.keys())
    M = np.zeros((NUM_JOINTS, len(limbs)), dtype=np.float32)
    for li, name in enumerate(limbs):
        for j in LIMB_GROUPS[name]:
            M[j, li] = 1.0
    return torch.from_numpy(M), limbs


def sinusoidal_pe(T, D, device):
    pe = torch.zeros(T, D, device=device)
    pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32)
                    * -(math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ---------------------------------------------------------------------------
# Spatial: multi-head adaptive joint-level GCN
# ---------------------------------------------------------------------------

class MultiHopAdaptiveGCN(nn.Module):
    """Joint-level GCN with three complementary adjacency heads.

    Head A (skeleton 1-hop, with learnable residual):
        \\tilde A_static + A_learn   — captures direct kinematic neighbours
    Head B (skeleton 2-hop):
        \\tilde A_static^2           — captures 2-step structural relations
        (e.g. wrist ↔ shoulder)
    Head C (data-driven attention):
        softmax(Q K^T / sqrt(d_k))  — captures pose-dependent affinities
        (e.g. left hand ↔ right hand when crossed)

    Each head has its own value projection; a 2-layer MLP fuses them.
    """

    def __init__(self, dim, A_static, attn_dim=None):
        super().__init__()
        self.register_buffer("A_static", A_static)
        self.register_buffer("A_2hop", A_static @ A_static)
        self.A_learn = nn.Parameter(torch.zeros_like(A_static))

        attn_dim = attn_dim or max(dim // 4, 16)
        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.scale = attn_dim ** -0.5

        self.v_proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(3)])
        self.out = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        # x: (B, T, J, D)
        A_skel = self.A_static + self.A_learn  # (J, J)

        h0 = torch.einsum("ij,btjd->btid", A_skel, self.v_proj[0](x))
        h1 = torch.einsum("ij,btjd->btid", self.A_2hop, self.v_proj[1](x))

        # data-driven adjacency, computed per (batch, time) frame
        q = self.q_proj(x)                            # (B, T, J, dk)
        k = self.k_proj(x)
        attn = (q @ k.transpose(-1, -2)) * self.scale  # (B, T, J, J)
        attn = attn.softmax(dim=-1)
        h2 = torch.einsum("btij,btjd->btid", attn, self.v_proj[2](x))

        return self.out(torch.cat([h0, h1, h2], dim=-1))


# ---------------------------------------------------------------------------
# Spatial: learned limb / body aggregation
# ---------------------------------------------------------------------------

class LearnableLimbPool(nn.Module):
    """Pool joint features to L=6 limb tokens via per-limb softmax attention.

    Each limb has a learnable scalar weight per joint; non-member joints
    are masked out. This lets, e.g., the hand contribute more to the
    "arm" feature than the collar bone, learned from data.
    """
    def __init__(self, M_limb):
        super().__init__()
        J, L = M_limb.shape
        self.J, self.L = J, L
        self.register_buffer("M", M_limb)
        # initialize logits to 0 → uniform within each limb
        self.attn_logits = nn.Parameter(torch.zeros(J, L))

    def forward(self, x):
        # x: (B, T, J, D) → (B, T, L, D)
        logits = self.attn_logits.masked_fill(self.M < 0.5, float("-inf"))
        attn = logits.softmax(dim=0)                       # softmax over J per L
        return torch.einsum("jl,btjd->btld", attn, x)


class QueryPool(nn.Module):
    """Reduce a sequence of N tokens to one via single-query attention."""
    def __init__(self, dim, nhead=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B*, N, D)
        B = x.size(0)
        q = self.q.expand(B, -1, -1)
        z, _ = self.attn(q, x, x, need_weights=False)
        return self.norm(z.squeeze(1))   # (B*, D)


# ---------------------------------------------------------------------------
# Spatial: gated cross-scale fusion
# ---------------------------------------------------------------------------

class CrossScaleGate(nn.Module):
    """Per-joint gated mixture of {joint, limb, body} features.

    The gate is a learned softmax over the three scales, computed from
    the concatenated representation, so each joint adaptively decides
    how much local detail vs. limb context vs. global body context to
    use at this moment.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h_joint, h_limb_b, h_body_b):
        cat = torch.cat([h_joint, h_limb_b, h_body_b], dim=-1)
        g = self.gate(cat).softmax(dim=-1)                  # (B, T, J, 3)
        mix = (g[..., 0:1] * h_joint
             + g[..., 1:2] * h_limb_b
             + g[..., 2:3] * h_body_b)
        return self.proj(mix)


# ---------------------------------------------------------------------------
# Temporal: spectral (DCT) mixer  — Innovation 3
# ---------------------------------------------------------------------------

class SpectralTemporalBlock(nn.Module):
    """Frequency-domain temporal mixer.

    For each (joint, channel) sequence over T frames:
      1. orthonormal type-II DCT along T → spectral coefficients X_k
      2. multiply by a learnable per-frequency × per-channel gain + bias
      3. multiply by a learned soft low-pass mask sigmoid(profile),
         initialised high-low so low frequencies pass and high
         frequencies are damped at start of training (matches the smooth
         periodic motion priors of Dunhuang dance: S-shape sway, held
         postures, slow rhythmic flow)
      4. inverse DCT (B^T) → time-domain residual
      5. residual + LayerNorm + pointwise FFN to remix channels

    Receptive field is global along T with only O(T·D) added parameters,
    versus O(T²) for full attention or limited support for dilated TCN.
    """

    def __init__(self, dim, max_T=512, expand=2):
        super().__init__()
        self.dim = dim
        self.max_T = max_T
        # per-frequency, per-channel gain & bias (init: identity transform)
        self.gain = nn.Parameter(torch.ones(max_T, dim))
        self.bias = nn.Parameter(torch.zeros(max_T, dim))
        # soft low-pass mask: monotone profile high → low across frequencies
        self.lp_logits = nn.Parameter(torch.linspace(3.0, -3.0, max_T))
        # cache DCT bases by (T, device, dtype) — small matrices, cheap
        self._basis_cache = {}

        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand), nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def _dct_basis(self, T, device, dtype):
        key = (T, device, dtype)
        B = self._basis_cache.get(key)
        if B is not None:
            return B
        n = torch.arange(T, device=device, dtype=dtype).unsqueeze(0)  # (1, T)
        k = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
        B = torch.cos(math.pi * (2 * n + 1) * k / (2 * T))            # (T, T)
        # orthonormal scaling: row 0 = 1/sqrt(T), other rows = sqrt(2/T)
        scale = torch.full((T, 1), math.sqrt(2.0 / T), device=device, dtype=dtype)
        scale[0, 0] = math.sqrt(1.0 / T)
        B = B * scale
        self._basis_cache[key] = B
        return B

    def forward(self, x):
        # x: (B, T, J, D)
        B, T, J, D = x.shape
        assert T <= self.max_T, f"T={T} exceeds SpectralTemporalBlock.max_T={self.max_T}"
        Bmat = self._dct_basis(T, x.device, x.dtype)                   # (T, T)

        # forward DCT: X_f[k] = sum_t Bmat[k,t] * x[t]
        x_perm = x.permute(0, 2, 3, 1)                                 # (B, J, D, T)
        Xf = torch.einsum("kt,bjdt->bjdk", Bmat, x_perm)               # (B, J, D, T) freq last

        # per-frequency, per-channel modulation + low-pass mask
        g = self.gain[:T].t()                                          # (D, T)
        b = self.bias[:T].t()                                          # (D, T)
        lp = torch.sigmoid(self.lp_logits[:T])                         # (T,)
        Xf = Xf * g + b
        Xf = Xf * lp                                                   # broadcasts to (B,J,D,T)

        # inverse DCT: x_t = sum_k Bmat[k,t] * Xf[k] = (Bmat^T) @ Xf
        x_back = torch.einsum("kt,bjdk->bjdt", Bmat, Xf)
        x_back = x_back.permute(0, 3, 1, 2)                            # (B, T, J, D)

        h = self.norm(x + x_back)
        h = h + self.ffn(h)
        return h


# ---------------------------------------------------------------------------
# HST-GCN block
# ---------------------------------------------------------------------------

class HSTGCNBlock(nn.Module):
    """One block: spatial (joint→limb→body, gated fuse) + spectral time + FFN."""

    def __init__(self, dim, adj_joint, M_limb, nhead=4, max_T=512):
        super().__init__()
        self.J, self.L = M_limb.shape
        self.register_buffer("M", M_limb)

        self.joint_gcn = MultiHopAdaptiveGCN(dim, adj_joint)

        self.limb_pool = LearnableLimbPool(M_limb)
        self.limb_proc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.body_pool = QueryPool(dim, nhead=nhead)
        self.body_proc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        self.fuse = CrossScaleGate(dim)
        self.norm1 = nn.LayerNorm(dim)

        self.spectral = SpectralTemporalBlock(dim, max_T=max_T)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T, J, D)
        B, T, J, D = x.shape

        # joint-level adaptive GCN
        h_j = self.joint_gcn(x)                                 # (B, T, J, D)

        # limb pooling (learned per-limb attention) + per-limb MLP
        h_l = self.limb_pool(x)                                 # (B, T, L, D)
        h_l = h_l + self.limb_proc(h_l)

        # body pooling (single learnable query attending over L limb tokens)
        h_b = self.body_pool(h_l.reshape(B * T, self.L, D)).reshape(B, T, 1, D)
        h_b = h_b + self.body_proc(h_b)

        # broadcast both back to joint shape
        limb_b = torch.einsum("jl,btld->btjd", self.M, h_l)     # (B, T, J, D)
        body_b = h_b.expand(-1, -1, J, -1)                       # (B, T, J, D)

        # gated cross-scale fuse + residual
        fused = self.fuse(h_j, limb_b, body_b)
        x = self.norm1(x + fused)

        # spectral (DCT) temporal mixing — replaces dilated TCN
        x = self.spectral(x)

        # FFN with residual via norm2
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# Style encoder (multi-layer)
# ---------------------------------------------------------------------------

class StyleEncoder(nn.Module):
    """Stack of attention-pool layers producing a global style vector.

    Each layer: K learnable queries cross-attend to per-frame features
    (visible frames only), then a per-query FFN. The final K output
    tokens are concatenated and projected through a small MLP head.
    """

    def __init__(self, dim, num_queries=4, nhead=4, depth=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(dim, nhead, batch_first=True,
                                              dropout=dropout),
                "norm1": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                ),
                "norm2": nn.LayerNorm(dim),
            })
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(dim * num_queries),
            nn.Linear(dim * num_queries, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h, visible_mask):
        # h: (B, T, D); visible_mask: (B, T) True = visible
        B = h.size(0)
        z = self.queries.unsqueeze(0).expand(B, -1, -1)
        kpm = ~visible_mask
        all_hidden = kpm.all(dim=1)
        if all_hidden.any():
            kpm = kpm.clone()
            kpm[all_hidden] = False

        for layer in self.layers:
            attn_out, _ = layer["attn"](z, h, h,
                                        key_padding_mask=kpm, need_weights=False)
            z = layer["norm1"](z + attn_out)
            z = layer["norm2"](z + layer["ffn"](z))

        return self.head(z.reshape(B, -1))   # (B, D)


# ---------------------------------------------------------------------------
# Style → time conditioning (FiLM)
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Affine modulation: target = target * (1 + γ) + β where (γ, β) = f(style)."""
    def __init__(self, dim_style, dim_target):
        super().__init__()
        self.to_gb = nn.Sequential(
            nn.LayerNorm(dim_style),
            nn.Linear(dim_style, dim_target * 2),
        )

    def forward(self, target, style):
        # target: (B, T, D); style: (B, D)
        gb = self.to_gb(style)
        gamma, beta = gb.chunk(2, dim=-1)
        return target * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


# ---------------------------------------------------------------------------
# Frame pooling (attention over joints)
# ---------------------------------------------------------------------------

class FramePool(nn.Module):
    """Per-frame, query-attention pool over the J joint tokens."""
    def __init__(self, dim, nhead=4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, T, J, D) → (B, T, D)
        B, T, J, D = x.shape
        q = self.q.expand(B * T, -1, -1)
        kv = x.reshape(B * T, J, D)
        z, _ = self.attn(q, kv, kv, need_weights=False)
        return self.norm(z.squeeze(1)).reshape(B, T, D)


# ---------------------------------------------------------------------------
# Temporal transformer
# ---------------------------------------------------------------------------

class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth=2, nhead=4, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4,
            batch_first=True, activation="gelu", dropout=dropout,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.enc(x)


# ---------------------------------------------------------------------------
# Diffusion decoder — Innovation 4
# ---------------------------------------------------------------------------

def _cosine_alpha_bar(n_steps, s=0.008):
    """Nichol & Dhariwal (2021) cosine α̅ schedule, length n_steps."""
    steps = torch.arange(n_steps + 1, dtype=torch.float64)
    f = torch.cos((steps / n_steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(min=1e-8, max=1.0)[:n_steps]
    return ab.float()


class _ResFiLMBlock(nn.Module):
    """LayerNorm → FiLM(γ, β) → MLP, with residual."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, hidden)
        self.lin2 = nn.Linear(hidden, dim)

    def forward(self, x, gamma, beta):
        h = self.norm(x)
        h = h * (1.0 + gamma) + beta
        h = F.gelu(self.lin1(h))
        h = self.lin2(h)
        return x + h


class DiffusionDecoder(nn.Module):
    """Conditional denoising decoder with x-prediction parameterisation.

    Trained by sampling t ∈ [0, n_steps), corrupting the ground-truth
    coordinates at masked frames with Gaussian noise according to the
    cosine α̅ schedule, and asking the network to reconstruct the clean
    coordinates given:
      (a) per-(frame, joint) backbone features  (B, T, J, D)
      (b) the noisy coordinates                 (B, T, J, 2)
      (c) a sinusoidal embedding of t           (B,)  → FiLM(γ, β)

    Inference: K-step DDIM (default K=8) starting from Gaussian noise at
    masked frames; visible frames are clamped to their input value at
    every step so the conditioning is exact.

    x-prediction (not ε-prediction) keeps the loss in the same units as
    the project's existing reconstruction L1, and is more stable for
    low-dimensional bounded targets like 2D coordinates.
    """

    def __init__(self, dim, n_steps=200, hidden_mult=2):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.register_buffer("alpha_bar", _cosine_alpha_bar(n_steps))

        # noisy-coord encoder
        self.x_proj = nn.Linear(2, dim)

        # timestep embedding → FiLM (γ, β)
        self.t_to_gb = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(),
            nn.Linear(dim * 2, dim * 2),
        )

        # denoising MLP (FiLM-conditioned)
        h = dim * hidden_mult
        self.block1 = _ResFiLMBlock(dim, h)
        self.block2 = _ResFiLMBlock(dim, h)
        self.out_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def _t_embed(self, t):
        # t: (B,) long → (B, dim) sinusoidal
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb

    def predict_x0(self, x_t, cond, t):
        """x_t: (B, T, J, 2); cond: (B, T, J, D); t: (B,) long → x0_pred (B, T, J, 2)."""
        B = x_t.size(0)
        h = self.x_proj(x_t) + cond
        gb = self.t_to_gb(self._t_embed(t))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.view(B, 1, 1, -1)
        beta = beta.view(B, 1, 1, -1)
        h = self.block1(h, gamma, beta)
        h = self.block2(h, gamma, beta)
        return self.head(self.out_norm(h))

    def add_noise(self, x0, t, noise=None):
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, cond, x_visible, mask, K=8):
        """K-step DDIM (η=0). visible frames stay clamped to x_visible.

        cond: (B, T, J, D)   x_visible: (B, T, J, 2)   mask: (B, T) bool, True=hidden
        Returns the final clean x0 estimate (B, T, J, 2).
        """
        B, T, J, _ = cond.shape
        device = cond.device
        ts = torch.linspace(self.n_steps - 1, 0, K + 1, device=device).round().long()
        m = mask.float().view(B, T, 1, 1)

        # init: noise at masked positions, GT at visible
        x_t = torch.randn(B, T, J, 2, device=device)
        x_t = m * x_t + (1.0 - m) * x_visible

        for i in range(K):
            t = ts[i].repeat(B)
            t_next = ts[i + 1].repeat(B)
            ab = self.alpha_bar[t].view(B, 1, 1, 1)
            ab_n = self.alpha_bar[t_next].view(B, 1, 1, 1)
            x0 = self.predict_x0(x_t, cond, t)
            eps = (x_t - ab.sqrt() * x0) / (1.0 - ab).sqrt().clamp_min(1e-6)
            x_t = ab_n.sqrt() * x0 + (1.0 - ab_n).sqrt() * eps
            x_t = m * x_t + (1.0 - m) * x_visible
        # final t_next = 0 has α̅ ≈ 1, so x_t ≈ x0 at masked frames, exactly
        # x_visible at visible frames (clamped above).
        return x_t


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class DunhuangMotionModel(nn.Module):
    def __init__(self, dim=128, hst_blocks=3, trf_depth=2, nhead=4,
                 use_style_clf=False, num_families=NUM_FAMILIES,
                 diff_steps=200, ddim_steps=8, max_T=512):
        super().__init__()
        self.use_style_clf = use_style_clf
        self.num_families = num_families
        self.ddim_steps = ddim_steps
        adj = build_joint_adjacency()
        M_limb, _ = build_limb_assignment()

        # Embedding
        self.input_proj = nn.Sequential(
            nn.Linear(2, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, dim),
        )
        self.joint_embed = nn.Parameter(torch.randn(NUM_JOINTS, dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(dim) * 0.02)

        # Spatial-temporal backbone
        self.hst_blocks = nn.ModuleList(
            [HSTGCNBlock(dim, adj, M_limb, nhead=nhead, max_T=max_T) for _ in range(hst_blocks)]
        )

        # Per-frame pooling for the temporal transformer / style branch
        self.frame_pool = FramePool(dim, nhead=nhead)

        # Style-rhythm disentanglement
        self.style_enc = StyleEncoder(dim, num_queries=4, nhead=nhead, depth=2)
        self.film = FiLM(dim, dim)

        # Long-range temporal transformer over per-frame features
        self.temporal = TemporalTransformer(dim, depth=trf_depth, nhead=nhead)

        # Diffusion decoder
        self.decoder = DiffusionDecoder(dim, n_steps=diff_steps)

        # Optional family classifier head on the style vector
        if use_style_clf:
            self.style_clf = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dim, num_families),
            )
        else:
            self.style_clf = None

    # ------------------------------------------------------------------
    # backbone
    # ------------------------------------------------------------------
    def encode(self, x, mask):
        """Spatial-temporal backbone → (cond (B,T,J,D), style (B,D))."""
        B, T, J, _ = x.shape
        device = x.device

        h_in = self.input_proj(x)                                    # (B, T, J, D)
        mask_f = mask.float().unsqueeze(-1).unsqueeze(-1)            # (B, T, 1, 1)
        h_in = h_in * (1.0 - mask_f) + self.mask_token.view(1, 1, 1, -1) * mask_f
        h = h_in + self.joint_embed
        h = h + sinusoidal_pe(T, h.size(-1), device).view(1, T, 1, -1)

        for blk in self.hst_blocks:
            h = blk(h)                                               # (B, T, J, D)

        h_frame = self.frame_pool(h)                                 # (B, T, D)
        visible = ~mask
        style = self.style_enc(h_frame, visible)                     # (B, D)
        h_time = self.temporal(self.film(h_frame, style))            # (B, T, D)

        cond = h + h_time.unsqueeze(2)                               # (B, T, J, D)
        return cond, style

    # ------------------------------------------------------------------
    # inference  — DDIM sampling, returns "pred"
    # ------------------------------------------------------------------
    def forward(self, x, mask):
        """x: (B, T, 27, 2)  mask: (B, T) bool, True = hidden.

        Runs the diffusion decoder via K-step DDIM and returns the
        completed motion. Same return-dict shape as the previous model.
        """
        cond, style = self.encode(x, mask)
        pred = self.decoder.sample(cond, x, mask, K=self.ddim_steps)
        out = {"pred": pred, "style": style}
        if self.style_clf is not None:
            out["family_logits"] = self.style_clf(style)
        return out

    # ------------------------------------------------------------------
    # training  — single noisy step, predicts clean x0
    # ------------------------------------------------------------------
    def forward_train(self, x, mask):
        """One diffusion training step. Returns dict with x0-prediction at "pred"."""
        B = x.size(0)
        cond, style = self.encode(x, mask)

        t = torch.randint(0, self.decoder.n_steps, (B,), device=x.device)
        x_t, _ = self.decoder.add_noise(x, t)
        # only denoise hidden frames; visible frames stay clean to feed the decoder GT context
        m = mask.float().view(B, -1, 1, 1)
        x_t = m * x_t + (1.0 - m) * x

        x0_pred = self.decoder.predict_x0(x_t, cond, t)
        out = {"pred": x0_pred, "style": style, "t": t}
        if self.style_clf is not None:
            out["family_logits"] = self.style_clf(style)
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def diffusion_x0_loss(pred, target, mask):
    """L1 between x0-prediction and ground-truth coords; hidden + small visible.

    Same shape contract as the previous masked_reconstruction_loss so
    train.py callsites need no change beyond the import alias.

    pred/target: (B, T, J, 2)
    mask: (B, T) bool — True = hidden
    """
    diff = (pred - target).abs().mean(dim=(-1, -2))                  # (B, T)
    mask_f = mask.float()
    vis_f = 1.0 - mask_f
    hid = (diff * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    vis = (diff * vis_f).sum() / vis_f.sum().clamp_min(1.0)
    return hid + 0.1 * vis, {"loss_hidden": hid.detach(), "loss_visible": vis.detach()}


# Backward-compat alias so older training scripts keep importing the same name.
masked_reconstruction_loss = diffusion_x0_loss