# Spectral Graph Diffusion for Style-Preserving Skeletal Motion Completion

Official implementation of **Hierarchical Spectral Graph Diffusion (HSGD)** for style-preserving skeletal motion completion on Dunhuang dance.

> **Bin Niu**, **Yitong Wang**, and **Rui Yang**  
> *IEEE Signal Processing Letters*

## Overview

Skeletal motion completion recovers missing motion segments from partially observed joint sequences. This repository presents HSGD, a framework specifically designed for traditional artistic dances like Dunhuang dance, where preserving structural plausibility, temporal continuity, and contextual consistency is critical.

### Key Features

- **Unified Masked Formulation**: Handles motion prediction, interpolation, and random missing-frame completion in a unified framework
- **Hierarchical Spectral Encoding**: Combines multi-scale body structure modeling (joint, limb, body levels) with frequency-domain temporal spectral modulation using DCT
- **Style-Rhythm Conditioning**: Preserves dance style and rhythmic patterns through style-rhythm conditioned diffusion decoding
- **Strong Performance**: 
  - 5.4% and 8.3% MAE reduction over strongest baseline in prediction and interpolation
  - 8.0% and 9.0% improvement in random missing-frame completion
  - High style preservation (88.6% Family Consistency on Dunhuang-Video)

## Architecture

The HSGD framework consists of four main components:

### 1. **Unified Masked Motion Modeling**
- Represents visible frames with coordinate projections and unknown frames with learnable mask tokens
- Includes joint embeddings and temporal positional encodings
- Enables unified handling of different observation modes (prediction, interpolation, random completion)

### 2. **Hierarchical Spectral Spatio-Temporal Encoder**
- **Joint-level**: Multi-branch adaptive graph convolution fusing first-order, second-order, and dynamic adjacencies
- **Limb-level**: Progressive pooling from joint features
- **Body-level**: Global posture representation via query pooling
- **Cross-scale Fusion**: Gated mechanism to fuse representations across scales
- **Temporal Spectral Modulation**: DCT-based frequency domain processing with learnable frequency gains and low-pass filtering

### 3. **Style-Rhythm Conditioned Representation**
- Extracts style vector exclusively from visible frames using attention pooling
- Applies FiLM modulation to inject style into frame-level representations
- Ensures recovered motion remains stylistically consistent with observed context

### 4. **Conditional Diffusion Motion Decoder**
- Iterative denoising from Gaussian noise at masked positions
- Visible frame clamping during inference for context consistency
- Direct $x_0$ prediction parameterization for coordinate reconstruction

## Datasets

HSGD is evaluated on two public 3D Dunhuang dance motion datasets:

| Dataset | Description | Scale |
|---------|-------------|-------|
| **Chang-E** | Optical motion-capture | 8 categories, ~40 minutes |
| **Dunhuang-Video** | Monocular video reconstruction | 7 themes, 99 actions, 26,220 frames |

Both datasets provide 30 fps BVH sequences retargeted to a unified 27-joint skeleton, converted to front-view 2D normalized coordinates.

## Experimental Results

### Motion Prediction & Interpolation (Table 1)

| Method | Dunhuang-Video (Avg.) | Chang-E (Avg.) |
|--------|----------------------|----------------|
| **HSGD (ours)** | **0.0847** | **0.0791** |
| HumanMAC | 0.0895 | 0.0863 |
| ST-GraphFormer | 0.0937 | 0.0886 |
| ST-GCN+Trans. | 0.0927 | 0.0894 |

### Random Missing-Frame Completion (40% masked, Table 2)

| Method | Dunhuang-Video | Chang-E |
|--------|----------------|---------|
| **HSGD (ours)** | **0.0642** | **0.0589** |
| HumanMAC | 0.0698 | 0.0647 |
| ST-GCN+Trans. | 0.0741 | 0.0682 |

### Ablation Study (Dunhuang-Video)

| Component | Impact (Δ MAE) | Family Consistency |
|-----------|----------------|-------------------|
| Full Model | 0.0000 | 88.6% |
| w/o unified mask | +0.0307 | 80.4% |
| w/o hierarchical graph | +0.0318 | 78.9% |
| w/o frequency module | +0.0243 | 83.5% |
| w/o style encoder | +0.0136 | 76.8% |
| w/o diffusion decoder | +0.0279 | 79.6% |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU acceleration)

## Installation

```bash
git clone https://github.com/BinNiu-Dance/hsgd-dunhuang-dance.git
cd hsgd-dunhuang-dance
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/hsgd.yaml --dataset dunhuang_video
```

**Training Configuration:**
- GPU: NVIDIA RTX 3090 (single GPU)
- Input length: T=60 (2.0 seconds at 30 fps)
- Joints: J=27, Coordinates: C=2
- Hidden size: 128
- Encoder blocks: 3 hierarchical spectral graph blocks
- Epochs: 200
- Batch size: 16
- Learning rate: 3×10⁻⁴
- Diffusion steps: 200 (training), 8 (inference)

### Inference

```bash
python inference.py --model_path checkpoint.pth --task prediction
```

Supported tasks: `prediction`, `interpolation`, `random_completion`

## Citation

If you use this work, please cite:

```bibtex
@article{niu2024spectral,
  title={Spectral Graph Diffusion for Style-Preserving Skeletal Motion Completion},
  author={Niu, Bin and Wang, Yitong and Yang, Rui},
  journal={IEEE Signal Processing Letters},
  year={2024}
}
```

## Acknowledgments

This work was supported by the Humanities and Social Sciences Research Project of the Ministry of Education of China under Grant 25XJC760005.

**Authors:**
- Bin Niu, School of Dance, Northwest Normal University, Lanzhou, China
- Yitong Wang, School of Dance, Northwest Normal University, Lanzhou, China
- Rui Yang, School of Information Science and Engineering, Lanzhou University, Lanzhou, China

## License

[Add your license here]

## Contact

For questions or inquiries, please contact:
- Bin Niu: bin_niu@nwnu.edu.cn
- Rui Yang: rykeryang@163.com
