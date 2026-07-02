"""Lightweight smoke test for the HSGD model module.

The repository does not ship dataset-specific skeleton metadata. This script
injects a tiny metadata module so the model can be instantiated before users
create src/hsgd/bvh_utils.py from the example template.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def install_demo_bvh_utils() -> None:
    module = types.ModuleType("bvh_utils")
    module.NUM_JOINTS = 27
    module.NUM_FAMILIES = 7
    module.PARENTS = [
        -1, 0, 1, 2, 3, 0, 5, 6, 7,
        0, 9, 10, 11, 0, 13, 14, 15,
        0, 17, 18, 19, 0, 21, 22, 23,
        0, 25,
    ]
    module.LIMB_GROUPS = {
        "torso": [0, 1, 2, 3, 4],
        "left_arm": [5, 6, 7, 8],
        "right_arm": [9, 10, 11, 12],
        "left_leg": [13, 14, 15, 16],
        "right_leg": [17, 18, 19, 20],
        "head": [21, 22, 23, 24, 25, 26],
    }
    sys.modules["bvh_utils"] = module


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    install_demo_bvh_utils()

    import torch
    from hsgd.model import DunhuangMotionModel

    model = DunhuangMotionModel(dim=32, hst_blocks=1, trf_depth=1, nhead=4, diff_steps=8, ddim_steps=2)
    x = torch.randn(1, 12, 27, 2)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[:, 8:] = True

    with torch.no_grad():
        out = model(x, mask)

    print("pred", tuple(out["pred"].shape))


if __name__ == "__main__":
    main()

