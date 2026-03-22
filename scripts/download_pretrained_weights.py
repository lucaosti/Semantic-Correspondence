#!/usr/bin/env python3
"""
Download official pretrained weights for DINOv2 ViT-B/14, DINOv3 ViT-B/16, and SAM ViT-B
into ``checkpoints/``.

URLs match ``models/dinov2/hub_loader.py`` and ``models/dinov3/hub_loader.py`` (same as
``torch.hub.load_state_dict_from_url`` used during training). SAM uses
``scripts/download_sam_vit_b.sh``.

Usage (from repository root)::

   python scripts/download_pretrained_weights.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch

    from models.dinov2.hub_loader import _DINOV2_BASE_URL, _make_dinov2_model_name
    from models.dinov3.hub_loader import Weights, _make_dinov3_vit_model_url

    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    mb = _make_dinov2_model_name("vit_base", 14, 0)
    url_d2 = f"{_DINOV2_BASE_URL}/{mb}/{mb}_pretrain.pth"
    dest_d2 = ckpt / "dinov2_vitb14_pretrain.pth"
    if not dest_d2.is_file():
        print(f"Downloading DINOv2: {url_d2}")
        torch.hub.download_url_to_file(url_d2, str(dest_d2))
    else:
        print(f"Already present: {dest_d2}")

    url_d3 = _make_dinov3_vit_model_url(
        patch_size=16,
        compact_arch_name="vitb",
        version=None,
        weights=Weights.LVD1689M,
        hash="73cec8be",
    )
    dest_d3 = ckpt / os.path.basename(url_d3.replace("\\", "/"))
    if not dest_d3.is_file():
        print(f"Downloading DINOv3: {url_d3}")
        torch.hub.download_url_to_file(url_d3, str(dest_d3))
    else:
        print(f"Already present: {dest_d3}")

    sam_script = ROOT / "scripts" / "download_sam_vit_b.sh"
    subprocess.run(["bash", str(sam_script)], cwd=str(ROOT), check=True)
    print(f"All pretrained weights under {ckpt}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
