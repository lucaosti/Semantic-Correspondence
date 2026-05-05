"""SAM image encoder smoke test (random init, no checkpoint)."""

from __future__ import annotations

import torch

from models.sam.backbone import build_sam_vit_b_image_encoder, extract_dense_grid_sam


def test_sam_encoder_forward_no_checkpoint():
    enc = build_sam_vit_b_image_encoder(checkpoint_path=None)
    x = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        feats, meta = extract_dense_grid_sam(enc, x)
    assert feats.dim() == 4
    assert meta["backbone"] == "sam_vit_b"
