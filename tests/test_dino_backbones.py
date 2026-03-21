"""Smoke tests for DINOv2/DINOv3 builders (no pretrained weight download)."""

from __future__ import annotations

from models.dinov2.backbone import build_dinov2_vit_b14
from models.dinov3.backbone import build_dinov3_vit_b16


def test_build_dinov2_vitb14_no_weights():
    m = build_dinov2_vit_b14(pretrained=False)
    assert m.__class__.__name__ == "DinoVisionTransformer"


def test_build_dinov3_vitb16_no_weights():
    m = build_dinov3_vit_b16(pretrained=False)
    assert m.__class__.__name__ == "DinoVisionTransformer"
