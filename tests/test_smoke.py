"""Lightweight import and shape checks (no dataset or external backbone repos required)."""

from __future__ import annotations

import torch

from training.engine import correspondence_gaussian_loss_dino_vit, correspondence_gaussian_loss_sam
from training.losses import gaussian_ce_loss_from_similarity_maps, pixel_xy_to_feat_xy


def test_pixel_xy_to_feat_xy_shape():
    xy = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    out = pixel_xy_to_feat_xy(xy, img_hw=(100, 200), feat_hw=(10, 20))
    assert out.shape == (2, 2)


def test_gaussian_ce_scalar():
    sim = torch.randn(3, 8, 8)
    gt = torch.tensor([[4.0, 4.0], [2.0, 2.0], [1.0, 1.0]])
    loss = gaussian_ce_loss_from_similarity_maps(sim, gt, img_hw=(64, 64), sigma_feat=1.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_engine_exports():
    assert callable(correspondence_gaussian_loss_dino_vit)
    assert callable(correspondence_gaussian_loss_sam)
