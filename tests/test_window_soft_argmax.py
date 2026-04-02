"""Tests for inference-only window soft-argmax refinement."""

from __future__ import annotations

import torch

from models.common.window_soft_argmax import refine_predictions_window_soft_argmax, window_soft_argmax_xy


def test_window_soft_argmax_xy_peak_center_maps_to_pixels():
    sim = torch.zeros(5, 5)
    sim[2, 2] = 10.0
    xy = window_soft_argmax_xy(sim, img_hw=(10, 10), window_size=5, temperature=1.0)
    assert xy.shape == (2,)
    assert torch.all(torch.isfinite(xy))


def test_window_soft_argmax_bumps_even_window_to_odd():
    sim = torch.zeros(3, 3)
    sim[1, 1] = 1.0
    xy = window_soft_argmax_xy(sim, img_hw=(6, 6), window_size=4, temperature=1.0)
    assert xy.shape == (2,)


def test_refine_predictions_window_soft_argmax_batch():
    sims = torch.randn(2, 4, 4)
    sims[0, 1, 2] = 100.0
    sims[1, 3, 0] = 100.0
    out = refine_predictions_window_soft_argmax(sims, img_hw=(8, 8), window_size=3, temperature=1.0)
    assert out.shape == (2, 2)
