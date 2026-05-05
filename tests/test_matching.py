"""Tests for cosine matching and argmax correspondence helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.common.matching import (
    argmax_to_pixel_xy,
    match_cosine_similarity_map,
    predict_correspondences_cosine_argmax,
    sample_features_bilinear,
)


def test_match_cosine_similarity_map_shape():
    feat_pts = F.normalize(torch.randn(3, 8), dim=-1)
    feat_tgt = F.normalize(torch.randn(1, 8, 4, 4), dim=1)
    smaps = match_cosine_similarity_map(feat_pts, feat_tgt)
    assert smaps.shape == (3, 4, 4)


def test_argmax_to_pixel_xy_single_map():
    sm = torch.zeros(4, 4)
    sm[1, 3] = 1.0
    out = argmax_to_pixel_xy(sm, img_hw=(8, 8), feat_hw=(4, 4))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_sample_features_bilinear_center():
    c, h, w = 4, 4, 4
    feat = torch.arange(float(c * h * w)).view(1, c, h, w)
    feat = F.normalize(feat, dim=1)
    xy = torch.tensor([[1.5, 1.5]], dtype=torch.float32)
    samp = sample_features_bilinear(feat, xy, img_hw=(4, 4))
    assert samp.shape == (1, c)


def test_predict_correspondences_invalid_mask():
    c, hf, wf = 8, 4, 4
    fs = F.normalize(torch.randn(1, c, hf, wf), dim=1)
    ft = F.normalize(torch.randn(1, c, hf, wf), dim=1)
    kps = torch.zeros(2, 2)
    mask = torch.tensor([False, False])
    out = predict_correspondences_cosine_argmax(
        fs, ft, kps, img_hw=(16, 16), valid_mask=mask
    )
    assert out["pred_tgt_xy"].shape == (2, 2)
    assert (out["sim_maps"] == 0).all()
