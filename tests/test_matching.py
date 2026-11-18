"""Tests for coordinate conventions, descriptor sampling and argmax matching."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.common.coord_utils import rescale_coords, rescale_keypoints_xy
from models.common.matching import (
    argmax_to_pixel_xy,
    predict_correspondences_cosine_argmax,
    sample_descriptors,
    sample_features_bilinear,
    similarity_maps,
)


def _cell_centre_pixels(hf: int, wf: int, img_hw):
    """Pixel coordinates of every feature-cell centre, ``(Hf*Wf, 2)`` in row-major order."""
    yi, xi = torch.meshgrid(torch.arange(hf, dtype=torch.float32), torch.arange(wf, dtype=torch.float32), indexing="ij")
    x = rescale_coords(xi.reshape(-1), wf, img_hw[1])
    y = rescale_coords(yi.reshape(-1), hf, img_hw[0])
    return torch.stack([x, y], dim=-1)


@pytest.mark.parametrize("hf,wf,img_hw", [(37, 37, (518, 518)), (32, 32, (512, 512)), (5, 7, (40, 90))])
def test_sampling_at_cell_centre_returns_that_cell_exactly(hf, wf, img_hw):
    # Regression: the old (x / (Wf-1)) normalisation with align_corners=False sampled up to
    # half a cell away from the keypoint near the borders.
    feats = torch.randn(1, 6, hf, wf)
    pts = _cell_centre_pixels(hf, wf, img_hw)
    desc = sample_descriptors(feats, pts.unsqueeze(0), img_hw)[0]  # (Hf*Wf, C)
    expected = feats[0].reshape(6, -1).T
    assert torch.allclose(desc, expected, atol=1e-4)


def test_sampling_between_two_cells_is_their_average():
    feats = torch.zeros(1, 1, 1, 4)
    feats[0, 0, 0, 1] = 2.0
    feats[0, 0, 0, 2] = 4.0
    # img width 8 -> cell i centre at pixel 2i + 0.5; halfway between cells 1 and 2 is x=3.5
    desc = sample_descriptors(feats, torch.tensor([[[3.5, 0.5]]]), (2, 8))
    assert desc.item() == pytest.approx(3.0)


def test_argmax_round_trip_lands_on_source_cell():
    torch.manual_seed(0)
    hf, wf, img_hw = 6, 9, (60, 90)
    feats = F.normalize(torch.randn(1, 16, hf, wf), dim=1)
    pts = _cell_centre_pixels(hf, wf, img_hw)
    desc = sample_descriptors(feats, pts.unsqueeze(0), img_hw)
    pred = argmax_to_pixel_xy(similarity_maps(desc, feats), img_hw)[0]
    assert torch.allclose(pred, pts, atol=1e-4)


def test_similarity_maps_stay_fp32_under_autocast():
    feats = F.normalize(torch.randn(1, 8, 4, 4), dim=1)
    desc = F.normalize(torch.randn(1, 3, 8), dim=-1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        sims = similarity_maps(desc, feats)
    assert sims.dtype == torch.float32
    assert torch.allclose(sims, torch.einsum("bnc,bchw->bnhw", desc, feats), atol=1e-6)


def test_rescale_keypoints_round_trip_and_pixel_centres():
    xy = torch.tensor([[0.0, 0.0], [499.0, 374.0], [123.4, 56.7]])
    there = rescale_keypoints_xy(xy, (375, 500), (518, 518))
    back = rescale_keypoints_xy(there, (518, 518), (375, 500))
    assert torch.allclose(back, xy, atol=1e-4)
    # The edges of the image map to the edges: pixel 0's left edge (-0.5) stays at -0.5.
    edge = rescale_keypoints_xy(torch.tensor([[-0.5, -0.5]]), (375, 500), (518, 518))
    assert torch.allclose(edge, torch.tensor([[-0.5, -0.5]]))


def test_sample_features_bilinear_single_image_wrapper():
    feat = torch.randn(1, 4, 4, 4)
    out = sample_features_bilinear(feat, torch.tensor([[1.0, 1.0]]), img_hw=(4, 4))
    assert torch.allclose(out[0], feat[0, :, 1, 1])


def test_predict_correspondences_invalid_mask():
    fs = F.normalize(torch.randn(1, 8, 4, 4), dim=1)
    ft = F.normalize(torch.randn(1, 8, 4, 4), dim=1)
    out = predict_correspondences_cosine_argmax(
        fs, ft, torch.zeros(2, 2), img_hw=(16, 16), valid_mask=torch.tensor([False, False])
    )
    assert out["pred_tgt_xy"].shape == (2, 2)
    assert (out["sim_maps"] == 0).all()
