"""
Smoke tests for the SD4Match portability layer.

These tests avoid requiring the dataset to be present; they only validate:
- imports of vendored SD4Match subset
- configuration objects can be constructed
- metric wrappers produce expected key shapes on toy inputs
"""

from __future__ import annotations

import torch

from data.interface import DatasetConfig, RuntimeConfig
from evaluation.sd4match_metrics import default_alphas_pdf, evaluate_matches_sd4match


def test_default_alphas_pdf():
    assert default_alphas_pdf() == (0.05, 0.1, 0.2)


def test_evaluate_matches_sd4match_shapes():
    # B=2, N=4 (last points treated as padding via n_pts)
    B, N = 2, 4
    trg_kps = torch.tensor(
        [
            [[10.0, 10.0], [20.0, 20.0], [-2.0, -2.0], [-2.0, -2.0]],
            [[5.0, 5.0], [8.0, 8.0], [9.0, 9.0], [-2.0, -2.0]],
        ]
    )
    matches = trg_kps.clone()
    n_pts = torch.tensor([2, 3])
    categories = ["cat", "dog"]
    pckthres = torch.tensor([100.0, 50.0])

    out = evaluate_matches_sd4match(
        trg_kps=trg_kps,
        matches=matches,
        n_pts=n_pts,
        categories=categories,
        pckthres=pckthres,
        alphas=default_alphas_pdf(),
    )

    # Expect both granularities present and summary dicts exist.
    assert "custom_pck0.05" in out.per_image
    assert "custom_pck0.05" in out.per_point
    assert "image" in out.summary and "point" in out.summary


def test_sd4match_config_objects_constructible():
    ds = DatasetConfig(backend="sd4match", name="spair", root="data")
    rt = RuntimeConfig(preprocess="FIXED_RESIZE", image_height=784, image_width=784, num_workers=-1)
    assert ds.backend == "sd4match"
    assert rt.image_height == 784

