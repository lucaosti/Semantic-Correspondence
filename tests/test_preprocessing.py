"""Tests for SPair preprocessing and keypoint geometry helpers."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from data.dataset import PreprocessMode, preprocess_pair_images_and_keypoints, scale_keypoints_xy


def test_scale_keypoints_xy_resize():
    kps = torch.tensor([[10.0, 20.0]], dtype=torch.float32)
    out = scale_keypoints_xy(kps, orig_size_xy=(100, 200), new_size_xy=(50, 100))
    assert out[0, 0].item() == 5.0
    assert out[0, 1].item() == 10.0


def test_preprocess_fixed_resize_keypoints_scaled():
    src = Image.new("RGB", (100, 80))
    tgt = Image.new("RGB", (100, 80))
    sk = torch.tensor([[50.0, 40.0]], dtype=torch.float32)
    tk = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
    s2, t2, sk2, tk2, meta = preprocess_pair_images_and_keypoints(
        src,
        tgt,
        sk,
        tk,
        mode=PreprocessMode.FIXED_RESIZE,
        patch_size=14,
        fixed_size_hw=(784, 784),
    )
    assert s2.size == (784, 784)
    assert sk2.shape == sk.shape
    assert sk2[0, 0].item() == pytest.approx(50.0 * 784.0 / 100.0)
    assert sk2[0, 1].item() == pytest.approx(40.0 * 784.0 / 80.0)
    assert meta["mode"] == str(PreprocessMode.FIXED_RESIZE)
