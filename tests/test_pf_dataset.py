"""Tests for the PF-Willow / PF-Pascal loaders on synthetic data in the official CSV formats."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
from PIL import Image

from data.dataset import INVALID_KP_COORD, MAX_KEYPOINTS, spair_collate_fn
from data.pf_dataset import PF_PASCAL_CATEGORIES, PFPascalPairDataset, PFWillowPairDataset


def _img(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), color=(128, 64, 32)).save(path)


@pytest.fixture()
def pf_willow_root(tmp_path: Path) -> Path:
    a, b = "PF-dataset/car(S)/a.png", "PF-dataset/car(S)/b.png"
    _img(tmp_path / a, 80, 60)
    _img(tmp_path / b, 100, 50)
    xa = [float(10 + i) for i in range(10)]
    ya = [float(20 + i) for i in range(10)]
    xb = [float(11 + 2 * i) for i in range(10)]  # 1-based: 11..29 -> 0-based 10..28
    yb = [float(5 + i) for i in range(10)]       # 1-based: 5..14  -> 0-based 4..13
    header = ["imageA", "imageB"] + [f"{p}{i}" for p in ("XA", "YA", "XB", "YB") for i in range(1, 11)]
    with open(tmp_path / "test_pairs_pf.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([a, b] + xa + ya + xb + yb)
    return tmp_path


@pytest.fixture()
def pf_pascal_root(tmp_path: Path) -> Path:
    a, b = "PF-dataset-PASCAL/JPEGImages/a.jpg", "PF-dataset-PASCAL/JPEGImages/b.jpg"
    _img(tmp_path / a, 80, 60)
    _img(tmp_path / b, 120, 90)
    with open(tmp_path / "test_pairs_pf_pascal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_image", "target_image", "class", "XA", "YA", "XB", "YB"])
        w.writerow([a, b, 14, "11;21;31", "6;16;26", "2;3;4", "5;6;7"])
        w.writerow([b, a, 1, "1;2", "1;2", "3;4", "5;6"])
    return tmp_path


def test_willow_keypoints_are_zero_based_and_threshold_is_keypoint_bbox(pf_willow_root):
    for out_hw in ((48, 64), (64, 64)):
        item = PFWillowPairDataset(pf_willow_root, output_size_hw=out_hw)[0]
        assert int(item["n_valid_keypoints"]) == 10
        assert item["tgt_kps_orig"][0].tolist() == [10.0, 4.0]
        assert item["tgt_kps_orig"][9].tolist() == [28.0, 13.0]
        # alpha_bbox: max side of the target keypoint box, original frame (18 x 9)
        assert float(item["pck_threshold_bbox"]) == pytest.approx(18.0)
        assert item["tgt_imsize"].tolist() == [100.0, 50.0]
        assert item["category"] == "car(S)"
        assert torch.all(item["src_kps"][10:] == INVALID_KP_COORD)


def test_pascal_parses_semicolon_lists_class_and_image_threshold(pf_pascal_root):
    ds = PFPascalPairDataset(pf_pascal_root, output_size_hw=(64, 64))
    assert len(ds) == 2
    item = ds[0]
    assert item["category"] == PF_PASCAL_CATEGORIES[13] == "motorbike"
    assert int(item["n_valid_keypoints"]) == 3
    assert item["tgt_kps_orig"][:3].tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert float(item["pck_threshold_bbox"]) == pytest.approx(120.0)  # alpha_img: max(W, H)
    assert ds[1]["category"] == "aeroplane"


def test_pf_items_collate_with_spair_schema(pf_pascal_root):
    ds = PFPascalPairDataset(pf_pascal_root, output_size_hw=(64, 64))
    batch = spair_collate_fn([ds[0], ds[1]])
    assert batch["src_img"].shape == (2, 3, 64, 64)
    assert batch["tgt_kps_orig"].shape == (2, MAX_KEYPOINTS, 2)


def test_pf_only_test_split(pf_willow_root):
    with pytest.raises(ValueError):
        PFWillowPairDataset(pf_willow_root, split="train", output_size_hw=(64, 64))


def test_pf_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PFPascalPairDataset(tmp_path, output_size_hw=(64, 64))
