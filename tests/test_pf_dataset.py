"""Unit tests for PFWillowPairDataset and PFPascalPairDataset.

All tests work without the real datasets on disk: synthetic images and CSV
annotation files are created in a temporary directory via pytest fixtures.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image

from data.dataset import INVALID_KP_COORD, MAX_KEYPOINTS, spair_collate_fn
from data.pf_dataset import (
    PF_PASCAL_CATEGORIES,
    PF_WILLOW_CATEGORIES,
    PFPascalPairDataset,
    PFWillowPairDataset,
    _parse_bbox_field,
    _parse_kps_field,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(path: Path, width: int = 64, height: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(str(path))


def _kps_json(pts: List[List[float]]) -> str:
    return json.dumps(pts)


def _bbox_json(box: List[float]) -> str:
    return json.dumps(box)


# ---------------------------------------------------------------------------
# _parse_kps_field
# ---------------------------------------------------------------------------


def test_parse_kps_field_json_pairs():
    pts = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result = _parse_kps_field(json.dumps(pts))
    assert result.shape == (3, 2)
    assert result.dtype == torch.float32
    assert torch.allclose(result, torch.tensor(pts, dtype=torch.float32))


def test_parse_kps_field_flat_list():
    flat = [1.0, 2.0, 3.0, 4.0]
    result = _parse_kps_field(json.dumps(flat))
    assert result.shape == (2, 2)
    assert torch.allclose(result, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))


def test_parse_kps_field_empty():
    result = _parse_kps_field("[]")
    assert result.shape == (0, 2)


def test_parse_bbox_field_valid():
    box = [10.0, 20.0, 50.0, 80.0]
    result = _parse_bbox_field(json.dumps(box))
    assert result.shape == (4,)
    assert torch.allclose(result, torch.tensor(box, dtype=torch.float32))


def test_parse_bbox_field_wrong_length_raises():
    with pytest.raises(ValueError, match="Expected 4"):
        _parse_bbox_field(json.dumps([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# PFWillowPairDataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pf_willow_root(tmp_path: Path) -> Path:
    """Synthetic PF-Willow dataset with 3 pairs in car(S) category."""
    category = "car(S)"
    img_dir = tmp_path / category
    img_dir.mkdir(parents=True)
    for i in range(1, 5):
        _make_rgb_image(img_dir / f"{i:03d}.jpg")

    kps_src = [[10.0, 15.0], [20.0, 25.0], [30.0, 35.0]]
    kps_trg = [[11.0, 16.0], [21.0, 26.0], [31.0, 36.0]]
    rows = [
        {
            "src_imname": f"{category}/001.jpg",
            "trg_imname": f"{category}/002.jpg",
            "src_kps": _kps_json(kps_src),
            "trg_kps": _kps_json(kps_trg),
            "n_kps": 3,
        },
        {
            "src_imname": f"{category}/002.jpg",
            "trg_imname": f"{category}/003.jpg",
            "src_kps": _kps_json(kps_src[:2]),
            "trg_kps": _kps_json(kps_trg[:2]),
            "n_kps": 2,
        },
        {
            "src_imname": f"{category}/003.jpg",
            "trg_imname": f"{category}/004.jpg",
            "src_kps": _kps_json(kps_src),
            "trg_kps": _kps_json(kps_trg),
            "n_kps": 3,
        },
    ]

    import csv as _csv

    csv_path = tmp_path / "test_pairs_pf.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=["src_imname", "trg_imname", "src_kps", "trg_kps", "n_kps"])
        writer.writeheader()
        writer.writerows(rows)

    return tmp_path


# ---------------------------------------------------------------------------
# PFWillowPairDataset tests
# ---------------------------------------------------------------------------


def test_pf_willow_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="test_pairs_pf.csv"):
        PFWillowPairDataset(tmp_path)


def test_pf_willow_len(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    assert len(ds) == 3


def test_pf_willow_item_schema(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64), normalize=False)
    item = ds[0]
    required_keys = {
        "pair_id_str", "category", "category_id",
        "src_img", "tgt_img",
        "src_kps", "tgt_kps", "kp_ids", "n_valid_keypoints",
        "src_bbox", "tgt_bbox", "pck_threshold_bbox",
        "src_imsize", "tgt_imsize",
        "viewpoint_variation", "scale_variation", "truncation", "occlusion",
        "src_kp_names",
    }
    assert required_keys.issubset(item.keys()), f"Missing keys: {required_keys - item.keys()}"

    assert item["src_img"].shape == (3, 64, 64)
    assert item["tgt_img"].shape == (3, 64, 64)
    assert item["src_kps"].shape == (MAX_KEYPOINTS, 2)
    assert item["tgt_kps"].shape == (MAX_KEYPOINTS, 2)
    assert item["kp_ids"].shape == (MAX_KEYPOINTS,)
    assert item["n_valid_keypoints"].shape == (1,)
    assert item["src_bbox"].shape == (4,)
    assert item["tgt_bbox"].shape == (4,)
    assert item["pck_threshold_bbox"].ndim == 0
    assert len(item["src_kp_names"]) == MAX_KEYPOINTS


def test_pf_willow_image_size_based_pck_threshold(pf_willow_root: Path):
    """pck_threshold_bbox must equal max(out_H, out_W) for image-size normalisation."""
    out_h, out_w = 48, 64
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(out_h, out_w))
    item = ds[0]
    expected = float(max(out_h, out_w))
    assert float(item["pck_threshold_bbox"].item()) == pytest.approx(expected)


def test_pf_willow_pck_threshold_constant_across_items(pf_willow_root: Path):
    """All items share the same pck_threshold_bbox under fixed-resize."""
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    thresholds = [float(ds[i]["pck_threshold_bbox"].item()) for i in range(len(ds))]
    assert len(set(thresholds)) == 1


def test_pf_willow_n_valid_keypoints_correct(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    assert int(ds[0]["n_valid_keypoints"].item()) == 3
    assert int(ds[1]["n_valid_keypoints"].item()) == 2


def test_pf_willow_padded_slots_are_sentinel(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    item = ds[1]  # 2 valid keypoints
    n = int(item["n_valid_keypoints"].item())
    assert torch.all(item["src_kps"][n:] == INVALID_KP_COORD)
    assert torch.all(item["tgt_kps"][n:] == INVALID_KP_COORD)


def test_pf_willow_difficulty_flags_zero(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    item = ds[0]
    for flag in ("viewpoint_variation", "scale_variation", "truncation", "occlusion"):
        assert float(item[flag].item()) == 0.0, f"{flag} should be 0"


def test_pf_willow_category_inferred_from_path(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    assert ds[0]["category"] == "car(S)"


def test_pf_willow_collate(pf_willow_root: Path):
    ds = PFWillowPairDataset(pf_willow_root, output_size_hw=(64, 64))
    batch = spair_collate_fn([ds[0], ds[1]])
    assert batch["src_img"].shape == (2, 3, 64, 64)
    assert batch["n_valid_keypoints"].shape == (2, 1)


# ---------------------------------------------------------------------------
# PFPascalPairDataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pf_pascal_root(tmp_path: Path) -> Path:
    """Synthetic PF-Pascal dataset with 2 pairs."""
    category = "aeroplane"
    img_dir = tmp_path / "JPEGImages" / category
    img_dir.mkdir(parents=True)
    for i in range(1, 5):
        _make_rgb_image(img_dir / f"img{i:03d}.jpg", width=80, height=60)

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    kps_src = [[10.0, 15.0], [20.0, 25.0]]
    kps_trg = [[11.0, 16.0], [21.0, 26.0]]
    src_bbox = [5.0, 5.0, 40.0, 50.0]
    trg_bbox = [6.0, 6.0, 42.0, 52.0]

    rows = [
        {
            "src_imname": f"JPEGImages/{category}/img001.jpg",
            "trg_imname": f"JPEGImages/{category}/img002.jpg",
            "src_kps": _kps_json(kps_src),
            "trg_kps": _kps_json(kps_trg),
            "src_bbox": _bbox_json(src_bbox),
            "trg_bbox": _bbox_json(trg_bbox),
            "category": category,
        },
        {
            "src_imname": f"JPEGImages/{category}/img003.jpg",
            "trg_imname": f"JPEGImages/{category}/img004.jpg",
            "src_kps": _kps_json(kps_src),
            "trg_kps": _kps_json(kps_trg),
            "src_bbox": _bbox_json(src_bbox),
            "trg_bbox": _bbox_json(trg_bbox),
            "category": category,
        },
    ]

    import csv as _csv

    fields = ["src_imname", "trg_imname", "src_kps", "trg_kps", "src_bbox", "trg_bbox", "category"]
    for split_file in ("trn_pairs.csv", "val_pairs.csv", "test_pairs.csv"):
        with open(ann_dir / split_file, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    return tmp_path


# ---------------------------------------------------------------------------
# PFPascalPairDataset tests
# ---------------------------------------------------------------------------


def test_pf_pascal_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="test_pairs.csv"):
        PFPascalPairDataset(tmp_path, split="test")


def test_pf_pascal_len(pf_pascal_root: Path):
    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(64, 64))
    assert len(ds) == 2


def test_pf_pascal_split_routing(pf_pascal_root: Path):
    for split in ("train", "val", "test"):
        ds = PFPascalPairDataset(pf_pascal_root, split=split, output_size_hw=(64, 64))
        assert len(ds) == 2


def test_pf_pascal_item_schema(pf_pascal_root: Path):
    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(64, 64), normalize=False)
    item = ds[0]
    required_keys = {
        "pair_id_str", "category", "category_id",
        "src_img", "tgt_img",
        "src_kps", "tgt_kps", "kp_ids", "n_valid_keypoints",
        "src_bbox", "tgt_bbox", "pck_threshold_bbox",
        "src_imsize", "tgt_imsize",
        "viewpoint_variation", "scale_variation", "truncation", "occlusion",
        "src_kp_names",
    }
    assert required_keys.issubset(item.keys())
    assert item["src_img"].shape == (3, 64, 64)
    assert item["src_kps"].shape == (MAX_KEYPOINTS, 2)


def test_pf_pascal_bbox_pck_normalisation(pf_pascal_root: Path):
    """pck_threshold_bbox must equal max(rescaled_bbox_W, rescaled_bbox_H)."""
    out_h, out_w = 64, 64
    orig_w, orig_h = 80, 60  # matches _make_rgb_image defaults

    trg_bbox_orig = [6.0, 6.0, 42.0, 52.0]  # x1, y1, x2, y2
    sx = out_w / orig_w
    sy = out_h / orig_h
    x1 = trg_bbox_orig[0] * sx
    y1 = trg_bbox_orig[1] * sy
    x2 = trg_bbox_orig[2] * sx
    y2 = trg_bbox_orig[3] * sy
    expected = max(x2 - x1, y2 - y1)

    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(out_h, out_w))
    item = ds[0]
    assert float(item["pck_threshold_bbox"].item()) == pytest.approx(expected, rel=1e-5)


def test_pf_pascal_bboxes_nonzero(pf_pascal_root: Path):
    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(64, 64))
    item = ds[0]
    assert item["src_bbox"].sum() > 0
    assert item["tgt_bbox"].sum() > 0


def test_pf_pascal_category_field(pf_pascal_root: Path):
    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(64, 64))
    assert ds[0]["category"] == "aeroplane"


def test_pf_pascal_difficulty_flags_zero(pf_pascal_root: Path):
    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(64, 64))
    item = ds[0]
    for flag in ("viewpoint_variation", "scale_variation", "truncation", "occlusion"):
        assert float(item[flag].item()) == 0.0


def test_pf_pascal_collate(pf_pascal_root: Path):
    ds = PFPascalPairDataset(pf_pascal_root, split="test", output_size_hw=(64, 64))
    batch = spair_collate_fn([ds[0], ds[1]])
    assert batch["src_img"].shape == (2, 3, 64, 64)
    assert batch["tgt_bbox"].shape == (2, 4)


# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------


def test_pf_willow_categories_count():
    assert len(PF_WILLOW_CATEGORIES) == 10


def test_pf_pascal_categories_count():
    assert len(PF_PASCAL_CATEGORIES) == 20


def test_pf_pascal_categories_no_duplicates():
    assert len(PF_PASCAL_CATEGORIES) == len(set(PF_PASCAL_CATEGORIES))
