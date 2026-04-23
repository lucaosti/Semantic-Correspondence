"""Tests for `data.dataset`: split normalization, keypoint padding, collate,
and a live `SPair71kPairDataset` smoke test gated on disk availability.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from data.dataset import (
    INVALID_KP_COORD,
    MAX_KEYPOINTS,
    SPAIR_CATEGORIES,
    PreprocessMode,
    SPair71kPairDataset,
    SplitSpec,
    normalize_split_name,
    pad_keypoints_to_max,
    parse_spair_pair_line,
    spair_collate_fn,
    spair_split_filename,
)


# ---------------------------------------------------------------------------
# Pure helpers (no disk)
# ---------------------------------------------------------------------------


def test_normalize_split_name_aliases():
    assert normalize_split_name("train") is SplitSpec.TRAIN
    assert normalize_split_name("trn") is SplitSpec.TRAIN
    assert normalize_split_name("TRAIN") is SplitSpec.TRAIN
    assert normalize_split_name("val") is SplitSpec.VAL
    assert normalize_split_name("test") is SplitSpec.TEST


def test_normalize_split_name_invalid_raises():
    with pytest.raises(ValueError):
        normalize_split_name("trainset")


def test_spair_split_filename_maps_train_to_trn():
    # SPair-71k ships the training list as `trn.txt`; val/test keep their names.
    assert spair_split_filename("train") == "trn.txt"
    assert spair_split_filename("val") == "val.txt"
    assert spair_split_filename("test") == "test.txt"


def test_parse_spair_pair_line_valid():
    idx, src, tgt, cat = parse_spair_pair_line("0-2008_006481-2008_007698:aeroplane")
    assert idx == "0"
    assert src == "2008_006481"
    assert tgt == "2008_007698"
    assert cat == "aeroplane"


def test_parse_spair_pair_line_malformed():
    with pytest.raises(ValueError):
        parse_spair_pair_line("missing_colon")
    with pytest.raises(ValueError):
        parse_spair_pair_line("0-only_one:aeroplane")


def test_pad_keypoints_pads_with_sentinel():
    kps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out, n_valid = pad_keypoints_to_max(kps)
    assert out.shape == (MAX_KEYPOINTS, 2)
    assert n_valid == 2
    assert torch.equal(out[:2], kps)
    # Remaining slots filled with INVALID_KP_COORD sentinel.
    assert torch.all(out[2:] == INVALID_KP_COORD)


def test_pad_keypoints_truncates_when_over_max():
    kps = torch.arange(2 * (MAX_KEYPOINTS + 5), dtype=torch.float32).view(-1, 2)
    out, n_valid = pad_keypoints_to_max(kps)
    assert out.shape == (MAX_KEYPOINTS, 2)
    assert n_valid == MAX_KEYPOINTS
    # No sentinel values should appear (all slots valid).
    assert not torch.any(out == INVALID_KP_COORD)


def test_pad_keypoints_empty_input():
    kps = torch.empty(0, 2)
    out, n_valid = pad_keypoints_to_max(kps)
    assert n_valid == 0
    assert torch.all(out == INVALID_KP_COORD)


def test_pad_keypoints_bad_shape_raises():
    with pytest.raises(ValueError):
        pad_keypoints_to_max(torch.zeros(5, 3))


def test_collate_stacks_tensors_and_lists_strings():
    sample = {
        "pair_id_str": "pair-0",
        "category": "dog",
        "src_img": torch.zeros(3, 4, 4),
        "src_kps": torch.zeros(MAX_KEYPOINTS, 2),
        "n_valid_keypoints": torch.tensor([3], dtype=torch.int64),
    }
    batch = [sample, {**sample, "pair_id_str": "pair-1", "category": "cat"}]
    out = spair_collate_fn(batch)

    assert out["pair_id_str"] == ["pair-0", "pair-1"]
    assert out["category"] == ["dog", "cat"]
    assert out["src_img"].shape == (2, 3, 4, 4)
    assert out["src_kps"].shape == (2, MAX_KEYPOINTS, 2)
    assert out["n_valid_keypoints"].shape == (2, 1)


def test_collate_rejects_empty_batch():
    with pytest.raises(ValueError):
        spair_collate_fn([])


# ---------------------------------------------------------------------------
# Live dataset smoke test — only runs if SPair-71k is available on disk.
# ---------------------------------------------------------------------------


def _spair_root() -> Path | None:
    env = os.environ.get("SPAIR_ROOT")
    candidates = [env] if env else []
    candidates.append(str(Path(__file__).resolve().parents[1] / "data" / "SPair-71k"))
    for c in candidates:
        if c and Path(c).is_dir() and (Path(c) / "Layout" / "large").is_dir():
            return Path(c)
    return None


@pytest.mark.skipif(_spair_root() is None, reason="SPair-71k dataset not present on disk")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_spair_dataset_smoke(split: str):
    root = _spair_root()
    assert root is not None
    ds = SPair71kPairDataset(
        spair_root=str(root),
        split=split,
        preprocess=PreprocessMode.FIXED_RESIZE,
        output_size_hw=(224, 224),
        normalize=True,
    )
    assert len(ds) > 0, f"empty {split} split"

    sample = ds[0]
    assert sample["src_img"].shape == (3, 224, 224)
    assert sample["tgt_img"].shape == (3, 224, 224)
    assert sample["src_kps"].shape == (MAX_KEYPOINTS, 2)
    assert sample["tgt_kps"].shape == (MAX_KEYPOINTS, 2)
    assert sample["category"] in SPAIR_CATEGORIES
    n_valid = int(sample["n_valid_keypoints"].item())
    assert 0 < n_valid <= MAX_KEYPOINTS
    # Valid keypoints must not coincide with the invalid sentinel.
    assert not torch.any(sample["src_kps"][:n_valid] == INVALID_KP_COORD)

    batch = spair_collate_fn([ds[0], ds[1 if len(ds) > 1 else 0]])
    assert batch["src_img"].shape[0] == 2
    assert isinstance(batch["pair_id_str"], list)
    assert len(batch["pair_id_str"]) == 2
