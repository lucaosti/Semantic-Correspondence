"""
Ensure ``SPair71kPairDataset`` is picklable for ``DataLoader(num_workers>0)`` (macOS spawn).

Skipped if SPair-71k is not installed locally.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn
from data.paths import resolve_spair_root


def test_dataloader_two_workers() -> None:
    root = resolve_spair_root(None)
    layout = os.path.join(root, "Layout", "large", "trn.txt")
    if not os.path.isfile(layout):
        pytest.skip("SPair-71k not found at expected path")

    ds = SPair71kPairDataset(
        spair_root=root,
        split="train",
        preprocess=PreprocessMode.FIXED_RESIZE,
        output_size_hw=(784, 784),
        normalize=True,
        photometric_augment=None,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=spair_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(loader))
    assert batch["src_img"].shape == (1, 3, 784, 784)
