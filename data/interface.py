"""
Canonical dataset interface for this project.

This module provides a single entry point to construct datasets and dataloaders while
supporting multiple backends:

- ``native``: the project's existing SPair-71k pair dataset implementation.
- ``sd4match``: a vendored subset of the `ActiveVisionLab/SD4Match` dataset loader.

Per PDF precedence, split discipline must be:
- training uses ``trn`` (train)
- model selection uses ``val``
- final reporting uses ``test``
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn
from data.paths import resolve_spair_root


DatasetBackend = Literal["native", "sd4match"]


def _normalize_split_for_sd4match(split: str) -> str:
    s = str(split).strip().lower()
    if s == "train":
        return "trn"
    if s in ("val", "test"):
        return s
    raise ValueError(f"Unsupported split {split!r}. Expected one of: train, val, test.")


@dataclass(frozen=True)
class DatasetConfig:
    backend: DatasetBackend = "sd4match"
    name: str = "spair"
    root: Optional[str] = None  # parent directory containing SPair-71k (or repo data/)


@dataclass(frozen=True)
class RuntimeConfig:
    preprocess: str = "FIXED_RESIZE"
    image_height: int = 784
    image_width: int = 784
    num_workers: int = -1


def _sd4match_cfg(dataset_root: str, *, img_size: int) -> Any:
    """
    Create a minimal SD4Match-style cfg object with attribute access.

    SD4Match expects cfg.DATASET.{NAME, ROOT, IMG_SIZE, MEAN, STD} at minimum.
    """
    # ImageNet defaults (match SD4Match conventions).
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    DATASET = SimpleNamespace(NAME="spair", ROOT=dataset_root, IMG_SIZE=int(img_size), MEAN=mean, STD=std)
    return SimpleNamespace(DATASET=DATASET)


class _SD4MatchSampleAdapter(Dataset):
    """
    Wrap a SD4Match dataset and expose *both* naming styles:

    - SD4Match uses ``trg_*``; this project often uses ``tgt_*``.
    - To minimize breakage, keep original keys and add aliases.
    """

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.base[idx])
        # Aliases
        if "trg_img" in sample and "tgt_img" not in sample:
            sample["tgt_img"] = sample["trg_img"]
        if "trg_kps" in sample and "tgt_kps" not in sample:
            sample["tgt_kps"] = sample["trg_kps"]
        if "trg_bbox" in sample and "tgt_bbox" not in sample:
            sample["tgt_bbox"] = sample["trg_bbox"]
        if "trg_imname" in sample and "tgt_imname" not in sample:
            sample["tgt_imname"] = sample["trg_imname"]
        if "trg_imsize" in sample and "tgt_imsize" not in sample:
            sample["tgt_imsize"] = sample["trg_imsize"]

        # Difficulty flag aliases (PDF reporting: "difficulty levels").
        # SD4Match short keys -> project explicit names.
        if "vpvar" in sample and "viewpoint_variation" not in sample:
            sample["viewpoint_variation"] = sample["vpvar"]
        if "scvar" in sample and "scale_variation" not in sample:
            sample["scale_variation"] = sample["scvar"]
        if "trncn" in sample and "truncation" not in sample:
            sample["truncation"] = sample["trncn"]
        if "occln" in sample and "occlusion" not in sample:
            sample["occlusion"] = sample["occln"]
        return sample


def build_dataset(
    *,
    dataset: DatasetConfig,
    runtime: RuntimeConfig,
    split: str,
    spair_root: Optional[str] = None,
) -> Dataset:
    """
    Build a dataset for the requested backend.

    Parameters
    ----------
    dataset:
        Dataset backend selection and root.
    runtime:
        Preprocess and image sizing.
    split:
        ``train`` / ``val`` / ``test``.
    spair_root:
        Optional override path to ``SPair-71k`` folder itself (native backend only).
    """
    if dataset.name.lower() not in ("spair", "spair-71k", "spair71k"):
        raise ValueError(f"Only SPair is supported for now, got dataset.name={dataset.name!r}")

    if dataset.backend == "native":
        root = resolve_spair_root(spair_root)
        mode = PreprocessMode[runtime.preprocess.strip().upper()]
        return SPair71kPairDataset(
            spair_root=root,
            split=split,
            preprocess=mode,
            output_size_hw=(int(runtime.image_height), int(runtime.image_width)),
            normalize=True,
            photometric_augment=None,
        )

    if dataset.backend == "sd4match":
        from third_party.sd4match.dataset.spair import SPairDataset as _SD4MatchSPair

        dataset_root = dataset.root or "data"
        cfg = _sd4match_cfg(dataset_root, img_size=int(runtime.image_height))
        sd_split = _normalize_split_for_sd4match(split)
        base = _SD4MatchSPair(cfg, split=sd_split, category="all")
        return _SD4MatchSampleAdapter(base)

    raise ValueError(f"Unknown dataset backend: {dataset.backend!r}")


def build_dataloader(
    ds: Dataset,
    *,
    backend: DatasetBackend,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """
    Build a DataLoader with backend-appropriate collation.
    """
    if backend == "native":
        collate = spair_collate_fn
    else:
        # SD4Match datasets already return tensors; default collation is sufficient.
        collate = None

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )


__all__ = [
    "DatasetBackend",
    "DatasetConfig",
    "RuntimeConfig",
    "build_dataset",
    "build_dataloader",
]

