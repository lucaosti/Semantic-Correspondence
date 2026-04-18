"""Dataset interface (sd4match backend only).

Split discipline: training uses ``trn`` (train), model selection uses ``val``,
final reporting uses ``test``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

from torch.utils.data import DataLoader, Dataset


def _normalize_split_for_sd4match(split: str) -> str:
    s = str(split).strip().lower()
    if s == "train":
        return "trn"
    if s in ("val", "test"):
        return s
    raise ValueError(f"Unsupported split {split!r}. Expected one of: train, val, test.")


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "spair"
    root: Optional[str] = None


@dataclass(frozen=True)
class RuntimeConfig:
    preprocess: str = "FIXED_RESIZE"
    image_height: int = 784
    image_width: int = 784
    num_workers: int = -1


def _sd4match_cfg(dataset_root: str, *, img_size: int) -> Any:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    DATASET = SimpleNamespace(
        NAME="spair", ROOT=dataset_root, IMG_SIZE=int(img_size), MEAN=mean, STD=std
    )
    return SimpleNamespace(DATASET=DATASET)


class _SD4MatchSampleAdapter(Dataset):
    """Expose both ``trg_*``/``tgt_*`` key styles and short/long difficulty flags."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.base[idx])
        aliases = [
            ("trg_img", "tgt_img"),
            ("trg_kps", "tgt_kps"),
            ("trg_bbox", "tgt_bbox"),
            ("trg_imname", "tgt_imname"),
            ("trg_imsize", "tgt_imsize"),
            ("vpvar", "viewpoint_variation"),
            ("scvar", "scale_variation"),
            ("trncn", "truncation"),
            ("occln", "occlusion"),
        ]
        for src, dst in aliases:
            if src in sample and dst not in sample:
                sample[dst] = sample[src]
        return sample


def build_dataset(
    *,
    dataset: DatasetConfig,
    runtime: RuntimeConfig,
    split: str,
) -> Dataset:
    if dataset.name.lower() not in ("spair", "spair-71k", "spair71k"):
        raise ValueError(f"Only SPair is supported, got {dataset.name!r}")
    from third_party.sd4match.dataset.spair import SPairDataset as _SD4MatchSPair

    dataset_root = dataset.root or "data"
    cfg = _sd4match_cfg(dataset_root, img_size=int(runtime.image_height))
    sd_split = _normalize_split_for_sd4match(split)
    base = _SD4MatchSPair(cfg, split=sd_split, category="all")
    return _SD4MatchSampleAdapter(base)


def build_dataloader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = [
    "DatasetConfig",
    "RuntimeConfig",
    "build_dataset",
    "build_dataloader",
]
