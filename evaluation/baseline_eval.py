"""SPair-71k evaluation dataloader (sd4match backend)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from data.interface import DatasetConfig, RuntimeConfig, build_dataloader, build_dataset


def build_eval_dataloader(
    spair_root: str,
    split: str,
    *,
    batch_size: int = 1,
    num_workers: int = 1,
    preprocess: str = "fixed_resize",
    output_size_hw: Tuple[int, int] = (784, 784),
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """Build a sd4match-backed dataloader for evaluation.

    ``spair_root`` points to the ``SPair-71k`` folder; sd4match expects its parent.
    """
    from utils.hardware import pin_memory_for, resolve_num_workers
    import os

    if num_workers < 0:
        num_workers = resolve_num_workers(num_workers)
    if pin_memory is None:
        pin_memory = pin_memory_for(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    dataset_parent = os.path.dirname(os.path.abspath(spair_root))
    ds = build_dataset(
        dataset=DatasetConfig(name="spair", root=dataset_parent),
        runtime=RuntimeConfig(
            preprocess=preprocess,
            image_height=int(output_size_hw[0]),
            image_width=int(output_size_hw[1]),
            num_workers=num_workers,
        ),
        split=split,
    )
    return build_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = ["build_eval_dataloader"]
