"""
Host-aware defaults for PyTorch jobs: accelerator choice, DataLoader workers, pin memory.

Call :func:`recommended_device_str` / :func:`recommended_dataloader_workers` when CLI args use
``None`` or ``-1`` (auto). Keep batch size at ``1`` where the Gaussian loss requires it; workers
and device selection still improve throughput.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


def recommended_dataloader_workers(*, absolute_max: int = 32) -> int:
    """
    Pick a DataLoader ``num_workers`` from the CPU count and OS.

    Reserves roughly one logical CPU for the main process. Caps worker count on macOS where
    spawn-based workers are heavier.
    """
    n = os.cpu_count() or 4
    if n <= 1:
        return 0
    if sys.platform == "darwin":
        return int(max(2, min(8, n - 2)))
    return int(max(2, min(absolute_max, n - 1)))


def recommended_device_str() -> str:
    """Prefer CUDA, then Apple MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device_str(explicit: Optional[str]) -> str:
    """Return ``explicit`` if set, else :func:`recommended_device_str`."""
    return explicit if explicit else recommended_device_str()


def resolve_num_workers(explicit: int) -> int:
    """If ``explicit < 0``, use :func:`recommended_dataloader_workers`; else return ``explicit``."""
    if explicit < 0:
        return recommended_dataloader_workers()
    return explicit


def pin_memory_for(device: "torch.device") -> bool:
    """Pinned host memory helps CUDA async copies; it is not used for MPS/CPU."""
    return device.type == "cuda"


def maybe_tune_threads_for_cpu_device(device_type: str) -> None:
    """On CPU-only runs, allow PyTorch to use all logical cores for ops."""
    if device_type != "cpu":
        return
    import torch

    total = os.cpu_count() or 1
    try:
        torch.set_num_threads(total)
    except RuntimeError:
        pass
    try:
        inter = max(1, min(8, max(1, total // 4)))
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        pass


def dataloader_extra_kwargs(num_workers: int) -> dict:
    """Persistent workers and prefetch when ``num_workers > 0`` (reduces fork/open overhead)."""
    if num_workers <= 0:
        return {}
    prefetch = min(8, max(2, num_workers))
    return {"persistent_workers": True, "prefetch_factor": prefetch}
