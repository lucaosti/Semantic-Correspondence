"""
Host-aware defaults for PyTorch jobs: accelerator choice, DataLoader workers, pin memory.

Call :func:`recommended_device_str` / :func:`recommended_dataloader_workers` when CLI args use
``None`` or ``-1`` (auto). Training defaults to larger pair batches (see ``--batch-size``); workers
and device selection still improve throughput.

``recommended_dataloader_workers`` picks aggressive counts: many host threads for CUDA prefetch,
on **CPU** a smaller worker pool (about **n/4**, cap 16) so the main process keeps most cores
for tensor ops (see :func:`maybe_tune_threads_for_cpu_device`).
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    import torch


def _detect_accelerator_str() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def recommended_dataloader_workers(
    *,
    accelerator: Optional[str] = None,
    cpu_worker_cap: int = 64,
    cuda_worker_cap: int = 64,
) -> int:
    """
    Pick ``num_workers`` to stress host parallelism without going past ``*_worker_cap``.

    * **cuda** — many workers to keep the GPU fed (up to ~¾ of logical CPUs, floored at 8).
    * **cpu** — about ``max(2, n//4)`` workers (cap 16): prefetch only; main runs the ViT.
    * **mps / darwin** — capped (spawn cost); still higher than the old default.
    """
    n = os.cpu_count() or 4
    if n <= 1:
        return 0

    acc = accelerator or _detect_accelerator_str()

    if acc == "cuda":
        want = max(8, (n * 3) // 4)
        capped = min(cuda_worker_cap, max(1, n - 1), want)
        return int(max(2, capped))

    if acc == "mps" or sys.platform == "darwin":
        # macOS unified memory: each worker is a separate process that maps the full dataset
        # and model weights; with MPS the address space is shared with the GPU, so more than
        # one worker quickly exhausts RAM on typical Apple Silicon configs.
        return 1

    if sys.platform == "win32":
        # Windows uses spawn for DataLoader workers; conservative cap reduces overhead.
        return int(max(2, min(4, max(1, n // 2))))

    # CPU training: main process does forward/backward; workers only load/decode.
    io_workers = max(2, n // 4)
    return int(max(2, min(16, min(cpu_worker_cap, io_workers))))


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


def resolve_num_workers(explicit: int, *, accelerator: Optional[str] = None) -> int:
    """If ``explicit < 0``, use :func:`recommended_dataloader_workers` for this accelerator."""
    if explicit < 0:
        return recommended_dataloader_workers(accelerator=accelerator)
    return explicit


def pin_memory_for(device: "torch.device") -> bool:
    """Pinned host memory helps CUDA async copies; it is not used for MPS/CPU."""
    return device.type == "cuda"


def maybe_tune_threads_for_cpu_device(device_type: str, *, dataloader_workers: int = 0) -> None:
    """
    On CPU-only runs, set PyTorch intra-/inter-op thread pools.

    With ``DataLoader(num_workers>0)``, worker processes already use CPU; capping the **main**
    process thread count reduces oversubscription (many processes × many BLAS threads fighting
    for the same cores). With ``dataloader_workers == 0``, all logical CPUs go to PyTorch.
    """
    if device_type != "cpu":
        return
    import torch

    total = os.cpu_count() or 1
    w = max(0, int(dataloader_workers))
    if w == 0:
        intra = total
    elif total > w + 1:
        intra = max(1, total - w)
    else:
        intra = max(1, total // 2)

    try:
        torch.set_num_threads(intra)
    except RuntimeError:
        pass
    try:
        inter = max(1, min(16, max(2, intra // 2)))
        torch.set_num_interop_threads(inter)
    except RuntimeError:
        pass


def apply_accelerator_throughput_tweaks(device: "torch.device") -> None:
    """CUDA: cuDNN/TF32. CPU/MPS: faster matmul where supported."""
    import torch

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    try:
        torch.set_float32_matmul_precision("high")
    except (RuntimeError, AttributeError):
        pass


def cpu_dataloader_worker_init(_worker_id: int) -> None:
    """Single-thread BLAS inside DataLoader workers (avoids fighting the main PyTorch thread pool)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def loader_worker_init_for_device(device_type: str, num_workers: int) -> Optional[Callable[[int], None]]:
    if device_type == "cpu" and num_workers > 0:
        return cpu_dataloader_worker_init
    return None


def dataloader_extra_kwargs(num_workers: int, *, for_device: str = "cuda") -> dict:
    """
    Persistent workers and prefetch when ``num_workers > 0``.

    On **CPU**, prefetch is kept smaller (main RAM goes to the model + smaller loader queues).
    Set ``SEMANTIC_CORRESPONDENCE_PREFETCH_CAP`` to raise the ceiling (max 48).
    """
    if num_workers <= 0:
        return {}
    nw = num_workers
    if for_device == "cpu":
        cap = int(os.environ.get("SEMANTIC_CORRESPONDENCE_PREFETCH_CAP", "12"))
        cap = max(2, min(24, cap))
        prefetch = max(2, min(cap, max(2, nw)))
    else:
        cap = int(os.environ.get("SEMANTIC_CORRESPONDENCE_PREFETCH_CAP", "24"))
        cap = max(4, min(48, cap))
        want = max(5, min(cap, nw // 2 + 2))
        prefetch = max(4, want)
    # On Windows, persistent workers with spawn can leave zombie processes on abrupt exit.
    persistent = sys.platform != "win32"
    return {"persistent_workers": persistent, "prefetch_factor": prefetch}
