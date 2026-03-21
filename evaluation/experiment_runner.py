"""
Structured SPair-71k PCK evaluation for scripts and notebooks.

Use :class:`EvalRunSpec` to describe a single run (backbone, split, optional checkpoint,
window soft-argmax). Call :func:`run_spair_pck_eval` to obtain metric dictionaries suitable
for logging, JSON export, and comparison tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import spair_collate_fn
from data.paths import resolve_spair_root
from evaluation.baseline_eval import build_eval_dataloader, evaluate_spair_loader
from evaluation.checkpoint_loader import load_encoder_weights_from_pt
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from utils.hardware import pin_memory_for, recommended_device_str


@dataclass
class EvalRunSpec:
    """
    One evaluation configuration (matches ``scripts/eval_baseline.py`` options).

    Attributes
    ----------
    name:
        Human-readable label for tables and plots (e.g. ``dinov2_baseline``).
    backbone:
        One of :class:`models.common.dense_extractor.BackboneName` string values.
    split:
        ``train``, ``val``, or ``test`` (report final numbers on ``test`` per project rules).
    dinov2_weights, dinov3_weights, sam_checkpoint:
        Optional official weight paths (same semantics as the CLI).
    checkpoint:
        Optional fine-tuned / LoRA checkpoint produced by training scripts.
    use_window_soft_argmax:
        Task 3: inference-only refinement (no retraining).
    preprocess:
        :class:`data.dataset.PreprocessMode` enum name.
    height, width:
        Resize resolution for FIXED-style preprocess modes.
    limit:
        If ``> 0``, evaluate only the first ``limit`` pairs (debug / quick comparison).
    """

    name: str
    backbone: str
    split: str = "val"
    dinov2_weights: Optional[str] = None
    dinov3_weights: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    checkpoint: Optional[str] = None
    use_window_soft_argmax: bool = False
    wsa_window: int = 5
    wsa_temperature: float = 1.0
    preprocess: str = "FIXED_RESIZE"
    height: int = 784
    width: int = 784
    limit: int = 0
    num_workers: int = 4
    dino_layer_indices: Any = field(default_factory=lambda: 4)

    def to_dense_config(self) -> DenseExtractorConfig:
        """Build a :class:`DenseExtractorConfig` for this spec."""
        return DenseExtractorConfig(
            name=BackboneName(self.backbone),
            dinov2_weights_path=self.dinov2_weights,
            dinov3_weights_path=self.dinov3_weights,
            sam_checkpoint_path=self.sam_checkpoint,
            dino_layer_indices=self.dino_layer_indices,
        )


def _maybe_limit_loader(
    loader: DataLoader,
    limit: int,
    *,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    if limit <= 0:
        return loader
    n = min(limit, len(loader.dataset))
    dl_kw = {}
    if num_workers > 0:
        from utils.hardware import dataloader_extra_kwargs

        dl_kw = dataloader_extra_kwargs(num_workers)
    return DataLoader(
        Subset(loader.dataset, list(range(n))),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=spair_collate_fn,
        pin_memory=pin_memory,
        **dl_kw,
    )


def run_spair_pck_eval(
    spec: EvalRunSpec,
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.15),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run micro-averaged PCK for one configuration.

    Returns
    -------
    dict
        ``metrics`` (``pck@...``), ``spec`` (serialized), ``spair_root``, optional ``load_info``.
    """
    root = resolve_spair_root(spair_root)
    eval_device = device if device is not None else torch.device(recommended_device_str())
    pin_mem = pin_memory_for(eval_device)
    cfg = spec.to_dense_config()
    extractor = DenseFeatureExtractor(cfg, freeze=True)
    load_info: Optional[Dict[str, int]] = None
    if spec.checkpoint:
        load_info = load_encoder_weights_from_pt(extractor, spec.checkpoint, map_location="cpu")

    loader = build_eval_dataloader(
        root,
        split=spec.split,
        batch_size=1,
        num_workers=spec.num_workers,
        preprocess=spec.preprocess,
        output_size_hw=(spec.height, spec.width),
        pin_memory=pin_mem,
    )
    loader = _maybe_limit_loader(
        loader, spec.limit, num_workers=spec.num_workers, pin_memory=pin_mem
    )

    metrics = evaluate_spair_loader(
        loader,
        extractor,
        alphas=alphas,
        use_window_soft_argmax=spec.use_window_soft_argmax,
        wsa_window=spec.wsa_window,
        wsa_temperature=spec.wsa_temperature,
        device=eval_device,
    )

    out: Dict[str, Any] = {
        "name": spec.name,
        "spair_root": root,
        "metrics": metrics,
        "spec": asdict(spec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if load_info is not None:
        out["checkpoint_load"] = load_info
    return out


def run_comparison_batch(
    specs: Sequence[EvalRunSpec],
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.15),
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Evaluate multiple specs sequentially (safe for GPU memory)."""
    return [run_spair_pck_eval(s, spair_root=spair_root, alphas=alphas, device=device) for s in specs]


def metrics_rows_for_table(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten ``run_spair_pck_eval`` outputs into one dict per row (name + metric columns).

    Handy for :class:`pandas.DataFrame` or CSV export.
    """
    rows: List[Dict[str, Any]] = []
    for r in results:
        row = {"name": r["name"], "split": r["spec"]["split"], "wsa": r["spec"]["use_window_soft_argmax"]}
        row.update(r["metrics"])
        rows.append(row)
    return rows


__all__ = [
    "EvalRunSpec",
    "metrics_rows_for_table",
    "run_comparison_batch",
    "run_spair_pck_eval",
]
