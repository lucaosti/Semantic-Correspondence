"""
Structured SPair-71k PCK evaluation for scripts and notebooks.

Use :class:`EvalRunSpec` to describe a single run (backbone, split, optional checkpoint,
window soft-argmax). Call :func:`run_spair_pck_eval` to obtain metric dictionaries suitable
for logging, JSON export, and comparison tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from data.paths import resolve_spair_root
from evaluation.baseline_eval import build_eval_dataloader, evaluate_spair_loader
from evaluation.checkpoint_loader import load_encoder_weights_from_pt
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from utils.hardware import pin_memory_for, recommended_device_str


@dataclass
class EvalRunSpec:
    """
    One evaluation configuration (mirrors CLI evaluation options).

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
    dataset_backend: str = "sd4match"  # "sd4match" | "native"
    metrics_backend: str = "sd4match"  # "sd4match" | "native"
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

        dl_kw = dataloader_extra_kwargs(num_workers, for_device="cuda" if pin_memory else "cpu")
    # Preserve the original backend-specific collation (native uses spair_collate_fn;
    # sd4match uses default collation).
    collate = getattr(loader, "collate_fn", None)
    return DataLoader(
        Subset(loader.dataset, list(range(n))),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        **dl_kw,
    )


def run_spair_pck_eval(
    spec: EvalRunSpec,
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
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
        dataset_backend=spec.dataset_backend,
    )
    loader = _maybe_limit_loader(
        loader, spec.limit, num_workers=spec.num_workers, pin_memory=pin_mem
    )

    if spec.metrics_backend == "native":
        metrics = evaluate_spair_loader(
            loader,
            extractor,
            alphas=alphas,
            use_window_soft_argmax=spec.use_window_soft_argmax,
            wsa_window=spec.wsa_window,
            wsa_temperature=spec.wsa_temperature,
            device=eval_device,
        )
        extras: Dict[str, Any] = {}
    elif spec.metrics_backend == "sd4match":
        from models.common.matching import predict_correspondences_cosine_argmax
        from models.common.window_soft_argmax import refine_predictions_window_soft_argmax

        # Re-implement the prediction loop (Stage 1 baseline + optional Stage 3 WSA),
        # but aggregate PCK using SD4Match-identical logic and report both granularities.
        extractor.eval()
        extractor.to(eval_device)

        summary = None

        from models.common.coord_utils import rescale_keypoints_xy

        def _hw(img_bchw: torch.Tensor) -> Tuple[int, int]:
            return int(img_bchw.shape[2]), int(img_bchw.shape[3])

        def _rescale_if(xy: torch.Tensor, hw_from: Tuple[int, int], hw_to: Tuple[int, int]) -> torch.Tensor:
            if hw_from == hw_to:
                return xy
            return rescale_keypoints_xy(xy, hw_from, hw_to)

        def _bucket01(v: Any) -> int:
            """
            Normalize a flag value to {0,1} robustly across backends.

            - native dataset yields tensors shaped (B,1)
            - sd4match yields scalar tensors (or sometimes Python ints)
            """
            if isinstance(v, torch.Tensor):
                x = int(v.reshape(-1)[0].item())
            else:
                x = int(v)
            return 1 if x != 0 else 0

        def _get_flag(batch: Dict[str, Any], name: str) -> Optional[int]:
            # Prefer native-style explicit names; fall back to SD4Match short keys.
            alias = {
                "viewpoint_variation": "vpvar",
                "scale_variation": "scvar",
                "truncation": "trncn",
                "occlusion": "occln",
            }
            if name in batch:
                return _bucket01(batch[name])
            alt = alias.get(name)
            if alt and alt in batch:
                return _bucket01(batch[alt])
            return None

        # Build SD4Match evaluators and extend their internal accumulator lists.

        # Build fresh evaluators and extend their internal lists.
        from third_party.sd4match.utils.evaluator import PCKEvaluator as _P
        from types import SimpleNamespace

        cfg_image = SimpleNamespace(EVALUATOR=SimpleNamespace(ALPHA=tuple(float(a) for a in alphas), BY="image"))
        cfg_point = SimpleNamespace(EVALUATOR=SimpleNamespace(ALPHA=tuple(float(a) for a in alphas), BY="point"))
        ev_image = _P(cfg_image)
        ev_point = _P(cfg_point)

        def _ensure_method_slots(ev: _P, method_name: str) -> None:
            for a in ev.alpha:
                key = f"{method_name}_pck{a}"
                if key not in ev.result:
                    ev.result[key] = {"all": []}

        _ensure_method_slots(ev_image, "custom")
        _ensure_method_slots(ev_point, "custom")

        # Difficulty buckets (PDF: analyze across difficulty levels).
        # We implement a flag-wise breakdown using SPair-71k indicators.
        diff_flags = ("viewpoint_variation", "scale_variation", "truncation", "occlusion")
        diff_evals: Dict[str, Dict[int, Dict[str, _P]]] = {
            f: {
                0: {"image": _P(cfg_image), "point": _P(cfg_point)},
                1: {"image": _P(cfg_image), "point": _P(cfg_point)},
            }
            for f in diff_flags
        }
        for f in diff_flags:
            for b in (0, 1):
                _ensure_method_slots(diff_evals[f][b]["image"], "custom")
                _ensure_method_slots(diff_evals[f][b]["point"], "custom")

        for batch in loader:
            # Support both backends by looking for aliases.
            src = batch["src_img"].to(eval_device)
            if "tgt_img" in batch:
                tgt = batch["tgt_img"].to(eval_device)
            else:
                tgt = batch["trg_img"].to(eval_device)

            # Keypoints can be (B,N,2) or nested lists depending on collation.
            if isinstance(batch.get("tgt_kps"), torch.Tensor):
                tgt_kps_b = batch["tgt_kps"].to(eval_device)
                src_kps_b = batch["src_kps"].to(eval_device)
            else:
                # native collate (batch_size=1) stores as list-like
                tgt_kps_b = batch["tgt_kps"][0].unsqueeze(0).to(eval_device)
                src_kps_b = batch["src_kps"][0].unsqueeze(0).to(eval_device)

            if "n_valid_keypoints" in batch:
                n_pts = batch["n_valid_keypoints"].to(eval_device).reshape(-1)
            else:
                n_pts = batch["n_pts"].to(eval_device).reshape(-1)

            # PCK threshold: prefer SD4Match `pckthres`, else native bbox threshold.
            if "pckthres" in batch:
                pckthres = batch["pckthres"].to(eval_device).reshape(-1)
            else:
                pckthres = batch["pck_threshold_bbox"].to(eval_device).reshape(-1)

            # Category list (strings).
            if "category" in batch:
                cat = batch["category"]
                categories = [cat[0]] if isinstance(cat, list) else [str(cat)]
            else:
                # native dataset: category_id is a (B,1) int tensor; map it to known SPair category list
                from data.dataset import SPAIR_CATEGORIES

                cid = int(batch["category_id"].reshape(-1)[0].item())
                categories = [SPAIR_CATEGORIES[cid]]

            feat_src, meta_src = extractor(src)
            feat_tgt, meta_tgt = extractor(tgt)

            ds_src = _hw(src)
            ds_tgt = _hw(tgt)
            coord_src = meta_src.get("coord_hw", ds_src)
            coord_tgt = meta_tgt.get("coord_hw", ds_tgt)

            src_match = _rescale_if(src_kps_b[0], ds_src, coord_src)

            out_pred = predict_correspondences_cosine_argmax(
                feat_src,
                feat_tgt,
                src_match,
                img_hw=coord_src,
                img_hw_src=coord_src,
                img_hw_tgt=coord_tgt,
                valid_mask=None,
            )
            pred_tgt = out_pred["pred_tgt_xy"]
            sims = out_pred["sim_maps"]
            if spec.use_window_soft_argmax:
                pred_tgt = refine_predictions_window_soft_argmax(
                    sims,
                    coord_tgt,
                    window_size=spec.wsa_window,
                    temperature=spec.wsa_temperature,
                )

            pred_in_dataset = _rescale_if(pred_tgt, coord_tgt, ds_tgt)
            # Shapes for SD4Match calc: (B,N,2)
            matches_b = pred_in_dataset.unsqueeze(0)
            trg_kps_for_eval = tgt_kps_b

            ev_image.calculate_pck(trg_kps_for_eval, matches_b, n_pts, categories, pckthres, method_name="custom")
            ev_point.calculate_pck(trg_kps_for_eval, matches_b, n_pts, categories, pckthres, method_name="custom")

            # Difficulty breakdown: route sample into each flag’s bucket (0 or 1) when available.
            for flag in diff_flags:
                b = _get_flag(batch, flag)
                if b is None:
                    continue
                diff_evals[flag][b]["image"].calculate_pck(
                    trg_kps_for_eval, matches_b, n_pts, categories, pckthres, method_name="custom"
                )
                diff_evals[flag][b]["point"].calculate_pck(
                    trg_kps_for_eval, matches_b, n_pts, categories, pckthres, method_name="custom"
                )

        def _mean_or_nan(xs: List[float]) -> float:
            if not xs:
                return float("nan")
            return float(sum(xs) / float(len(xs)))

        def _custom_summary_all(ev: _P) -> Dict[str, Dict[str, float]]:
            out: Dict[str, Dict[str, float]] = {}
            for a in ev.alpha:
                key = f"custom_pck{a}"
                vals = ev.result.get(key, {}).get("all", [])
                out[key] = {"all": _mean_or_nan(list(vals))}
            return out

        # SD4Match's `summerize_result()` only covers built-in method_options; for our external
        # predictions we summarize the "custom" lists ourselves.
        summary = {
            "image": _custom_summary_all(ev_image),
            "point": _custom_summary_all(ev_point),
        }
        # Flatten summary: macro (per-image mean) and micro (per-point mean).
        # Most SPair-71k papers report per-point (micro); both are exported for flexibility.
        metrics = {}
        for a in alphas:
            metrics[f"pck@{a:g}"]    = float(summary["image"][f"custom_pck{a}"]["all"])  # macro
            metrics[f"pck_pt@{a:g}"] = float(summary["point"][f"custom_pck{a}"]["all"])  # micro
        # Serialize difficulty-bucketed evaluator outputs.
        by_difficulty_flag: Dict[str, Any] = {}
        for flag in diff_flags:
            by_difficulty_flag[flag] = {}
            for bucket in (0, 1):
                ev_i = diff_evals[flag][bucket]["image"]
                ev_p = diff_evals[flag][bucket]["point"]
                by_difficulty_flag[flag][str(bucket)] = {
                    "per_image": ev_i.result,
                    "per_point": ev_p.result,
                    "summary": {
                        "image": _custom_summary_all(ev_i),
                        "point": _custom_summary_all(ev_p),
                    },
                }

        extras = {
            "sd4match_per_image": ev_image.result,
            "sd4match_per_point": ev_point.result,
            "sd4match_summary": summary,
            "sd4match_by_difficulty_flag": by_difficulty_flag,
        }
    else:
        raise ValueError(f"Unknown metrics_backend={spec.metrics_backend!r} (expected 'native' or 'sd4match').")

    out: Dict[str, Any] = {
        "name": spec.name,
        "spair_root": root,
        "metrics": metrics,
        "spec": asdict(spec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.update(extras)
    if load_info is not None:
        out["checkpoint_load"] = load_info
    return out


def run_comparison_batch(
    specs: Sequence[EvalRunSpec],
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
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
