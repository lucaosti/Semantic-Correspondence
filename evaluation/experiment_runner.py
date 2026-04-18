"""SPair-71k PCK evaluation runner (sd4match backend).

Use :class:`EvalRunSpec` to describe a single run (backbone, split, optional checkpoint,
window soft-argmax). Call :func:`run_spair_pck_eval` to obtain metric dictionaries
suitable for logging, JSON export, and comparison tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset

from data.paths import resolve_spair_root
from evaluation.baseline_eval import build_eval_dataloader
from evaluation.checkpoint_loader import load_encoder_weights_from_pt
from models.common.coord_utils import rescale_keypoints_xy
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from models.common.matching import predict_correspondences_cosine_argmax
from models.common.window_soft_argmax import refine_predictions_window_soft_argmax
from third_party.sd4match.utils.evaluator import PCKEvaluator
from utils.hardware import pin_memory_for, recommended_device_str


@dataclass
class EvalRunSpec:
    """One evaluation configuration.

    Report final numbers on ``test`` per project rules; use ``val`` for model selection.
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
        return DenseExtractorConfig(
            name=BackboneName(self.backbone),
            dinov2_weights_path=self.dinov2_weights,
            dinov3_weights_path=self.dinov3_weights,
            sam_checkpoint_path=self.sam_checkpoint,
            dino_layer_indices=self.dino_layer_indices,
        )


def _maybe_limit_loader(
    loader: DataLoader, limit: int, *, num_workers: int, pin_memory: bool
) -> DataLoader:
    if limit <= 0:
        return loader
    n = min(limit, len(loader.dataset))
    return DataLoader(
        Subset(loader.dataset, list(range(n))),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _bucket01(v: Any) -> int:
    if isinstance(v, torch.Tensor):
        return 1 if int(v.reshape(-1)[0].item()) != 0 else 0
    return 1 if int(v) != 0 else 0


def _get_flag(batch: Dict[str, Any], name: str) -> Optional[int]:
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


def _ensure_method_slots(ev: PCKEvaluator, method_name: str) -> None:
    for a in ev.alpha:
        key = f"{method_name}_pck{a}"
        if key not in ev.result:
            ev.result[key] = {"all": []}


def _mean_or_nan(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(sum(xs) / float(len(xs)))


def _custom_summary_all(ev: PCKEvaluator) -> Dict[str, Dict[str, float]]:
    return {
        f"custom_pck{a}": {"all": _mean_or_nan(list(ev.result.get(f"custom_pck{a}", {}).get("all", [])))}
        for a in ev.alpha
    }


def run_spair_pck_eval(
    spec: EvalRunSpec,
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run PCK for one configuration using the sd4match evaluator.

    Returns a dict with ``metrics`` (``pck@...`` macro per-image and ``pck_pt@...`` micro
    per-point), ``spec``, ``sd4match_per_image``/``per_point``/``summary``, and
    ``sd4match_by_difficulty_flag``.
    """
    root = resolve_spair_root(spair_root)
    eval_device = device if device is not None else torch.device(recommended_device_str())
    pin_mem = pin_memory_for(eval_device)

    extractor = DenseFeatureExtractor(spec.to_dense_config(), freeze=True)
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

    extractor.eval()
    extractor.to(eval_device)

    cfg_image = SimpleNamespace(EVALUATOR=SimpleNamespace(ALPHA=tuple(float(a) for a in alphas), BY="image"))
    cfg_point = SimpleNamespace(EVALUATOR=SimpleNamespace(ALPHA=tuple(float(a) for a in alphas), BY="point"))
    ev_image = PCKEvaluator(cfg_image)
    ev_point = PCKEvaluator(cfg_point)
    _ensure_method_slots(ev_image, "custom")
    _ensure_method_slots(ev_point, "custom")

    diff_flags = ("viewpoint_variation", "scale_variation", "truncation", "occlusion")
    diff_evals: Dict[str, Dict[int, Dict[str, PCKEvaluator]]] = {
        f: {
            0: {"image": PCKEvaluator(cfg_image), "point": PCKEvaluator(cfg_point)},
            1: {"image": PCKEvaluator(cfg_image), "point": PCKEvaluator(cfg_point)},
        }
        for f in diff_flags
    }
    for f in diff_flags:
        for b in (0, 1):
            _ensure_method_slots(diff_evals[f][b]["image"], "custom")
            _ensure_method_slots(diff_evals[f][b]["point"], "custom")

    def _hw(img_bchw: torch.Tensor) -> Tuple[int, int]:
        return int(img_bchw.shape[2]), int(img_bchw.shape[3])

    def _rescale_if(xy: torch.Tensor, hw_from: Tuple[int, int], hw_to: Tuple[int, int]) -> torch.Tensor:
        if hw_from == hw_to:
            return xy
        return rescale_keypoints_xy(xy, hw_from, hw_to)

    for batch in loader:
        src = batch["src_img"].to(eval_device)
        tgt = batch.get("tgt_img", batch.get("trg_img")).to(eval_device)

        if isinstance(batch.get("tgt_kps"), torch.Tensor):
            tgt_kps_b = batch["tgt_kps"].to(eval_device)
            src_kps_b = batch["src_kps"].to(eval_device)
        else:
            tgt_kps_b = batch["tgt_kps"][0].unsqueeze(0).to(eval_device)
            src_kps_b = batch["src_kps"][0].unsqueeze(0).to(eval_device)

        n_pts = batch.get("n_valid_keypoints", batch.get("n_pts")).to(eval_device).reshape(-1)
        pckthres = batch.get("pckthres", batch.get("pck_threshold_bbox")).to(eval_device).reshape(-1)
        cat = batch.get("category")
        categories = [cat[0]] if isinstance(cat, list) else [str(cat)]

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
                sims, coord_tgt, window_size=spec.wsa_window, temperature=spec.wsa_temperature
            )

        pred_in_dataset = _rescale_if(pred_tgt, coord_tgt, ds_tgt)
        matches_b = pred_in_dataset.unsqueeze(0)

        ev_image.calculate_pck(tgt_kps_b, matches_b, n_pts, categories, pckthres, method_name="custom")
        ev_point.calculate_pck(tgt_kps_b, matches_b, n_pts, categories, pckthres, method_name="custom")

        for flag in diff_flags:
            b = _get_flag(batch, flag)
            if b is None:
                continue
            diff_evals[flag][b]["image"].calculate_pck(
                tgt_kps_b, matches_b, n_pts, categories, pckthres, method_name="custom"
            )
            diff_evals[flag][b]["point"].calculate_pck(
                tgt_kps_b, matches_b, n_pts, categories, pckthres, method_name="custom"
            )

    summary = {"image": _custom_summary_all(ev_image), "point": _custom_summary_all(ev_point)}
    metrics: Dict[str, float] = {}
    for a in alphas:
        metrics[f"pck@{a:g}"] = float(summary["image"][f"custom_pck{a}"]["all"])  # macro per-image
        metrics[f"pck_pt@{a:g}"] = float(summary["point"][f"custom_pck{a}"]["all"])  # micro per-point

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

    out: Dict[str, Any] = {
        "name": spec.name,
        "spair_root": root,
        "metrics": metrics,
        "spec": asdict(spec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sd4match_per_image": ev_image.result,
        "sd4match_per_point": ev_point.result,
        "sd4match_summary": summary,
        "sd4match_by_difficulty_flag": by_difficulty_flag,
    }
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
    return [run_spair_pck_eval(s, spair_root=spair_root, alphas=alphas, device=device) for s in specs]


def metrics_rows_for_table(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
