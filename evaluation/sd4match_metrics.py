"""
SD4Match-identical evaluation wrapper.

This module provides a thin adapter around the vendored SD4Match `PCKEvaluator` so that:
- metric computation matches SD4Match behavior
- defaults follow the project PDF (PCK@{0.05, 0.1, 0.2} and per-image + per-point reporting)
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Literal, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from third_party.sd4match.utils.evaluator import PCKEvaluator


PCKGranularity = Literal["image", "point"]


def _sd4match_eval_cfg(*, alphas: Sequence[float], by: PCKGranularity) -> Any:
    """
    Minimal SD4Match-style cfg object with attribute access.
    """
    EVALUATOR = SimpleNamespace(ALPHA=tuple(float(a) for a in alphas), BY=str(by))
    return SimpleNamespace(EVALUATOR=EVALUATOR)


def _ensure_method_slots(ev: PCKEvaluator, method_name: str) -> None:
    """
    SD4Match's evaluator pre-allocates only its built-in method names.

    For this project we sometimes feed externally-computed matches (argmax/WSA/etc.),
    so we create the same result slots for a custom method label.
    """
    for a in ev.alpha:
        key = f"{method_name}_pck{a}"
        if key not in ev.result:
            ev.result[key] = {"all": []}


@dataclass(frozen=True)
class SD4MatchPCKOutputs:
    """
    Normalized outputs for both granularities + aggregated means.
    """

    per_image: Dict[str, Dict[str, float]]
    per_point: Dict[str, Dict[str, List[float]]]
    summary: Dict[str, Dict[str, float]]


@torch.no_grad()
def evaluate_matches_sd4match(
    *,
    trg_kps: torch.Tensor,
    matches: torch.Tensor,
    n_pts: torch.Tensor,
    categories: Sequence[str],
    pckthres: torch.Tensor,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
) -> SD4MatchPCKOutputs:
    """
    Compute SD4Match-identical PCK summaries given predicted matches.

    This uses SD4Match's internal `calculate_pck` and result aggregation, but it does NOT
    impose SD4Match's matching strategy. That lets the project keep its own Stage 1–3
    prediction rules (argmax + optional window soft-argmax) while having identical metrics.

    Shapes follow SD4Match conventions:
    - trg_kps: (B, N, 2)
    - matches: (B, N, 2)
    - n_pts: (B,) integer tensor
    - pckthres: (B,) float tensor (bbox/image tolerance scale)
    - categories: length B list/tuple of category strings
    """
    if trg_kps.shape != matches.shape:
        raise ValueError(f"Expected trg_kps and matches same shape, got {trg_kps.shape} vs {matches.shape}")
    if trg_kps.dim() != 3 or trg_kps.shape[-1] != 2:
        raise ValueError(f"Expected (B,N,2) keypoints, got {trg_kps.shape}")

    cfg_image = _sd4match_eval_cfg(alphas=alphas, by="image")
    cfg_point = _sd4match_eval_cfg(alphas=alphas, by="point")
    ev_image = PCKEvaluator(cfg_image)
    ev_point = PCKEvaluator(cfg_point)
    _ensure_method_slots(ev_image, "custom")
    _ensure_method_slots(ev_point, "custom")

    ev_image.calculate_pck(trg_kps, matches, n_pts, categories, pckthres, method_name="custom")
    ev_point.calculate_pck(trg_kps, matches, n_pts, categories, pckthres, method_name="custom")

    per_image = ev_image.result
    per_point = ev_point.result
    summary = {
        "image": ev_image.summerize_result(),
        "point": ev_point.summerize_result(),
    }
    return SD4MatchPCKOutputs(per_image=per_image, per_point=per_point, summary=summary)


@torch.no_grad()
def evaluate_feature_maps_sd4match(
    *,
    batch: Mapping[str, Any],
    src_featmaps: torch.Tensor,
    trg_featmaps: torch.Tensor,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
    softmax_temp: float = 0.04,
    gaussian_suppression_sigma: int = 7,
    enable_l2_norm: bool = True,
) -> SD4MatchPCKOutputs:
    """
    Evaluate SD4Match PCK on feature maps with both per-image and per-point reporting.

    Required batch keys (SD4Match style):
    - src_img, trg_img
    - src_kps, trg_kps
    - n_pts
    - category
    - pckthres
    """
    # Build two evaluators with identical settings except granularity.
    cfg_image = _sd4match_eval_cfg(alphas=alphas, by="image")
    cfg_point = _sd4match_eval_cfg(alphas=alphas, by="point")
    ev_image = PCKEvaluator(cfg_image)
    ev_point = PCKEvaluator(cfg_point)

    # SD4Match evaluator expects feature maps in the batch.
    payload = dict(batch)
    payload["src_featmaps"] = src_featmaps
    payload["trg_featmaps"] = trg_featmaps

    ev_image.evaluate_feature_map(
        payload,
        softmax_temp=softmax_temp,
        gaussian_suppression_sigma=gaussian_suppression_sigma,
        enable_l2_norm=enable_l2_norm,
    )
    ev_point.evaluate_feature_map(
        payload,
        softmax_temp=softmax_temp,
        gaussian_suppression_sigma=gaussian_suppression_sigma,
        enable_l2_norm=enable_l2_norm,
    )

    per_image = ev_image.result
    per_point = ev_point.result

    summary = {
        "image": ev_image.summerize_result(),
        "point": ev_point.summerize_result(),
    }
    return SD4MatchPCKOutputs(per_image=per_image, per_point=per_point, summary=summary)


def default_alphas_pdf() -> Tuple[float, float, float]:
    """
    PDF default thresholds: (0.05, 0.1, 0.2).
    """
    return (0.05, 0.1, 0.2)


__all__ = [
    "SD4MatchPCKOutputs",
    "default_alphas_pdf",
    "evaluate_feature_maps_sd4match",
    "evaluate_matches_sd4match",
]

