"""
Percentage of Correct Keypoints (PCK) for semantic correspondence evaluation.

Use the ``test`` split for final reporting; use ``val`` only for model selection.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def pck_distance(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    pck_threshold: torch.Tensor,
    *,
    alpha: float,
    invalid_value: float = -2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-keypoint correctness for PCK at a relative threshold ``alpha``.

    A keypoint is correct if::

        ||pred - gt||_2 <= alpha * pck_threshold

    where ``pck_threshold`` is typically the max bounding-box side length in the **same
    coordinate frame** as ``pred_xy`` and ``gt_xy`` (bbox-normalized PCK).

    Parameters
    ----------
    pred_xy, gt_xy:
        ``(N, 2)`` tensors in the same pixel coordinate system.
    pck_threshold:
        Scalar tensor (or broadcastable) with the bbox-based tolerance scale.
    alpha:
        Relative threshold (e.g., ``0.05``, ``0.10``, ``0.20``).
    invalid_value:
        Keypoints whose ground-truth coordinates equal this value are ignored (padding).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(correct, valid_mask)`` both boolean ``(N,)`` tensors.
    """
    if pred_xy.shape != gt_xy.shape or pred_xy.dim() != 2 or pred_xy.shape[1] != 2:
        raise ValueError(f"Expected pred/gt shaped (N,2), got {tuple(pred_xy.shape)}")

    valid = (gt_xy[:, 0] > invalid_value) & (gt_xy[:, 1] > invalid_value)
    dist = torch.norm(pred_xy - gt_xy, dim=-1)
    thresh = float(alpha) * pck_threshold.to(device=dist.device, dtype=dist.dtype)
    if thresh.dim() == 0:
        thresh = thresh.view(1)
    correct = dist <= thresh
    correct = correct & valid
    return correct, valid


def mean_pck(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    pck_threshold: torch.Tensor,
    *,
    alpha: float,
    invalid_value: float = -2.0,
) -> torch.Tensor:
    """
    Compute scalar mean PCK over valid keypoints.

    Returns
    -------
    torch.Tensor
        Scalar mean (``0`` if no valid points).
    """
    correct, valid = pck_distance(pred_xy, gt_xy, pck_threshold, alpha=alpha, invalid_value=invalid_value)
    denom = torch.sum(valid.to(dtype=torch.float32))
    if denom.item() == 0:
        return pred_xy.new_zeros(())
    return torch.sum((correct & valid).to(dtype=torch.float32)) / denom
