"""
Losses for dense semantic correspondence (Gaussian-style supervision on similarity maps).

This follows the common practice used in benchmarks such as SD4Match: build a target
distribution on the feature grid and compare it to a softmax over similarities.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def pixel_xy_to_feat_xy(
    xy_pix: torch.Tensor,
    img_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Map pixel coordinates ``(x, y)`` to continuous coordinates in feature-map space.

    Parameters
    ----------
    xy_pix:
        ``(N, 2)`` with ``x`` along width and ``y`` along height.
    img_hw:
        ``(H, W)`` of the image fed to the backbone.
    feat_hw:
        ``(Hf, Wf)`` dense feature map shape.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` with ``(xf, yf)`` in feature-map pixel coordinates (same convention as
        :func:`models.common.matching.sample_features_bilinear`).
    """
    if xy_pix.dim() != 2 or xy_pix.shape[1] != 2:
        raise ValueError(f"Expected (N,2), got {tuple(xy_pix.shape)}")
    h, w = float(img_hw[0]), float(img_hw[1])
    hf, wf = float(feat_hw[0]), float(feat_hw[1])
    x = xy_pix[:, 0]
    y = xy_pix[:, 1]
    xf = (x + 0.5) * (wf / w) - 0.5
    yf = (y + 0.5) * (hf / h) - 0.5
    return torch.stack([xf, yf], dim=-1)


def gaussian_grid_2d(
    feat_hw: Tuple[int, int],
    center_xy: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Build a normalized 2D Gaussian on a feature grid.

    Parameters
    ----------
    feat_hw:
        ``(Hf, Wf)``.
    center_xy:
        ``(N, 2)`` centers ``(xf, yf)`` in feature-map coordinates.
    sigma:
        Standard deviation in **feature pixels**.

    Returns
    -------
    torch.Tensor
        ``(N, Hf, Wf)`` where each map sums to ``1`` (numerically stable softmax fallback).
    """
    hf, wf = int(feat_hw[0]), int(feat_hw[1])
    device = center_xy.device
    dtype = center_xy.dtype
    yy, xx = torch.meshgrid(
        torch.arange(hf, device=device, dtype=dtype),
        torch.arange(wf, device=device, dtype=dtype),
        indexing="ij",
    )
    # (N, Hf, Wf)
    dx = xx[None, :, :] - center_xy[:, 0].view(-1, 1, 1)
    dy = yy[None, :, :] - center_xy[:, 1].view(-1, 1, 1)
    dist2 = dx * dx + dy * dy
    logits = -dist2 / (2.0 * float(sigma) ** 2 + 1e-8)
    flat = logits.reshape(logits.shape[0], -1)
    prob = torch.softmax(flat, dim=1).reshape(-1, hf, wf)
    return prob


def gaussian_ce_loss_from_similarity_maps(
    sim_maps: torch.Tensor,
    gt_xy_pix: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    sigma_feat: float = 1.0,
) -> torch.Tensor:
    """
    Gaussian cross-entropy between softmax(similarities) and a Gaussian target on the grid.

    Parameters
    ----------
    sim_maps:
        ``(N, Hf, Wf)`` raw similarity maps (logits); one map per keypoint.
    gt_xy_pix:
        ``(N, 2)`` ground-truth target points in **pixel** coordinates (same frame as ``img_hw``).
    img_hw:
        ``(H, W)`` of the model input used to produce ``sim_maps``.
    sigma_feat:
        Gaussian standard deviation in **feature** pixels (converted from pixel sigma implicitly
        by using feature-space centers).

    Returns
    -------
    torch.Tensor
        Scalar loss (mean over ``N`` valid points).
    """
    if sim_maps.dim() != 3:
        raise ValueError(f"Expected (N,Hf,Wf), got {tuple(sim_maps.shape)}")
    hf, wf = int(sim_maps.shape[1]), int(sim_maps.shape[2])
    feat_hw = (hf, wf)

    centers = pixel_xy_to_feat_xy(gt_xy_pix, img_hw, feat_hw)
    target = gaussian_grid_2d(feat_hw, centers, sigma=sigma_feat)

    logits = sim_maps.reshape(sim_maps.shape[0], -1)
    log_prob = F.log_softmax(logits, dim=1).reshape_as(sim_maps)
    loss_map = -target * log_prob
    return loss_map.sum(dim=(1, 2)).mean()


__all__ = [
    "gaussian_ce_loss_from_similarity_maps",
    "gaussian_grid_2d",
    "pixel_xy_to_feat_xy",
]
