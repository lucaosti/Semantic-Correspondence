"""
Coordinate rescaling utilities for multi-stage preprocessing (e.g., SAM's fixed 1024 input).
"""

from __future__ import annotations

from typing import Tuple

import torch


def rescale_keypoints_xy(
    xy: torch.Tensor,
    hw_src: Tuple[int, int],
    hw_dst: Tuple[int, int],
) -> torch.Tensor:
    """
    Linearly rescale ``(x, y)`` keypoints from one image resolution to another.

    Parameters
    ----------
    xy:
        ``(N, 2)`` tensor in pixel coordinates relative to ``hw_src``.
    hw_src:
        ``(H_src, W_src)`` source resolution.
    hw_dst:
        ``(H_dst, W_dst)`` destination resolution.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` keypoints in destination pixel coordinates.
    """
    if xy.dim() != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected (N,2) keypoints, got {tuple(xy.shape)}")
    h_s, w_s = int(hw_src[0]), int(hw_src[1])
    h_d, w_d = int(hw_dst[0]), int(hw_dst[1])
    if h_s <= 0 or w_s <= 0 or h_d <= 0 or w_d <= 0:
        raise ValueError("Heights/widths must be positive.")
    sx = w_d / float(w_s)
    sy = h_d / float(h_s)
    out = xy.clone()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out
