"""
Coordinate conventions shared by training, matching and evaluation.

Every pixel coordinate in this project refers to **pixel centres**: pixel ``i`` covers
``[i - 0.5, i + 0.5]``. A pure resize from size ``S`` to size ``S'`` therefore maps
``x -> (x + 0.5) * S'/S - 0.5`` (the convention used by PIL / ``F.interpolate`` with
``align_corners=False``). A dense feature grid of ``Wf`` cells over an image of width
``W`` is the same mapping with ``S' = Wf``.
"""

from __future__ import annotations

from typing import Tuple

import torch


def rescale_coords(x: torch.Tensor, size_src, size_dst) -> torch.Tensor:
    """Map a 1-D coordinate from a frame of length ``size_src`` to one of ``size_dst``.

    Sizes may be Python numbers or tensors broadcastable against ``x`` (per-image sizes).
    """
    return (x + 0.5) * (size_dst / size_src) - 0.5


def rescale_keypoints_xy(
    xy: torch.Tensor,
    hw_src: Tuple[float, float],
    hw_dst: Tuple[float, float],
) -> torch.Tensor:
    """Rescale ``(..., 2)`` ``(x, y)`` keypoints between two resolutions ``(H, W)``."""
    if xy.shape[-1] != 2:
        raise ValueError(f"Expected (..., 2) keypoints, got {tuple(xy.shape)}")
    h_s, w_s = float(hw_src[0]), float(hw_src[1])
    h_d, w_d = float(hw_dst[0]), float(hw_dst[1])
    if min(h_s, w_s, h_d, w_d) <= 0:
        raise ValueError("Heights/widths must be positive.")
    return torch.stack(
        [rescale_coords(xy[..., 0], w_s, w_d), rescale_coords(xy[..., 1], h_s, h_d)], dim=-1
    )


def feat_coords_to_grid(xf: torch.Tensor, size_feat: int) -> torch.Tensor:
    """Feature-cell coordinate -> ``grid_sample`` coordinate for ``align_corners=False``.

    Cell centre ``i`` sits at ``(2i + 1)/Wf - 1``, so the edges of the grid are ``-1``/``+1``.
    """
    return (2.0 * xf + 1.0) / float(size_feat) - 1.0


__all__ = ["feat_coords_to_grid", "rescale_coords", "rescale_keypoints_xy"]
