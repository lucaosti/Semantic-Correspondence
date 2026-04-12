"""
Window Soft-Argmax (inference-only refinement for Task 3).

This module is **strictly post-processing** on similarity maps: it does not train anything
and must not be used inside a loss during fine-tuning (see ``docs/info.md``).

The procedure follows GeoAware-SC / "Telling Left from Right" style refinement:

1. Find the discrete peak with ``argmax`` on the similarity map.
2. Crop a square patch of side ``window_size`` around the peak (with boundary handling).
3. Apply softmax with temperature over the window, then take the expected ``(x, y)``
   position (soft-argmax) in **feature-grid** coordinates.
4. Convert the refined position to **input image pixel** coordinates (same convention as
   :mod:`models.common.matching`).
"""

from __future__ import annotations

from typing import Tuple

import torch


def window_soft_argmax_xy(
    sim_map_hw: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    window_size: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Refine one similarity map of shape ``(Hf, Wf)`` to a sub-pixel ``(x, y)`` in pixel space.

    Parameters
    ----------
    sim_map_hw:
        Tensor shaped ``(Hf, Wf)`` or ``(1, Hf, Wf)``.
    img_hw:
        ``(H, W)`` of the **model input image** (height, width) for the correspondence.
    window_size:
        Odd side length (e.g., ``5`` or ``7``); if even, it is bumped up by one.
    temperature:
        Softmax temperature on the window logits (lower = sharper, higher = softer).

    Returns
    -------
    torch.Tensor
        ``(2,)`` tensor ``(x, y)`` in pixel coordinates (float32).
    """
    if sim_map_hw.dim() == 3:
        if sim_map_hw.shape[0] != 1:
            raise ValueError("Expected sim_map_hw with shape (1,Hf,Wf) or (Hf,Wf).")
        sim_map_hw = sim_map_hw[0]
    if sim_map_hw.dim() != 2:
        raise ValueError(f"Expected a 2D map, got {tuple(sim_map_hw.shape)}")

    hf, wf = int(sim_map_hw.shape[0]), int(sim_map_hw.shape[1])
    h_img, w_img = int(img_hw[0]), int(img_hw[1])

    ws = int(window_size)
    if ws % 2 == 0:
        ws += 1
    r = ws // 2

    flat = sim_map_hw.reshape(-1)
    peak = int(torch.argmax(flat).item())
    py = peak // wf
    px = peak % wf

    y0 = max(0, py - r)
    x0 = max(0, px - r)
    y1 = min(hf, py + r + 1)
    x1 = min(wf, px + r + 1)
    patch = sim_map_hw[y0:y1, x0:x1]

    logits = patch.reshape(-1) / max(float(temperature), 1e-6)
    w = torch.softmax(logits, dim=0)

    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, device=sim_map_hw.device, dtype=torch.float32),
        torch.arange(x0, x1, device=sim_map_hw.device, dtype=torch.float32),
        indexing="ij",
    )
    gx = torch.sum(w * xx.reshape(-1))
    gy = torch.sum(w * yy.reshape(-1))

    x_pix = (gx + 0.5) * (float(w_img) / float(wf)) - 0.5
    y_pix = (gy + 0.5) * (float(h_img) / float(hf)) - 0.5
    return torch.stack([x_pix, y_pix], dim=0)


def refine_predictions_window_soft_argmax(
    sim_maps: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    window_size: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Apply :func:`window_soft_argmax_xy` to a batch of per-point similarity maps.

    Parameters
    ----------
    sim_maps:
        ``(N, Hf, Wf)`` similarity maps (same order as source keypoints).
    img_hw:
        ``(H, W)`` input image size.
    window_size, temperature:
        Forwarded to :func:`window_soft_argmax_xy`.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` refined target points in pixel coordinates.
    """
    if sim_maps.dim() != 3:
        raise ValueError(f"Expected (N,Hf,Wf), got {tuple(sim_maps.shape)}")
    out = []
    for i in range(sim_maps.shape[0]):
        out.append(
            window_soft_argmax_xy(
                sim_maps[i],
                img_hw,
                window_size=window_size,
                temperature=temperature,
            )
        )
    return torch.stack(out, dim=0)


__all__ = ["refine_predictions_window_soft_argmax", "window_soft_argmax_xy"]
