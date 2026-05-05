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

Implementation note
-------------------
The batched helper :func:`refine_predictions_window_soft_argmax` is fully vectorized
across the ``N`` keypoints (no Python loop, no host/device synchronization). At eval
time SPair-71k yields up to 20 keypoints per pair times tens of thousands of pairs,
so removing the inner ``.item()`` saves a six-figure number of CUDA syncs per run.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _vectorized_window_soft_argmax(
    sim_maps: torch.Tensor,
    *,
    window_size: int,
    temperature: float,
) -> torch.Tensor:
    """Return ``(N, 2)`` refined ``(xf, yf)`` in **feature-grid** coordinates.

    Parameters
    ----------
    sim_maps:
        ``(N, Hf, Wf)`` similarity maps (one per keypoint).
    window_size:
        Side length of the square window around each peak (forced odd, ``>= 1``).
    temperature:
        Softmax temperature on the window logits (lower = sharper).
    """
    if sim_maps.dim() != 3:
        raise ValueError(f"Expected (N,Hf,Wf), got {tuple(sim_maps.shape)}")

    n, hf, wf = sim_maps.shape
    ws = int(window_size)
    if ws < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if ws % 2 == 0:
        ws += 1
    r = ws // 2

    flat = sim_maps.reshape(n, -1)
    peak = torch.argmax(flat, dim=1)  # (N,)
    py = (peak // wf).to(torch.long)
    px = (peak % wf).to(torch.long)

    # Pad once with very-negative logits so out-of-bounds positions vanish under softmax.
    neg_inf = torch.finfo(sim_maps.dtype).min / 4.0
    padded = F.pad(sim_maps, (r, r, r, r), value=neg_inf)  # (N, Hf+2r, Wf+2r)

    arange = torch.arange(ws, device=sim_maps.device, dtype=torch.long)  # (ws,)
    rows = py.view(n, 1, 1) + arange.view(1, ws, 1)  # (N, ws, 1)
    cols = px.view(n, 1, 1) + arange.view(1, 1, ws)  # (N, 1, ws)

    # Gather (N, ws, ws) windows via fancy indexing.
    n_idx = torch.arange(n, device=sim_maps.device, dtype=torch.long).view(n, 1, 1)
    patches = padded[n_idx, rows, cols]  # (N, ws, ws)

    logits = patches.reshape(n, -1) / max(float(temperature), 1e-6)
    weights = torch.softmax(logits, dim=1).reshape(n, ws, ws)

    # Coordinates of each window cell in **original feature-grid** space.
    # rows/cols above are in padded space; subtract r to map back.
    yy_feat = (rows.float() - float(r)).expand(n, ws, ws)  # (N, ws, ws)
    xx_feat = (cols.float() - float(r)).expand(n, ws, ws)
    gx = (weights * xx_feat).sum(dim=(1, 2))  # (N,)
    gy = (weights * yy_feat).sum(dim=(1, 2))  # (N,)
    return torch.stack([gx, gy], dim=-1)


def _feat_xy_to_pixel_xy(
    feat_xy: torch.Tensor,
    img_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
) -> torch.Tensor:
    """Map ``(N, 2)`` feature-grid coords back to input-image pixel coords."""
    h_img, w_img = int(img_hw[0]), int(img_hw[1])
    hf, wf = int(feat_hw[0]), int(feat_hw[1])
    out = feat_xy.clone()
    out[..., 0] = (out[..., 0] + 0.5) * (float(w_img) / float(wf)) - 0.5
    out[..., 1] = (out[..., 1] + 0.5) * (float(h_img) / float(hf)) - 0.5
    return out


def window_soft_argmax_xy(
    sim_map_hw: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    window_size: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Refine **one** similarity map of shape ``(Hf, Wf)`` to a sub-pixel ``(x, y)``.

    Thin wrapper over :func:`_vectorized_window_soft_argmax` for the single-map case;
    use :func:`refine_predictions_window_soft_argmax` for batched use.

    Returns ``(2,)`` tensor ``(x, y)`` in pixel coordinates (float32).
    """
    if sim_map_hw.dim() == 3:
        if sim_map_hw.shape[0] != 1:
            raise ValueError("Expected sim_map_hw with shape (1,Hf,Wf) or (Hf,Wf).")
        sim_map_hw = sim_map_hw[0]
    if sim_map_hw.dim() != 2:
        raise ValueError(f"Expected a 2D map, got {tuple(sim_map_hw.shape)}")

    feat_xy = _vectorized_window_soft_argmax(
        sim_map_hw.unsqueeze(0),
        window_size=window_size,
        temperature=temperature,
    )
    pix = _feat_xy_to_pixel_xy(feat_xy, img_hw, sim_map_hw.shape)
    return pix[0].to(torch.float32)


def refine_predictions_window_soft_argmax(
    sim_maps: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    window_size: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply window soft-argmax to a batch of per-keypoint similarity maps.

    Parameters
    ----------
    sim_maps:
        ``(N, Hf, Wf)`` similarity maps (same order as source keypoints).
    img_hw:
        ``(H, W)`` input image size (height, width).
    window_size, temperature:
        Forwarded to the inner kernel; see module docstring.

    Returns
    -------
    torch.Tensor
        ``(N, 2)`` refined target points in pixel coordinates (float32).
    """
    if sim_maps.dim() != 3:
        raise ValueError(f"Expected (N,Hf,Wf), got {tuple(sim_maps.shape)}")
    if sim_maps.shape[0] == 0:
        return sim_maps.new_zeros((0, 2), dtype=torch.float32)
    feat_xy = _vectorized_window_soft_argmax(
        sim_maps, window_size=window_size, temperature=temperature
    )
    pix = _feat_xy_to_pixel_xy(feat_xy, img_hw, sim_maps.shape[-2:])
    return pix.to(torch.float32)


__all__ = ["refine_predictions_window_soft_argmax", "window_soft_argmax_xy"]
