"""
Training-free semantic correspondence via cosine similarity + argmax.

Coordinates are expressed in **input image pixel space** (same frame as dataset keypoints
after preprocessing).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def sample_features_bilinear(
    feat_map: torch.Tensor,
    xy_pix: torch.Tensor,
    img_hw: Tuple[int, int],
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Sample per-channel features at continuous locations using bilinear interpolation.

    Parameters
    ----------
    feat_map:
        ``(B, C, Hf, Wf)`` dense features.
    xy_pix:
        ``(N, 2)`` keypoints ``(x, y)`` in **input image pixel coordinates** matching ``img_hw``.
    img_hw:
        ``(H, W)`` of the **model input image** (height, width).
    align_corners:
        Passed to ``grid_sample`` (PyTorch default for spatial transformers is often ``False``).

    Returns
    -------
    torch.Tensor
        ``(N, C)`` sampled vectors (same batch slice ``B=1`` expected; if ``B>1``, provide
        one batch index at a time).
    """
    if feat_map.dim() != 4:
        raise ValueError(f"feat_map must be 4D, got {feat_map.shape}")
    b, c, hf, wf = feat_map.shape
    if b != 1:
        raise ValueError("This helper currently expects B=1 (call inside a loop for B>1).")
    if xy_pix.dim() != 2 or xy_pix.shape[1] != 2:
        raise ValueError(f"xy_pix must be (N,2), got {tuple(xy_pix.shape)}")

    h, w = int(img_hw[0]), int(img_hw[1])
    x = xy_pix[:, 0]
    y = xy_pix[:, 1]
    xf = (x + 0.5) * (wf / float(w)) - 0.5
    yf = (y + 0.5) * (hf / float(h)) - 0.5

    gx = (xf / max(wf - 1, 1)) * 2.0 - 1.0
    gy = (yf / max(hf - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2).to(dtype=feat_map.dtype, device=feat_map.device)

    sampled = F.grid_sample(feat_map, grid, mode="bilinear", padding_mode="border", align_corners=align_corners)
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def match_cosine_similarity_map(
    feat_src_pts: torch.Tensor,
    feat_tgt: torch.Tensor,
) -> torch.Tensor:
    """
    Compute a dense cosine similarity map for each source descriptor.

    Parameters
    ----------
    feat_src_pts:
        ``(N, C)`` L2-normalized descriptors.
    feat_tgt:
        ``(1, C, Hf, Wf)`` L2-normalized target map.

    Returns
    -------
    torch.Tensor
        ``(N, Hf, Wf)`` similarity maps.
    """
    if feat_tgt.shape[0] != 1:
        raise ValueError("Expected feat_tgt with batch size 1.")
    if feat_src_pts.dim() != 2:
        raise ValueError("feat_src_pts must be (N, C).")
    return (feat_tgt * feat_src_pts[:, :, None, None]).sum(dim=1)


def argmax_to_pixel_xy(
    sim_map: torch.Tensor,
    img_hw: Tuple[int, int],
    feat_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Convert argmax indices on a feature map to **pixel coordinates** on the input image.

    Parameters
    ----------
    sim_map:
        ``(Hf, Wf)`` or ``(N, Hf, Wf)`` similarity map(s).
    img_hw:
        ``(H, W)`` input image size.
    feat_hw:
        ``(Hf, Wf)`` feature map size.

    Returns
    -------
    torch.Tensor
        ``(2,)`` or ``(N, 2)`` tensor of ``(x, y)`` pixel coordinates (float32).
    """
    single = sim_map.dim() == 2
    sm = sim_map.unsqueeze(0) if single else sim_map
    hf, wf = int(feat_hw[0]), int(feat_hw[1])
    h, w = int(img_hw[0]), int(img_hw[1])

    flat = sm.reshape(sm.shape[0], -1)
    idx = torch.argmax(flat, dim=1)
    xi = (idx % wf).to(dtype=torch.float32)
    yi = (idx // wf).to(dtype=torch.float32)

    x_pix = (xi + 0.5) * (float(w) / float(wf)) - 0.5
    y_pix = (yi + 0.5) * (float(h) / float(hf)) - 0.5
    out = torch.stack([x_pix, y_pix], dim=-1)
    return out[0] if single else out


def predict_correspondences_cosine_argmax(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    src_kps_xy: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    img_hw_src: Optional[Tuple[int, int]] = None,
    img_hw_tgt: Optional[Tuple[int, int]] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Training-free matcher: sample source descriptors, score against target map, argmax.

    Parameters
    ----------
    feat_src, feat_tgt:
        ``(1, C, Hf, Wf)`` L2-normalized dense features for the source/target images.
    src_kps_xy:
        ``(N, 2)`` source keypoints in the **source** pixel frame (see ``img_hw_src``).
    img_hw:
        Default ``(H, W)`` used when ``img_hw_src`` / ``img_hw_tgt`` are omitted (backward compatible).
    img_hw_src:
        ``(H, W)`` for mapping ``src_kps_xy`` into ``feat_src``. Defaults to ``img_hw``.
    img_hw_tgt:
        ``(H, W)`` for converting argmax indices on ``feat_tgt`` to target pixel coordinates.
        Defaults to ``img_hw`` (or ``img_hw_src`` when only one spatial size is used).
    valid_mask:
        Optional boolean mask ``(N,)`` selecting keypoints to match.

    Returns
    -------
    dict[str, torch.Tensor]
        ``pred_tgt_xy`` ``(N, 2)``, ``sim_maps`` ``(N, Hf, Wf)`` (invalid rows may be zero).
    """
    if valid_mask is None:
        valid_mask = torch.ones((src_kps_xy.shape[0],), dtype=torch.bool, device=src_kps_xy.device)

    hs = img_hw_src if img_hw_src is not None else img_hw
    ht = img_hw_tgt if img_hw_tgt is not None else img_hw

    hf, wf = feat_tgt.shape[-2], feat_tgt.shape[-1]
    pred = torch.zeros_like(src_kps_xy)
    sims = torch.zeros((src_kps_xy.shape[0], hf, wf), dtype=feat_tgt.dtype, device=feat_tgt.device)

    if bool(valid_mask.any()):
        pts = src_kps_xy[valid_mask]
        desc = sample_features_bilinear(feat_src, pts, hs)
        smaps = match_cosine_similarity_map(desc, feat_tgt)
        pred_masked = argmax_to_pixel_xy(smaps, ht, (hf, wf))
        pred[valid_mask] = pred_masked
        sims[valid_mask] = smaps

    return {"pred_tgt_xy": pred, "sim_maps": sims}
