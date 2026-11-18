"""
Training-free semantic correspondence via cosine similarity + argmax.

All keypoints are expressed in the **dataset pixel frame** (the preprocessed image fed to
the extractor). Every backbone's feature grid covers that whole image (SAM's internal
1024 resize is a uniform rescale of the same image), so pixels map to feature cells with
the pixel-centre convention of :mod:`models.common.coord_utils`. Matching always runs in
fp32, also when the backbone forward ran under autocast.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from models.common.coord_utils import feat_coords_to_grid, rescale_coords


def sample_descriptors(
    feats: torch.Tensor,
    kps_xy: torch.Tensor,
    img_hw: Tuple[int, int],
) -> torch.Tensor:
    """Bilinearly sample ``(B, N, C)`` descriptors at ``(B, N, 2)`` pixel keypoints.

    Points outside the image are clamped to the border cells (``padding_mode="border"``);
    MPS lacks that mode and falls back to zero padding.
    """
    if feats.dim() != 4 or kps_xy.dim() != 3 or kps_xy.shape[-1] != 2:
        raise ValueError(f"Expected feats (B,C,Hf,Wf) and kps (B,N,2), got {tuple(feats.shape)}, {tuple(kps_xy.shape)}")
    hf, wf = int(feats.shape[-2]), int(feats.shape[-1])
    h, w = float(img_hw[0]), float(img_hw[1])
    gx = feat_coords_to_grid(rescale_coords(kps_xy[..., 0].float(), w, wf), wf)
    gy = feat_coords_to_grid(rescale_coords(kps_xy[..., 1].float(), h, hf), hf)
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # (B, N, 1, 2)
    padding_mode = "zeros" if feats.device.type == "mps" else "border"
    sampled = F.grid_sample(
        feats.float(), grid, mode="bilinear", padding_mode=padding_mode, align_corners=False
    )  # (B, C, N, 1)
    return sampled.squeeze(-1).transpose(1, 2)


def similarity_maps(desc: torch.Tensor, feats_tgt: torch.Tensor) -> torch.Tensor:
    """``(B, N, C)`` x ``(B, C, Hf, Wf)`` -> ``(B, N, Hf, Wf)`` cosine maps in fp32.

    Both inputs are L2-normalized upstream. Autocast is disabled so the maps are never
    rounded to bf16/fp16.
    """
    with torch.autocast(device_type=feats_tgt.device.type, enabled=False):
        return torch.einsum("bnc,bchw->bnhw", desc.float(), feats_tgt.float())


def argmax_to_pixel_xy(sim_maps: torch.Tensor, img_hw: Tuple[int, int]) -> torch.Tensor:
    """``(..., Hf, Wf)`` similarity maps -> ``(..., 2)`` pixel ``(x, y)`` of the argmax cell centre."""
    hf, wf = int(sim_maps.shape[-2]), int(sim_maps.shape[-1])
    idx = torch.argmax(sim_maps.reshape(*sim_maps.shape[:-2], hf * wf), dim=-1)
    xi = (idx % wf).to(torch.float32)
    yi = torch.div(idx, wf, rounding_mode="floor").to(torch.float32)
    return torch.stack(
        [rescale_coords(xi, wf, float(img_hw[1])), rescale_coords(yi, hf, float(img_hw[0]))], dim=-1
    )


def sample_features_bilinear(
    feat_map: torch.Tensor,
    xy_pix: torch.Tensor,
    img_hw: Tuple[int, int],
) -> torch.Tensor:
    """Single-image wrapper: ``(1, C, Hf, Wf)`` and ``(N, 2)`` -> ``(N, C)``."""
    if feat_map.dim() != 4 or feat_map.shape[0] != 1:
        raise ValueError(f"feat_map must be (1, C, Hf, Wf), got {tuple(feat_map.shape)}")
    if xy_pix.dim() != 2 or xy_pix.shape[1] != 2:
        raise ValueError(f"xy_pix must be (N,2), got {tuple(xy_pix.shape)}")
    return sample_descriptors(feat_map, xy_pix.unsqueeze(0), img_hw)[0]


def predict_correspondences_cosine_argmax(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    src_kps_xy: torch.Tensor,
    img_hw: Tuple[int, int],
    *,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Single-pair matcher on ``(1, C, Hf, Wf)`` maps; both images share ``img_hw``.

    Returns ``pred_tgt_xy`` ``(N, 2)`` and ``sim_maps`` ``(N, Hf, Wf)``; rows outside
    ``valid_mask`` are zero.
    """
    n = src_kps_xy.shape[0]
    if valid_mask is None:
        valid_mask = torch.ones((n,), dtype=torch.bool, device=src_kps_xy.device)
    hf, wf = feat_tgt.shape[-2], feat_tgt.shape[-1]
    pred = torch.zeros((n, 2), dtype=torch.float32, device=src_kps_xy.device)
    sims = torch.zeros((n, hf, wf), dtype=torch.float32, device=feat_tgt.device)
    if bool(valid_mask.any()):
        desc = sample_descriptors(feat_src, src_kps_xy[valid_mask].unsqueeze(0), img_hw)
        smaps = similarity_maps(desc, feat_tgt)[0]
        pred[valid_mask] = argmax_to_pixel_xy(smaps, img_hw)
        sims[valid_mask] = smaps
    return {"pred_tgt_xy": pred, "sim_maps": sims}


__all__ = [
    "argmax_to_pixel_xy",
    "predict_correspondences_cosine_argmax",
    "sample_descriptors",
    "sample_features_bilinear",
    "similarity_maps",
]
