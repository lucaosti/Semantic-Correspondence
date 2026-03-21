"""
Training-step helpers for supervised correspondence (Gaussian CE on similarity maps).

The helpers are written to be backbone-agnostic as long as dense ``(1, C, Hf, Wf)`` maps
and DINO-style extraction are available.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.common.coord_utils import rescale_keypoints_xy
from models.common.input_norm import imagenet_to_sam_input
from models.common.matching import match_cosine_similarity_map, sample_features_bilinear
from models.common.vit_intermediate import extract_intermediate_dense_grid
from models.sam.backbone import extract_dense_grid_sam
from training.losses import gaussian_ce_loss_from_similarity_maps

from data.dataset import INVALID_KP_COORD


def _valid_keypoint_mask(tgt_kps: torch.Tensor, n_valid: int) -> torch.Tensor:
    """Build a boolean mask for the first ``n_valid`` keypoints (ignore padded slots)."""
    n = int(tgt_kps.shape[0])
    mask = torch.zeros((n,), dtype=torch.bool, device=tgt_kps.device)
    if n_valid > 0:
        mask[:n_valid] = (tgt_kps[:n_valid, 0] > INVALID_KP_COORD) & (tgt_kps[:n_valid, 1] > INVALID_KP_COORD)
    return mask


def correspondence_gaussian_loss_dino_vit(
    backbone: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    layer_indices,
    sigma_feat: float = 1.0,
) -> torch.Tensor:
    """
    Compute the Gaussian CE loss for a **single example** (batch size ``1``).

    Parameters
    ----------
    backbone:
        DINO ViT with ``get_intermediate_layers``.
    batch:
        Dictionary from :class:`data.dataset.SPair71kPairDataset` **with batch dimension 1**.
    layer_indices:
        Passed to ``extract_intermediate_dense_grid``.
    sigma_feat:
        Target Gaussian width in feature-map units.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    src = batch["src_img"]
    tgt = batch["tgt_img"]
    if src.shape[0] != 1 or tgt.shape[0] != 1:
        raise ValueError("This helper currently expects batch size 1.")

    feats_src, _ = extract_intermediate_dense_grid(
        backbone, src, layer_indices=layer_indices, reshape=True
    )
    feats_tgt, _ = extract_intermediate_dense_grid(
        backbone, tgt, layer_indices=layer_indices, reshape=True
    )

    src_kps = batch["src_kps"][0]
    tgt_kps = batch["tgt_kps"][0]
    n_valid = int(batch["n_valid_keypoints"][0].item())

    h_src, w_src = int(src.shape[2]), int(src.shape[3])
    h_tgt, w_tgt = int(tgt.shape[2]), int(tgt.shape[3])
    src_hw = (h_src, w_src)
    tgt_hw = (h_tgt, w_tgt)

    mask = _valid_keypoint_mask(src_kps, n_valid) & _valid_keypoint_mask(tgt_kps, n_valid)
    if not bool(mask.any()):
        return src.new_zeros(())

    pts = src_kps[mask]
    desc = sample_features_bilinear(feats_src, pts, src_hw)
    sims = match_cosine_similarity_map(desc, feats_tgt)
    gt = tgt_kps[mask]

    # Ground-truth target keypoints live in **target** image pixel space.
    return gaussian_ce_loss_from_similarity_maps(sims, gt, tgt_hw, sigma_feat=sigma_feat)


def correspondence_gaussian_loss_sam(
    image_encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    sam_input_size: int = 1024,
    sigma_feat: float = 1.0,
) -> torch.Tensor:
    """
    Gaussian CE for SAM's image encoder (batch size ``1``).

    Keypoints are rescaled from dataset resolution to SAM's fixed ``sam_input_size`` square,
    matching :class:`models.common.dense_extractor.DenseFeatureExtractor` evaluation.
    """
    src = batch["src_img"]
    tgt = batch["tgt_img"]
    if src.shape[0] != 1 or tgt.shape[0] != 1:
        raise ValueError("This helper currently expects batch size 1.")

    h_ds, w_ds = int(src.shape[2]), int(src.shape[3])
    ds_hw = (h_ds, w_ds)
    sam_hw = (sam_input_size, sam_input_size)

    src_kps = batch["src_kps"][0]
    tgt_kps = batch["tgt_kps"][0]
    n_valid = int(batch["n_valid_keypoints"][0].item())

    x_sam_src = imagenet_to_sam_input(src, target_size=sam_input_size)
    x_sam_tgt = imagenet_to_sam_input(tgt, target_size=sam_input_size)

    feats_src, _ = extract_dense_grid_sam(image_encoder, x_sam_src)
    feats_tgt, _ = extract_dense_grid_sam(image_encoder, x_sam_tgt)

    src_kps_s = rescale_keypoints_xy(src_kps, ds_hw, sam_hw)
    tgt_kps_s = rescale_keypoints_xy(tgt_kps, ds_hw, sam_hw)

    mask = _valid_keypoint_mask(src_kps, n_valid) & _valid_keypoint_mask(tgt_kps, n_valid)
    if not bool(mask.any()):
        return src.new_zeros(())

    pts = src_kps_s[mask]
    desc = sample_features_bilinear(feats_src, pts, sam_hw)
    sims = match_cosine_similarity_map(desc, feats_tgt)
    gt = tgt_kps_s[mask]

    return gaussian_ce_loss_from_similarity_maps(sims, gt, sam_hw, sigma_feat=sigma_feat)


__all__ = ["correspondence_gaussian_loss_dino_vit", "correspondence_gaussian_loss_sam"]
