"""
Training-step helpers for supervised correspondence (Gaussian CE on similarity maps).

The helpers are written to be backbone-agnostic as long as dense ``(1, C, Hf, Wf)`` maps
and DINO-style extraction are available.
"""

from __future__ import annotations

from typing import Dict, Tuple

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


def _gaussian_loss_dino_single_from_feats(
    feats_src_1: torch.Tensor,
    feats_tgt_1: torch.Tensor,
    src_kps: torch.Tensor,
    tgt_kps: torch.Tensor,
    n_valid: int,
    src_hw: Tuple[int, int],
    tgt_hw: Tuple[int, int],
    *,
    sigma_feat: float,
) -> torch.Tensor:
    """Scalar loss for one pair given dense features ``(1, C, Hf, Wf)``."""
    mask = _valid_keypoint_mask(src_kps, n_valid) & _valid_keypoint_mask(tgt_kps, n_valid)
    if not bool(mask.any()):
        return feats_src_1.new_zeros(())
    pts = src_kps[mask]
    desc = sample_features_bilinear(feats_src_1, pts, src_hw)
    sims = match_cosine_similarity_map(desc, feats_tgt_1)
    gt = tgt_kps[mask]
    return gaussian_ce_loss_from_similarity_maps(sims, gt, tgt_hw, sigma_feat=sigma_feat)


def correspondence_gaussian_loss_dino_vit(
    backbone: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    layer_indices,
    sigma_feat: float = 1.0,
) -> torch.Tensor:
    """
    Gaussian CE loss for DINO ViT: one or more pairs per step.

    Runs a **batched** forward through the backbone, then averages per-pair losses (each pair
    may have a different number of valid keypoints).

    Parameters
    ----------
    backbone:
        DINO ViT with ``get_intermediate_layers``.
    batch:
        Batched dictionary from :class:`data.dataset.SPair71kPairDataset` (``spair_collate_fn``).
    layer_indices:
        Passed to ``extract_intermediate_dense_grid``.
    sigma_feat:
        Target Gaussian width in feature-map units.

    Returns
    -------
    torch.Tensor
        Scalar loss (mean over batch).
    """
    src = batch["src_img"]
    tgt = batch["tgt_img"]
    bsz = int(src.shape[0])
    if tgt.shape[0] != bsz:
        raise ValueError("src/tgt batch mismatch.")

    feats_src, _ = extract_intermediate_dense_grid(
        backbone, src, layer_indices=layer_indices, reshape=True
    )
    feats_tgt, _ = extract_intermediate_dense_grid(
        backbone, tgt, layer_indices=layer_indices, reshape=True
    )

    h_src, w_src = int(src.shape[2]), int(src.shape[3])
    h_tgt, w_tgt = int(tgt.shape[2]), int(tgt.shape[3])
    src_hw = (h_src, w_src)
    tgt_hw = (h_tgt, w_tgt)

    losses: list[torch.Tensor] = []
    for b in range(bsz):
        lb = _gaussian_loss_dino_single_from_feats(
            feats_src[b : b + 1],
            feats_tgt[b : b + 1],
            batch["src_kps"][b],
            batch["tgt_kps"][b],
            int(batch["n_valid_keypoints"][b].item()),
            src_hw,
            tgt_hw,
            sigma_feat=sigma_feat,
        )
        losses.append(lb)
    return torch.stack(losses).mean()


def correspondence_gaussian_loss_sam(
    image_encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    sam_input_size: int = 1024,
    sigma_feat: float = 1.0,
) -> torch.Tensor:
    """
    Gaussian CE for SAM's image encoder (one or more pairs per step).

    Keypoints are rescaled from dataset resolution to SAM's fixed ``sam_input_size`` square,
    matching :class:`models.common.dense_extractor.DenseFeatureExtractor` evaluation.
    """
    src = batch["src_img"]
    tgt = batch["tgt_img"]
    bsz = int(src.shape[0])
    if tgt.shape[0] != bsz:
        raise ValueError("src/tgt batch mismatch.")

    h_ds, w_ds = int(src.shape[2]), int(src.shape[3])
    ds_hw = (h_ds, w_ds)
    sam_hw = (sam_input_size, sam_input_size)

    x_sam_src = imagenet_to_sam_input(src, target_size=sam_input_size)
    x_sam_tgt = imagenet_to_sam_input(tgt, target_size=sam_input_size)

    feats_src, _ = extract_dense_grid_sam(image_encoder, x_sam_src)
    feats_tgt, _ = extract_dense_grid_sam(image_encoder, x_sam_tgt)

    losses: list[torch.Tensor] = []
    for b in range(bsz):
        src_kps = batch["src_kps"][b]
        tgt_kps = batch["tgt_kps"][b]
        n_valid = int(batch["n_valid_keypoints"][b].item())
        src_kps_s = rescale_keypoints_xy(src_kps, ds_hw, sam_hw)
        tgt_kps_s = rescale_keypoints_xy(tgt_kps, ds_hw, sam_hw)
        mask = _valid_keypoint_mask(src_kps, n_valid) & _valid_keypoint_mask(tgt_kps, n_valid)
        if not bool(mask.any()):
            losses.append(src.new_zeros(()))
            continue
        pts = src_kps_s[mask]
        fs = feats_src[b : b + 1]
        ft = feats_tgt[b : b + 1]
        desc = sample_features_bilinear(fs, pts, sam_hw)
        sims = match_cosine_similarity_map(desc, ft)
        gt = tgt_kps_s[mask]
        losses.append(gaussian_ce_loss_from_similarity_maps(sims, gt, sam_hw, sigma_feat=sigma_feat))
    return torch.stack(losses).mean()


__all__ = ["correspondence_gaussian_loss_dino_vit", "correspondence_gaussian_loss_sam"]
