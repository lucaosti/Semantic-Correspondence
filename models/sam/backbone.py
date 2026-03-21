"""
SAM ViT-B image encoder — implementation under ``models/sam/`` (see ``PAPERS.md``).

Uses ``ImageEncoderViT`` from the integrated Segment Anything modeling code; the full
``Sam`` model is built to load official checkpoints, then only ``image_encoder`` is returned.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam.build_sam import build_sam_vit_b


def build_sam_vit_b_image_encoder(
    *,
    checkpoint_path: Optional[str] = None,
) -> nn.Module:
    """
    Build **only** the SAM ViT-B image encoder (``vit_b``).

    Parameters
    ----------
    checkpoint_path:
        Optional path to the official ``sam_vit_b_01ec64.pth`` checkpoint. If ``None``,
        the full SAM is randomly initialized (encoder weights not trained — use only for tests).

    Returns
    -------
    torch.nn.Module
        ``ImageEncoderViT`` module (neck output channels = 256).
    """
    sam = build_sam_vit_b(checkpoint=checkpoint_path)
    enc = sam.image_encoder
    enc.eval()
    return enc


def extract_dense_grid_sam(
    image_encoder: nn.Module,
    x_sam: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run SAM's image encoder and L2-normalize channels for cosine matching.

    Parameters
    ----------
    image_encoder:
        SAM ``ImageEncoderViT`` module.
    x_sam:
        ``(B, 3, 1024, 1024)`` tensor in SAM input space (see ``models.common.input_norm``).

    Returns
    -------
    tuple[torch.Tensor, dict]
        ``(B, C, Hf, Wf)`` normalized features and metadata.
    """
    feats = image_encoder(x_sam)
    if feats.dim() != 4:
        raise ValueError(f"Unexpected SAM encoder output rank: {feats.dim()}.")
    feats = F.normalize(feats.float(), dim=1)
    meta = {
        "channels": int(feats.shape[1]),
        "grid_hw": (int(feats.shape[2]), int(feats.shape[3])),
        "backbone": "sam_vit_b",
    }
    return feats, meta


__all__ = ["build_sam_vit_b_image_encoder", "extract_dense_grid_sam"]
