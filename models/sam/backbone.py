"""SAM image encoders (ViT-B and ViT-L) — mask/prompt branches are unused."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sam.modeling.image_encoder import ImageEncoderViT


def _load_sam_encoder_weights(encoder: nn.Module, checkpoint_path: str) -> None:
    """Load ``image_encoder.*`` prefixed weights from a SAM checkpoint into ``encoder``."""
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    prefix = "image_encoder."
    encoder_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, _ = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing SAM image_encoder weights: {missing[:5]}...")


def build_sam_vit_b_image_encoder(*, checkpoint_path: Optional[str] = None) -> nn.Module:
    """Build ``ImageEncoderViT`` (SAM ViT-B) and load the ``image_encoder.*`` weights."""
    encoder = ImageEncoderViT(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        out_chans=256,
    )
    if checkpoint_path is not None:
        _load_sam_encoder_weights(encoder, checkpoint_path)
    encoder.eval()
    return encoder


def build_sam_vit_l_image_encoder(*, checkpoint_path: Optional[str] = None) -> nn.Module:
    """Build ``ImageEncoderViT`` (SAM ViT-L) and load the ``image_encoder.*`` weights."""
    encoder = ImageEncoderViT(
        depth=24,
        embed_dim=1024,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_heads=16,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[5, 11, 17, 23],
        window_size=14,
        out_chans=256,
    )
    if checkpoint_path is not None:
        _load_sam_encoder_weights(encoder, checkpoint_path)
    encoder.eval()
    return encoder


def extract_dense_grid_sam(
    image_encoder: nn.Module,
    x_sam: torch.Tensor,
    *,
    backbone_label: str = "sam_vit_b",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run a SAM image encoder and L2-normalize channels for cosine matching."""
    feats = image_encoder(x_sam)
    if feats.dim() != 4:
        raise ValueError(f"Unexpected SAM encoder output rank: {feats.dim()}.")
    feats = F.normalize(feats.float(), dim=1)
    meta = {
        "channels": int(feats.shape[1]),
        "grid_hw": (int(feats.shape[2]), int(feats.shape[3])),
        "backbone": backbone_label,
    }
    return feats, meta


__all__ = [
    "build_sam_vit_b_image_encoder",
    "build_sam_vit_l_image_encoder",
    "extract_dense_grid_sam",
]
