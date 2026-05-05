"""
Shared helpers for DINO-style ViT backbones (intermediate layers + fusion).

Used by both ``models/dinov2`` and ``models/dinov3`` without cross-importing those packages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_layer_features(
    feats: Sequence[torch.Tensor],
    fusion: str = "mean_l2norm",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Fuse multi-layer dense features for a training-free matcher.

    Parameters
    ----------
    feats:
        List of tensors each shaped ``(B, C, H, W)``.
    fusion:
        ``mean_l2norm``: L2-normalize each layer along channels, average across layers,
        then L2-normalize again.

    Returns
    -------
    tuple[torch.Tensor, dict]
        Fused tensor and metadata.
    """
    if len(feats) == 0:
        raise ValueError("feats must be non-empty.")
    if fusion != "mean_l2norm":
        raise ValueError(f"Unsupported fusion mode: {fusion}")

    normed = [F.normalize(f, dim=1) for f in feats]
    out = torch.stack(normed, dim=0).mean(dim=0)
    out = F.normalize(out, dim=1)
    meta = {"fusion": fusion, "num_layers": len(feats), "channels": int(out.shape[1])}
    return out, meta


def extract_intermediate_dense_grid(
    model: nn.Module,
    x_imagenet: torch.Tensor,
    *,
    layer_indices: Union[int, Sequence[int]] = 4,
    reshape: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract dense patch-grid features using a ViT's ``get_intermediate_layers`` API.

    This matches the DINOv2 / DINOv3 reference implementations.

    Parameters
    ----------
    model:
        A ViT module exposing ``get_intermediate_layers`` (e.g., official DINO checkpoints).
    x_imagenet:
        ``(B, 3, H, W)`` batch in **ImageNet-normalized** space.
    layer_indices:
        Either an ``int`` (last ``n`` blocks) or an explicit list of block indices.
    reshape:
        If ``True``, returns ``(B, C, Hf, Wf)`` feature maps.

    Returns
    -------
    tuple[torch.Tensor, dict]
        Fused L2-normalized features and metadata.
    """
    if not hasattr(model, "get_intermediate_layers"):
        raise TypeError("Expected a ViT module with `get_intermediate_layers`.")

    outputs = model.get_intermediate_layers(
        x_imagenet,
        n=layer_indices,
        reshape=reshape,
        return_class_token=False,
        norm=True,
    )
    feats: List[torch.Tensor] = [t.float() for t in outputs]
    fused, meta = fuse_layer_features(feats, fusion="mean_l2norm")
    meta["layer_indices"] = layer_indices
    return fused, meta
