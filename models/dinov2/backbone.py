"""
DINOv2 ViT-B/14 backbone — implementation under ``models/dinov2/`` (see ``PAPERS.md``).

Weights load from Meta URLs or a local path; do not use Hugging Face checkpoints (``docs/info.md``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch.nn as nn

from models.common.vit_intermediate import extract_intermediate_dense_grid
from models.dinov2.hub_loader import dinov2_vitb14, dinov2_vits14, dinov2_vitl14


def _build_dinov2(loader_fn, *, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    if weights_path is not None:
        return loader_fn(pretrained=True, weights=weights_path)
    return loader_fn(pretrained=pretrained)


def build_dinov2_vit_s14(*, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build DINOv2 ViT-S/14."""
    return _build_dinov2(dinov2_vits14, pretrained=pretrained, weights_path=weights_path)


def build_dinov2_vit_b14(*, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build DINOv2 ViT-B/14."""
    return _build_dinov2(dinov2_vitb14, pretrained=pretrained, weights_path=weights_path)


def build_dinov2_vit_l14(*, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build DINOv2 ViT-L/14."""
    return _build_dinov2(dinov2_vitl14, pretrained=pretrained, weights_path=weights_path)


def extract_dense_grid(
    model: nn.Module,
    x_imagenet: torch.Tensor,
    *,
    layer_indices: Union[int, Sequence[int]] = 4,
    reshape: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Extract dense patch-grid features for a DINOv2 ViT using intermediate layers."""
    return extract_intermediate_dense_grid(
        model,
        x_imagenet,
        layer_indices=layer_indices,
        reshape=reshape,
    )


__all__ = [
    "build_dinov2_vit_s14",
    "build_dinov2_vit_b14",
    "build_dinov2_vit_l14",
    "extract_dense_grid",
]
