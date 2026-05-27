"""
DINOv3 ViT-B/16 backbone — implementation under ``models/dinov3/`` (see ``PAPERS.md``).

Architecture and weight URLs follow the official DINOv3 release; pretrained access may require Meta approval.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch.nn as nn

from models.common.vit_intermediate import extract_intermediate_dense_grid
from models.dinov3.hub_loader import dinov3_vits16, dinov3_vitb16, dinov3_vitl16


def _build_dinov3(loader_fn, *, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    if weights_path is not None:
        return loader_fn(pretrained=True, weights=weights_path)
    return loader_fn(pretrained=pretrained)


def build_dinov3_vit_s16(*, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build DINOv3 ViT-S/16."""
    return _build_dinov3(dinov3_vits16, pretrained=pretrained, weights_path=weights_path)


def build_dinov3_vit_b16(*, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build DINOv3 ViT-B/16."""
    return _build_dinov3(dinov3_vitb16, pretrained=pretrained, weights_path=weights_path)


def build_dinov3_vit_l16(*, pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build DINOv3 ViT-L/16."""
    return _build_dinov3(dinov3_vitl16, pretrained=pretrained, weights_path=weights_path)


def extract_dense_grid_dinov3(
    model: nn.Module,
    x_imagenet: torch.Tensor,
    *,
    layer_indices: Union[int, Sequence[int]] = 4,
    reshape: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Extract dense features from a DINOv3 ViT using ``get_intermediate_layers``."""
    if not hasattr(model, "get_intermediate_layers"):
        raise TypeError("Expected a DINOv3 ViT module with `get_intermediate_layers`.")
    return extract_intermediate_dense_grid(
        model,
        x_imagenet,
        layer_indices=layer_indices,
        reshape=reshape,
    )


__all__ = [
    "build_dinov3_vit_s16",
    "build_dinov3_vit_b16",
    "build_dinov3_vit_l16",
    "extract_dense_grid_dinov3",
]
