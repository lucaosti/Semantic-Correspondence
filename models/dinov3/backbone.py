"""
DINOv3 ViT-B/16 backbone — implementation under ``models/dinov3/`` (see ``PAPERS.md``).

Architecture and weight URLs follow the official DINOv3 release; pretrained access may require Meta approval.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch.nn as nn

from models.common.vit_intermediate import extract_intermediate_dense_grid
from models.dinov3.hub_loader import dinov3_vitb16


def build_dinov3_vit_b16(
    *,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Build DINOv3 ViT-B/16 (patch size 16).

    Parameters
    ----------
    pretrained:
        If ``True`` and ``weights_path`` is ``None``, uses the official hub URL.
    weights_path:
        Optional local checkpoint path forwarded to the hub loader as ``weights=``.

    Returns
    -------
    torch.nn.Module
        A DINOv3 ViT backbone module.
    """
    if weights_path is not None:
        return dinov3_vitb16(pretrained=True, weights=weights_path)
    return dinov3_vitb16(pretrained=pretrained)


def extract_dense_grid_dinov3(
    model: nn.Module,
    x_imagenet: torch.Tensor,
    *,
    layer_indices: Union[int, Sequence[int]] = 4,
    reshape: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract dense features from a DINOv3 ViT using ``get_intermediate_layers``.

    Parameters
    ----------
    model:
        A DINOv3 ``DinoVisionTransformer``-compatible module.
    x_imagenet:
        ``(B, 3, H, W)`` ImageNet-normalized batch.
    layer_indices:
        Passed through to ``get_intermediate_layers``.
    reshape:
        Request feature maps reshaped to ``(B, C, Hf, Wf)``.

    Returns
    -------
    tuple[torch.Tensor, dict]
        Fused L2-normalized features and metadata.
    """
    if not hasattr(model, "get_intermediate_layers"):
        raise TypeError("Expected a DINOv3 ViT module with `get_intermediate_layers`.")
    return extract_intermediate_dense_grid(
        model,
        x_imagenet,
        layer_indices=layer_indices,
        reshape=reshape,
    )


__all__ = ["build_dinov3_vit_b16", "extract_dense_grid_dinov3"]
