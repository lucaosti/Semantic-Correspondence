"""
DINOv2 ViT-B/14 backbone — implementation under ``models/dinov2/`` (see ``PAPERS.md``).

Weights load from Meta URLs or a local path; do not use Hugging Face checkpoints (``docs/info.md``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch.nn as nn

from models.common.vit_intermediate import extract_intermediate_dense_grid
from models.dinov2.hub_loader import dinov2_vitb14


def build_dinov2_vit_b14(
    *,
    pretrained: bool = True,
    weights_path: Optional[str] = None,
) -> nn.Module:
    """
    Build DINOv2 ViT-B/14.

    Parameters
    ----------
    pretrained:
        If ``True`` and ``weights_path`` is ``None``, loads official pretrained weights
        via the DINOv2 hub helpers (downloads once).
    weights_path:
        Optional local checkpoint path forwarded to the hub loader as ``weights=``.

    Returns
    -------
    torch.nn.Module
        A DINOv2 ViT backbone module.
    """
    if weights_path is not None:
        return dinov2_vitb14(pretrained=True, weights=weights_path)
    return dinov2_vitb14(pretrained=pretrained)


def extract_dense_grid(
    model: nn.Module,
    x_imagenet: torch.Tensor,
    *,
    layer_indices: Union[int, Sequence[int]] = 4,
    reshape: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Extract dense patch-grid features for a DINOv2 ViT using intermediate layers.

    Parameters
    ----------
    model:
        A DINOv2 ``DinoVisionTransformer`` instance.
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
    return extract_intermediate_dense_grid(
        model,
        x_imagenet,
        layer_indices=layer_indices,
        reshape=reshape,
    )


__all__ = ["build_dinov2_vit_b14", "extract_dense_grid"]
