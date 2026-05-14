"""
Unified dense feature extraction for multiple foundation backbones (Task 1 baseline).

Supports three backbone families (DINOv2, DINOv3, SAM) each in Small / Base / Large
size variants. All exposed via a single :class:`DenseFeatureExtractor` wrapper that
accepts ImageNet-normalized inputs and returns L2-normalized ``(B, C, Hf, Wf)`` maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.common.input_norm import imagenet_to_sam_input
from models.dinov2.backbone import (
    build_dinov2_vit_s14,
    build_dinov2_vit_b14,
    build_dinov2_vit_l14,
    extract_dense_grid as extract_dense_grid_dinov2,
)
from models.dinov3.backbone import (
    build_dinov3_vit_s16,
    build_dinov3_vit_b16,
    build_dinov3_vit_l16,
    extract_dense_grid_dinov3,
)
from models.sam.backbone import (
    build_sam_vit_b_image_encoder,
    build_sam_vit_l_image_encoder,
    extract_dense_grid_sam,
)


class BackboneName(str, Enum):
    """Supported foundation backbones (Small / Base / Large variants)."""

    # DINOv2 (patch size 14)
    DINOV2_VIT_S14 = "dinov2_vits14"
    DINOV2_VIT_B14 = "dinov2_vitb14"
    DINOV2_VIT_L14 = "dinov2_vitl14"

    # DINOv3 (patch size 16)
    DINOV3_VIT_S16 = "dinov3_vits16"
    DINOV3_VIT_B16 = "dinov3_vitb16"
    DINOV3_VIT_L16 = "dinov3_vitl16"

    # SAM (patch size 16, internal 1024×1024)
    SAM_VIT_B = "sam_vit_b"
    SAM_VIT_L = "sam_vit_l"


@dataclass(frozen=True)
class DenseExtractorConfig:
    """
    Configuration for :class:`DenseFeatureExtractor`.

    Attributes
    ----------
    name:
        Which backbone to use.
    weights_path:
        Optional local path (or URL) for the selected backbone's pretrained weights.
        For DINOv2/DINOv3, pass the ``.pth`` file path; for SAM, pass the full SAM
        checkpoint (``image_encoder.*`` keys are extracted automatically).
        If ``None``, weights are loaded from the official hub URL (downloaded once).
    dino_layer_indices:
        ``n`` (last-n blocks) or explicit block indices for DINO ``get_intermediate_layers``.
        Ignored for SAM backbones.
    """

    name: BackboneName
    weights_path: Optional[str] = None
    dino_layer_indices: Union[int, Sequence[int]] = 4


# ---------------------------------------------------------------------------
# Backbone metadata
# ---------------------------------------------------------------------------

_PATCH_SIZE: Dict[str, int] = {
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
    "dinov2_vitl14": 14,
    "dinov3_vits16": 16,
    "dinov3_vitb16": 16,
    "dinov3_vitl16": 16,
    "sam_vit_b": 16,
    "sam_vit_l": 16,
}

_DINOV2_BUILDERS = {
    "dinov2_vits14": build_dinov2_vit_s14,
    "dinov2_vitb14": build_dinov2_vit_b14,
    "dinov2_vitl14": build_dinov2_vit_l14,
}

_DINOV3_BUILDERS = {
    "dinov3_vits16": build_dinov3_vit_s16,
    "dinov3_vitb16": build_dinov3_vit_b16,
    "dinov3_vitl16": build_dinov3_vit_l16,
}

_SAM_BUILDERS = {
    "sam_vit_b": build_sam_vit_b_image_encoder,
    "sam_vit_l": build_sam_vit_l_image_encoder,
}


class DenseFeatureExtractor(nn.Module):
    """
    Wrapper that standardizes dense feature extraction for semantic correspondence.

    Inputs are **ImageNet-normalized** ``(B, 3, H, W)`` tensors (matching ``data.dataset``).

    Outputs are **L2-normalized along channel** dense maps ``(B, C, Hf, Wf)``.

    Notes
    -----
    - DINOv2 uses patch size ``14``; DINOv3 and SAM use patch size ``16``.
    - SAM internally resizes to ``1024x1024`` and uses SAM-specific normalization;
      ``coord_hw`` in the returned metadata reflects the SAM input frame.
    """

    def __init__(self, cfg: DenseExtractorConfig, *, freeze: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_name = cfg.name
        bname = cfg.name.value

        if bname in _DINOV2_BUILDERS:
            self.encoder = _DINOV2_BUILDERS[bname](
                pretrained=cfg.weights_path is None,
                weights_path=cfg.weights_path,
            )
            self._forward_impl = self._forward_dinov2
        elif bname in _DINOV3_BUILDERS:
            self.encoder = _DINOV3_BUILDERS[bname](
                pretrained=cfg.weights_path is None,
                weights_path=cfg.weights_path,
            )
            self._forward_impl = self._forward_dinov3
        elif bname in _SAM_BUILDERS:
            self.encoder = _SAM_BUILDERS[bname](checkpoint_path=cfg.weights_path)
            self._forward_impl = self._forward_sam
        else:
            raise ValueError(f"Unsupported backbone: {cfg.name}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @property
    def patch_size(self) -> int:
        """Return the patch size for the selected backbone."""
        return _PATCH_SIZE.get(self.backbone_name.value, 16)

    def forward(self, x_imagenet: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract dense features.

        Parameters
        ----------
        x_imagenet:
            ``(B, 3, H, W)`` tensor normalized with ImageNet mean/std.

        Returns
        -------
        tuple[torch.Tensor, dict]
            L2-normalized feature map ``(B, C, Hf, Wf)`` and metadata dict.
        """
        return self._forward_impl(x_imagenet)

    def _forward_dinov2(self, x_imagenet: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        feats, meta = extract_dense_grid_dinov2(
            self.encoder,
            x_imagenet,
            layer_indices=self.cfg.dino_layer_indices,
            reshape=True,
        )
        meta["backbone"] = self.backbone_name.value
        meta["patch_size"] = self.patch_size
        meta["coord_hw"] = (int(x_imagenet.shape[2]), int(x_imagenet.shape[3]))
        meta["dataset_hw"] = meta["coord_hw"]
        return feats, meta

    def _forward_dinov3(self, x_imagenet: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        feats, meta = extract_dense_grid_dinov3(
            self.encoder,
            x_imagenet,
            layer_indices=self.cfg.dino_layer_indices,
            reshape=True,
        )
        meta["backbone"] = self.backbone_name.value
        meta["patch_size"] = self.patch_size
        meta["coord_hw"] = (int(x_imagenet.shape[2]), int(x_imagenet.shape[3]))
        meta["dataset_hw"] = meta["coord_hw"]
        return feats, meta

    def _forward_sam(self, x_imagenet: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x_sam = imagenet_to_sam_input(x_imagenet, target_size=1024)
        feats, meta = extract_dense_grid_sam(
            self.encoder, x_sam, backbone_label=self.backbone_name.value
        )
        meta["patch_size"] = self.patch_size
        meta["sam_input_size"] = 1024
        meta["coord_hw"] = (1024, 1024)
        meta["dataset_hw"] = (int(x_imagenet.shape[2]), int(x_imagenet.shape[3]))
        return feats, meta


__all__ = ["BackboneName", "DenseExtractorConfig", "DenseFeatureExtractor"]
