"""
Unified dense feature extraction for multiple foundation backbones (Task 1 baseline).

This module orchestrates the per-backbone packages under ``models/dinov2``, ``models/dinov3``,
and ``models/sam``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models.common.input_norm import imagenet_to_sam_input
from models.dinov2.backbone import build_dinov2_vit_b14, extract_dense_grid as extract_dense_grid_dinov2
from models.dinov3.backbone import build_dinov3_vit_b16, extract_dense_grid_dinov3
from models.sam.backbone import build_sam_vit_b_image_encoder, extract_dense_grid_sam


class BackboneName(str, Enum):
    """Supported foundation backbones for Task 1 (Base variants)."""

    DINOV2_VIT_B14 = "dinov2_vitb14"
    DINOV3_VIT_B16 = "dinov3_vitb16"
    SAM_VIT_B = "sam_vit_b"


@dataclass(frozen=True)
class DenseExtractorConfig:
    """
    Configuration for :class:`DenseFeatureExtractor`.

    Attributes
    ----------
    name:
        Which backbone to use.
    dinov2_weights_path:
        Optional local path for DINOv2 weights (forwarded to the hub loader).
    dinov3_weights_path:
        Optional local path for DINOv3 weights.
    sam_checkpoint_path:
        Optional SAM ViT-B checkpoint path (``*.pth``).
    dino_layer_indices:
        ``n`` (last-n blocks) or explicit block indices for DINO ``get_intermediate_layers``.
    """

    name: BackboneName
    dinov2_weights_path: Optional[str] = None
    dinov3_weights_path: Optional[str] = None
    sam_checkpoint_path: Optional[str] = None
    dino_layer_indices: Union[int, Sequence[int]] = 4


class DenseFeatureExtractor(nn.Module):
    """
    Wrapper that standardizes dense feature extraction for semantic correspondence.

    Inputs are **ImageNet-normalized** ``(B, 3, H, W)`` tensors (matching ``data.dataset``).

    Outputs are **L2-normalized along channel** dense maps ``(B, C, Hf, Wf)``.

    Notes
    -----
    - DINOv2 uses patch size ``14``; DINOv3 uses patch size ``16``.
    - SAM internally resizes to ``1024x1024`` and uses SAM-specific normalization.
    """

    def __init__(self, cfg: DenseExtractorConfig, *, freeze: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_name = cfg.name

        if cfg.name == BackboneName.DINOV2_VIT_B14:
            self.encoder = build_dinov2_vit_b14(
                pretrained=cfg.dinov2_weights_path is None,
                weights_path=cfg.dinov2_weights_path,
            )
            self._forward_impl = self._forward_dinov2
        elif cfg.name == BackboneName.DINOV3_VIT_B16:
            self.encoder = build_dinov3_vit_b16(
                pretrained=cfg.dinov3_weights_path is None,
                weights_path=cfg.dinov3_weights_path,
            )
            self._forward_impl = self._forward_dinov3
        elif cfg.name == BackboneName.SAM_VIT_B:
            self.encoder = build_sam_vit_b_image_encoder(checkpoint_path=cfg.sam_checkpoint_path)
            self._forward_impl = self._forward_sam
        else:
            raise ValueError(f"Unsupported backbone: {cfg.name}")

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @property
    def patch_size(self) -> int:
        """Return a representative patch size for the selected backbone."""
        if self.backbone_name == BackboneName.DINOV2_VIT_B14:
            return 14
        if self.backbone_name == BackboneName.DINOV3_VIT_B16:
            return 16
        return 16

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
            Feature map and metadata (backbone-specific).
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
        meta["patch_size"] = 14
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
        meta["patch_size"] = 16
        meta["coord_hw"] = (int(x_imagenet.shape[2]), int(x_imagenet.shape[3]))
        meta["dataset_hw"] = meta["coord_hw"]
        return feats, meta

    def _forward_sam(self, x_imagenet: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x_sam = imagenet_to_sam_input(x_imagenet, target_size=1024)
        feats, meta = extract_dense_grid_sam(self.encoder, x_sam)
        meta["patch_size"] = 16
        meta["sam_input_size"] = 1024
        meta["coord_hw"] = (1024, 1024)
        meta["dataset_hw"] = (int(x_imagenet.shape[2]), int(x_imagenet.shape[3]))
        return feats, meta


__all__ = ["BackboneName", "DenseExtractorConfig", "DenseFeatureExtractor"]
