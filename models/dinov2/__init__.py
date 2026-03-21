"""
DINOv2 backbone package (implementation under ``models/dinov2/``, see ``PAPERS.md``).

ViT-B/14 construction and dense features: :mod:`models.dinov2.backbone`.
"""

from models.dinov2.backbone import build_dinov2_vit_b14, extract_dense_grid
from models.common.vit_intermediate import fuse_layer_features

__all__ = ["build_dinov2_vit_b14", "extract_dense_grid", "fuse_layer_features"]
