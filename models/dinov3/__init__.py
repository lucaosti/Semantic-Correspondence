"""
DINOv3 backbone package (implementation under ``models/dinov3/``, see ``PAPERS.md``).

ViT-B/16 and dense features: :mod:`models.dinov3.backbone`.
"""

from models.dinov3.backbone import build_dinov3_vit_b16, extract_dense_grid_dinov3

__all__ = ["build_dinov3_vit_b16", "extract_dense_grid_dinov3"]
