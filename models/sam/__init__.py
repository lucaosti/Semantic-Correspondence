"""
SAM (Segment Anything) backbone package — ViT-B image encoder only.

See :mod:`models.sam.backbone` for construction and dense feature extraction.
"""

from models.sam.backbone import build_sam_vit_b_image_encoder, extract_dense_grid_sam

__all__ = ["build_sam_vit_b_image_encoder", "extract_dense_grid_sam"]
