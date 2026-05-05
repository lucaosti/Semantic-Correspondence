"""
Model package: one subfolder per backbone (`dinov2/`, `dinov3/`, `sam/`) plus `common/`.

DINOv2 and DINOv3 live under ``models/dinov2/`` and ``models/dinov3/`` (see each folder’s ``PAPERS.md``).
SAM modeling code lives under ``models/sam/`` (see ``models/sam/PAPERS.md``).
This project does not use Hugging Face for these backbones (see ``docs/info.md``).

The stable high-level API is re-exported from :mod:`models.common`.
"""

from models.common import (
    BackboneName,
    DenseExtractorConfig,
    DenseFeatureExtractor,
    LoRALinear,
    apply_lora_to_last_blocks_mlp,
    argmax_to_pixel_xy,
    denormalize_imagenet,
    extract_intermediate_dense_grid,
    fuse_layer_features,
    imagenet_to_sam_input,
    lora_trainable_parameters,
    match_cosine_similarity_map,
    normalize_imagenet,
    predict_correspondences_cosine_argmax,
    refine_predictions_window_soft_argmax,
    rescale_keypoints_xy,
    sample_features_bilinear,
    window_soft_argmax_xy,
)
from models.dinov2.backbone import build_dinov2_vit_b14, extract_dense_grid
from models.dinov3.backbone import build_dinov3_vit_b16, extract_dense_grid_dinov3
from models.sam.backbone import build_sam_vit_b_image_encoder, extract_dense_grid_sam

__all__ = [
    "BackboneName",
    "DenseExtractorConfig",
    "DenseFeatureExtractor",
    "LoRALinear",
    "apply_lora_to_last_blocks_mlp",
    "argmax_to_pixel_xy",
    "build_dinov2_vit_b14",
    "build_dinov3_vit_b16",
    "build_sam_vit_b_image_encoder",
    "denormalize_imagenet",
    "extract_dense_grid",
    "extract_dense_grid_dinov3",
    "extract_dense_grid_sam",
    "extract_intermediate_dense_grid",
    "fuse_layer_features",
    "imagenet_to_sam_input",
    "lora_trainable_parameters",
    "match_cosine_similarity_map",
    "normalize_imagenet",
    "predict_correspondences_cosine_argmax",
    "refine_predictions_window_soft_argmax",
    "rescale_keypoints_xy",
    "sample_features_bilinear",
    "window_soft_argmax_xy",
]
