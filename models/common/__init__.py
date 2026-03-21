"""
Shared model utilities (not tied to a single backbone).

Contains input normalization, matching, coordinate helpers, ViT fusion helpers, the
multi-backbone :class:`DenseFeatureExtractor`, and inference-only refinement (window soft-argmax).
"""

from models.common.coord_utils import rescale_keypoints_xy
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from models.common.input_norm import denormalize_imagenet, imagenet_to_sam_input, normalize_imagenet
from models.common.matching import (
    argmax_to_pixel_xy,
    match_cosine_similarity_map,
    predict_correspondences_cosine_argmax,
    sample_features_bilinear,
)
from models.common.vit_intermediate import extract_intermediate_dense_grid, fuse_layer_features
from models.common.lora import LoRALinear, apply_lora_to_last_blocks_mlp, lora_trainable_parameters
from models.common.window_soft_argmax import refine_predictions_window_soft_argmax, window_soft_argmax_xy

__all__ = [
    "BackboneName",
    "DenseExtractorConfig",
    "DenseFeatureExtractor",
    "LoRALinear",
    "apply_lora_to_last_blocks_mlp",
    "argmax_to_pixel_xy",
    "denormalize_imagenet",
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
