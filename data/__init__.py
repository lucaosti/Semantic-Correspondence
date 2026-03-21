"""
Data loading package for semantic correspondence benchmarks.

The main entry point for SPair-71k is :class:`data.dataset.SPair71kPairDataset`.
"""

from .paths import resolve_spair_root
from .dataset import (
    MAX_KEYPOINTS,
    PreprocessMode,
    SPair71kPairDataset,
    SPairPaths,
    SplitSpec,
    build_imagenet_normalize,
    build_photometric_pair_transform,
    default_spair_root,
    parse_spair_pair_line,
    preprocess_pair_images_and_keypoints,
    round_side_to_patch_multiple,
    spair_collate_fn,
    spair_split_filename,
    visualize_correspondences,
    visualize_pair,
)

__all__ = [
    "resolve_spair_root",
    "MAX_KEYPOINTS",
    "PreprocessMode",
    "SPair71kPairDataset",
    "SPairPaths",
    "SplitSpec",
    "build_imagenet_normalize",
    "build_photometric_pair_transform",
    "default_spair_root",
    "parse_spair_pair_line",
    "preprocess_pair_images_and_keypoints",
    "round_side_to_patch_multiple",
    "spair_collate_fn",
    "spair_split_filename",
    "visualize_correspondences",
    "visualize_pair",
]
