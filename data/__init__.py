"""
Data loading package for semantic correspondence benchmarks.

Supported datasets
------------------
* **SPair-71k** — :class:`data.dataset.SPair71kPairDataset`
* **PF-Willow** — :class:`data.pf_dataset.PFWillowPairDataset`
* **PF-Pascal** — :class:`data.pf_dataset.PFPascalPairDataset`

All three datasets share the same ``__getitem__`` schema and can be used
interchangeably with :func:`data.dataset.spair_collate_fn` and the evaluation
runner in :mod:`evaluation.experiment_runner`.
"""

from .paths import resolve_pf_pascal_root, resolve_pf_willow_root, resolve_spair_root
from .dataset import (
    INVALID_KP_COORD,
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
    spair_collate_fn,
    spair_split_filename,
)
from .pf_dataset import (
    PF_PASCAL_CATEGORIES,
    PF_WILLOW_CATEGORIES,
    PFPascalPairDataset,
    PFWillowPairDataset,
)

__all__ = [
    # Path resolvers
    "resolve_spair_root",
    "resolve_pf_willow_root",
    "resolve_pf_pascal_root",
    # SPair-71k
    "INVALID_KP_COORD",
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
    "spair_collate_fn",
    "spair_split_filename",
    # PF datasets
    "PF_PASCAL_CATEGORIES",
    "PF_WILLOW_CATEGORIES",
    "PFPascalPairDataset",
    "PFWillowPairDataset",
]
