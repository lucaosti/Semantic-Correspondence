"""
Training utilities: losses, freezing last blocks, early stopping, and LoRA hooks.

Typical workflow (Task 2): load an official ViT backbone, call
:func:`training.unfreeze.unfreeze_last_transformer_blocks`, optimize with AdamW using
:func:`training.losses.gaussian_ce_loss_from_similarity_maps` or
:func:`training.engine.correspondence_gaussian_loss_dino_vit`.

Task 4 (LoRA): use :func:`models.common.lora.apply_lora_to_last_blocks_mlp` and optimize
only LoRA tensors.
"""

from training.config import EarlyStoppingConfig, FinetuneConfig, LoRAConfig, TrainPaths
from training.early_stopping import EarlyStopping
from training.engine import correspondence_gaussian_loss_dino_vit
from training.losses import (
    gaussian_ce_loss_from_similarity_maps,
    gaussian_grid_2d,
    pixel_xy_to_feat_xy,
)
from training.unfreeze import (
    collect_trainable_parameter_groups,
    freeze_all,
    set_requires_grad,
    unfreeze_last_transformer_blocks,
    unfreeze_parameters,
)

__all__ = [
    "EarlyStopping",
    "EarlyStoppingConfig",
    "FinetuneConfig",
    "LoRAConfig",
    "TrainPaths",
    "collect_trainable_parameter_groups",
    "correspondence_gaussian_loss_dino_vit",
    "freeze_all",
    "gaussian_ce_loss_from_similarity_maps",
    "gaussian_grid_2d",
    "pixel_xy_to_feat_xy",
    "set_requires_grad",
    "unfreeze_last_transformer_blocks",
    "unfreeze_parameters",
]
