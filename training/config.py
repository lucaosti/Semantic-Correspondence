"""
Training configuration dataclasses (fine-tuning and PEFT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FinetuneConfig:
    """
    Hyperparameters for supervised fine-tuning of the last ViT blocks (Task 2).

    Attributes
    ----------
    backbone:
        One of ``dinov2_vitb14``, ``dinov3_vitb16``, ``sam_vit_b`` (must match loaders).
    last_blocks:
        Number of final transformer blocks to keep trainable.
    learning_rate:
        Base learning rate for trainable parameters.
    weight_decay:
        AdamW weight decay.
    max_epochs:
        Upper bound on epochs (use early stopping in practice).
    batch_size:
        Pairs per optimizer step (loss is averaged over the batch). Default matches training CLIs.
    num_workers:
        DataLoader workers (CLI scripts use ``-1`` for host-aware defaults in ``utils.hardware``).
    """

    backbone: str = "dinov2_vitb14"
    last_blocks: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 50
    batch_size: int = 100
    num_workers: int = 4


@dataclass(frozen=True)
class LoRAConfig:
    """
    LoRA hyperparameters (Task 4).

    Attributes
    ----------
    rank:
        LoRA rank ``r``.
    alpha:
        LoRA scaling ``alpha`` (scaling factor is typically ``alpha / rank``).
    target:
        Which modules to adapt (implementation-specific string), e.g. ``mlp`` or ``attn+mlp``.
    max_epochs:
        Short schedules (often ~2 epochs on SPair-71k) are common for PEFT.
    batch_size:
        Pairs per optimizer step (loss is averaged over the batch). Default matches training CLIs.
    num_workers:
        DataLoader workers (CLI scripts use ``-1`` for host-aware defaults in ``utils.hardware``).
    """

    rank: int = 8
    alpha: float = 16.0
    target: str = "mlp"
    max_epochs: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 100
    num_workers: int = 4


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping on validation PCK or loss."""

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"  # "max" for PCK, "min" for loss


@dataclass(frozen=True)
class TrainPaths:
    """Dataset and checkpoint locations."""

    spair_root: str
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "default"
