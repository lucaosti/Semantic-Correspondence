"""
Selective freezing / unfreezing of ViT backbones (Task 2: last layers only).
"""

from __future__ import annotations

from typing import Iterable

import torch.nn as nn


def set_requires_grad(module: nn.Module, trainable: bool) -> None:
    """Set ``requires_grad`` for all parameters in ``module``."""
    for p in module.parameters():
        p.requires_grad = trainable


def freeze_all(module: nn.Module) -> None:
    """Freeze every parameter under ``module``."""
    set_requires_grad(module, False)


def unfreeze_last_transformer_blocks(
    model: nn.Module,
    *,
    n_blocks: int,
    blocks_attr: str = "blocks",
) -> None:
    """
    Freeze the full model, then unfreeze the last ``n_blocks`` in ``model.<blocks_attr>``.

    This matches DINO-style ViTs (``facebookresearch/dinov2``, ``dinov3``) where
    ``model.blocks`` is an ``nn.ModuleList``.

    Parameters
    ----------
    model:
        Root module (typically the ViT backbone).
    n_blocks:
        Number of final blocks to train.
    blocks_attr:
        Attribute name holding the block list (default: ``"blocks"``).
    """
    freeze_all(model)
    if not hasattr(model, blocks_attr):
        raise AttributeError(f"Model has no attribute {blocks_attr!r}.")
    blocks: nn.Module = getattr(model, blocks_attr)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError(f"Expected ModuleList at {blocks_attr}, got {type(blocks)}")
    if n_blocks <= 0:
        return
    chosen = list(blocks)[-n_blocks:]
    for b in chosen:
        set_requires_grad(b, True)


def unfreeze_parameters(params: Iterable[nn.Parameter], trainable: bool = True) -> None:
    """Set ``requires_grad`` for an explicit parameter list (e.g., LoRA tensors)."""
    for p in params:
        p.requires_grad = trainable


def collect_trainable_parameter_groups(
    model: nn.Module,
    *,
    base_lr: float,
    backbone_lr_multiplier: float = 1.0,
) -> list[dict]:
    """
    Build a single optimizer parameter group from all trainable parameters.

    Parameters
    ----------
    model:
        Any ``nn.Module``.
    base_lr:
        Learning rate for parameters with ``requires_grad=True``.
    backbone_lr_multiplier:
        Multiplier applied to ``base_lr`` (extend to multiple groups if separate
        backbone / head LRs are needed).

    Returns
    -------
    list[dict]
        A list with a single AdamW-style group for all trainable parameters.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters found.")
    return [{"params": trainable, "lr": base_lr * backbone_lr_multiplier}]


__all__ = [
    "collect_trainable_parameter_groups",
    "freeze_all",
    "set_requires_grad",
    "unfreeze_last_transformer_blocks",
    "unfreeze_parameters",
]
