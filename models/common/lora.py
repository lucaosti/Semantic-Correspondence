"""
Manual LoRA adapters for linear layers (Task 4, parameter-efficient fine-tuning).

This implementation avoids relying on third-party PEFT libraries so the project stays
easy to audit; only standard ``torch.nn`` is required.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Low-rank adaptation wrapped around a frozen :class:`torch.nn.Linear`.

    Forward::

        y = F.linear(x, W, b) + (alpha / r) * (x @ A^T @ B^T)

    where ``A ∈ R^{r×in}`` and ``B ∈ R^{out×r}``.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)

        self.weight = linear.weight
        self.bias = linear.bias

        for p in (self.weight, self.bias):
            if p is not None:
                p.requires_grad = False

        self.lora_a = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_b = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_a.T) @ self.lora_b.T
        return base + self.scale * lora


def _iter_dino_blocks(model: nn.Module) -> Sequence[nn.Module]:
    if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
        return list(model.blocks)
    raise TypeError("Expected a DINO-like ViT with a `blocks` ModuleList.")


def apply_lora_to_last_blocks_mlp(
    model: nn.Module,
    *,
    last_n_blocks: int,
    rank: int,
    alpha: float,
) -> List[nn.Parameter]:
    """
    Replace MLP linear layers in the last ``last_n_blocks`` transformer blocks.

    This targets DINO-style blocks exposing ``block.mlp.fc1`` and ``block.mlp.fc2``.

    Parameters
    ----------
    model:
        A DINOv2/DINOv3 ViT module.
    last_n_blocks:
        How many terminal blocks to modify.
    rank, alpha:
        LoRA hyperparameters.

    Returns
    -------
    list[torch.nn.Parameter]
        Trainable LoRA parameters (for optimizers).
    """
    blocks = _iter_dino_blocks(model)
    if last_n_blocks <= 0:
        return []
    chosen = blocks[-last_n_blocks:]
    trainable: List[nn.Parameter] = []

    for blk in chosen:
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            continue
        for name in ("fc1", "fc2"):
            lin = getattr(mlp, name, None)
            if not isinstance(lin, nn.Linear):
                continue
            lora = LoRALinear(lin, rank=rank, alpha=alpha)
            setattr(mlp, name, lora)
            trainable.extend([lora.lora_a, lora.lora_b])

    return trainable


def apply_lora_to_last_blocks_mlp_sam(
    model: nn.Module,
    *,
    last_n_blocks: int,
    rank: int,
    alpha: float,
) -> List[nn.Parameter]:
    """
    Replace SAM ``ImageEncoderViT`` MLP linears (``lin1`` / ``lin2``) in the last blocks.

    SAM uses ``block.mlp.lin1`` and ``block.mlp.lin2`` instead of DINO's ``fc1`` / ``fc2``.
    """
    blocks = _iter_dino_blocks(model)
    if last_n_blocks <= 0:
        return []
    chosen = blocks[-last_n_blocks:]
    trainable: List[nn.Parameter] = []

    for blk in chosen:
        mlp = getattr(blk, "mlp", None)
        if mlp is None:
            continue
        for name in ("lin1", "lin2"):
            lin = getattr(mlp, name, None)
            if not isinstance(lin, nn.Linear):
                continue
            lora = LoRALinear(lin, rank=rank, alpha=alpha)
            setattr(mlp, name, lora)
            trainable.extend([lora.lora_a, lora.lora_b])

    return trainable


def lora_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return parameters whose name suggests LoRA tensors (``lora_a`` / ``lora_b``)."""
    out: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if "lora_a" in name or "lora_b" in name:
            if p.requires_grad:
                out.append(p)
    return out


__all__ = [
    "LoRALinear",
    "apply_lora_to_last_blocks_mlp",
    "apply_lora_to_last_blocks_mlp_sam",
    "lora_trainable_parameters",
]
