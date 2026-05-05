"""Tests for ViT block freezing / unfreezing."""

from __future__ import annotations

import torch

from models.dinov2.backbone import build_dinov2_vit_b14
from training.unfreeze import freeze_all, unfreeze_last_transformer_blocks


def _n_trainable(m: torch.nn.Module) -> int:
    return sum(1 for p in m.parameters() if p.requires_grad)


def test_unfreeze_last_block_only():
    m = build_dinov2_vit_b14(pretrained=False)
    freeze_all(m)
    assert _n_trainable(m) == 0
    unfreeze_last_transformer_blocks(m, n_blocks=1)
    assert _n_trainable(m) > 0
    last_block = m.blocks[-1]
    assert all(p.requires_grad for p in last_block.parameters())
    if len(m.blocks) > 1:
        assert not any(p.requires_grad for p in m.blocks[0].parameters())
