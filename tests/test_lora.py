"""Tests for manual LoRA linear adapters."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common.lora import LoRALinear, apply_lora_to_last_blocks_mlp, lora_trainable_parameters
from models.dinov2.backbone import build_dinov2_vit_b14


def test_lora_linear_matches_base_plus_scaled_delta():
    lin = nn.Linear(4, 3, bias=True)
    lora = LoRALinear(lin, rank=2, alpha=4.0)
    x = torch.randn(5, 4)
    expected_base = F.linear(x, lin.weight, lin.bias)
    delta = (x @ lora.lora_a.T) @ lora.lora_b.T
    scale = lora.alpha / float(lora.rank)
    out = lora(x)
    assert torch.allclose(out, expected_base + scale * delta, atol=1e-5, rtol=1e-4)


def test_apply_lora_returns_trainable_params():
    m = build_dinov2_vit_b14(pretrained=False)
    params = apply_lora_to_last_blocks_mlp(m, last_n_blocks=1, rank=4, alpha=8.0)
    assert len(params) >= 2
    assert all(p.requires_grad for p in params)


def test_lora_trainable_parameters_non_empty_after_apply():
    m = build_dinov2_vit_b14(pretrained=False)
    apply_lora_to_last_blocks_mlp(m, last_n_blocks=1, rank=2, alpha=4.0)
    params = lora_trainable_parameters(m)
    assert len(params) > 0
