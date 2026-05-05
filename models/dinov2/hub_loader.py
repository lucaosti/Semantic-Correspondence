# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Integrated into this project under ``models/dinov2/`` (see ``PAPERS.md``).
# Adapted from ``facebookresearch/dinov2`` hub entry points for ViT backbones.

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import torch

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


class Weights(Enum):
    LVD142M = "LVD142M"
    XRAY_DINO = "XRay-DINO"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    hash: Optional[str] = None,
    check_hash: bool = False,
    **kwargs,
):
    from models.dinov2 import vision_transformer as vits

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        if type(weights) is Weights and weights not in {
            Weights.LVD142M,
            Weights.XRAY_DINO,
        }:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
            url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        model.load_state_dict(state_dict, strict=True)

    return model


def dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """DINOv2 ViT-B/14 (LVD-142M or local / URL weights)."""
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


__all__ = ["Weights", "dinov2_vitb14", "convert_path_or_url_to_url"]
