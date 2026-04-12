"""
Input-space conversions between dataset tensors and official backbone expectations.

The SPair dataset pipeline uses ImageNet mean/std normalization for ViT-style training.
SAM's official image encoder expects a different pixel normalization and a fixed
``1024 x 1024`` input resolution.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

# ImageNet statistics (torchvision convention).
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Official SAM preprocessing (RGB, 0-255 scale before mean subtraction).
SAM_PIXEL_MEAN: Tuple[float, float, float] = (123.675, 116.28, 103.53)
SAM_PIXEL_STD: Tuple[float, float, float] = (58.395, 57.12, 57.375)


def denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    Invert ImageNet normalization, returning approximately ``[0, 1]`` RGB.

    Parameters
    ----------
    x:
        Tensor shaped ``(B, 3, H, W)`` or ``(3, H, W)``.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``, roughly in ``[0, 1]``.
    """
    mean = x.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = x.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return x * std + mean


def normalize_imagenet(x01: torch.Tensor) -> torch.Tensor:
    """
    Apply ImageNet normalization to an ``[0, 1]`` RGB tensor.

    Parameters
    ----------
    x01:
        Tensor shaped ``(B, 3, H, W)`` with values typically in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        Normalized tensor.
    """
    mean = x01.new_tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = x01.new_tensor(IMAGENET_STD).view(-1, 1, 1)
    return (x01 - mean) / std


def imagenet_to_sam_input(
    x_imagenet: torch.Tensor,
    target_size: int = 1024,
    antialias: bool = True,
) -> torch.Tensor:
    """
    Convert an ImageNet-normalized batch to SAM-style normalized ``1024x1024`` tensors.

    This denormalizes to ``[0, 1]``, rescales to ``[0, 255]``, resizes to ``target_size``,
    then applies SAM mean/std normalization (same as the official SAM image encoder in ``models/sam/``).

    Parameters
    ----------
    x_imagenet:
        Tensor shaped ``(B, 3, H, W)`` in ImageNet-normalized space.
    target_size:
        SAM ViT-B defaults to ``1024`` training resolution.
    antialias:
        Passed to ``torch.nn.functional.interpolate`` when downsampling.

    Returns
    -------
    torch.Tensor
        Tensor shaped ``(B, 3, target_size, target_size)`` in SAM input space.
    """
    if x_imagenet.dim() != 4:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(x_imagenet.shape)}")
    x01 = denormalize_imagenet(x_imagenet).clamp(0.0, 1.0)
    x255 = x01 * 255.0
    x255 = F.interpolate(
        x255,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
        antialias=antialias,
    )
    mean = x255.new_tensor(SAM_PIXEL_MEAN).view(1, 3, 1, 1)
    std = x255.new_tensor(SAM_PIXEL_STD).view(1, 3, 1, 1)
    return (x255 - mean) / std
