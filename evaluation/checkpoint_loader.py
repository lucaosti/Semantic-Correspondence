"""
Load training checkpoints into :class:`models.common.dense_extractor.DenseFeatureExtractor`.

Shared by CLI scripts and notebooks so evaluation behavior stays consistent.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch

from models.common.dense_extractor import DenseFeatureExtractor


def load_encoder_weights_from_pt(
    extractor: DenseFeatureExtractor,
    checkpoint_path: str,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, int]:
    """
    Load a ``torch.save`` file that contains a ``\"model\"`` state dict (or a raw state dict).

    Parameters
    ----------
    extractor:
        Dense feature wrapper whose ``encoder`` will receive weights.
    checkpoint_path:
        Path to ``.pt`` / ``.pth`` from ``train_finetune.py`` or ``train_lora.py``.
    map_location:
        Forwarded to :func:`torch.load`.

    Returns
    -------
    dict[str, int]
        Counts of ``missing`` and ``unexpected`` keys from ``load_state_dict(..., strict=False)``.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = extractor.encoder.load_state_dict(state, strict=False)
    extractor.eval()
    return {"missing": len(missing), "unexpected": len(unexpected)}


__all__ = ["load_encoder_weights_from_pt"]
