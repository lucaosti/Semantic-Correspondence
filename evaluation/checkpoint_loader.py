"""
Load training checkpoints into :class:`models.common.dense_extractor.DenseFeatureExtractor`.

Shared by CLI scripts and notebooks so evaluation behavior stays consistent.
"""

from __future__ import annotations

from typing import Dict, Union

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

    If the checkpoint includes a ``\"lora\"`` configuration block, LoRA adapters are
    injected into the encoder before loading the state dict so that ``lora_a`` /
    ``lora_b`` parameters are recognised rather than silently dropped by
    ``strict=False``.

    Parameters
    ----------
    extractor:
        Dense feature wrapper whose ``encoder`` will receive weights.
    checkpoint_path:
        Path to ``.pt`` / ``.pth`` from ``scripts/train.py`` (finetune or LoRA mode).
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

    # If the checkpoint carries LoRA metadata, inject adapters before loading so that
    # lora_a / lora_b keys in the state dict are matched by load_state_dict rather
    # than silently skipped as "unexpected" keys.
    lora_cfg = ckpt.get("lora") if isinstance(ckpt, dict) else None
    if lora_cfg and isinstance(lora_cfg, dict):
        rank = int(lora_cfg.get("rank", 8))
        alpha = float(lora_cfg.get("alpha", 16.0))
        last_blocks = int(lora_cfg.get("last_blocks", 1))
        backbone_name = (
            extractor.backbone_name.value
            if hasattr(extractor, "backbone_name")
            else ""
        )
        if backbone_name.startswith("sam"):
            from models.common.lora import apply_lora_to_last_blocks_mlp_sam
            apply_lora_to_last_blocks_mlp_sam(
                extractor.encoder,
                last_n_blocks=last_blocks,
                rank=rank,
                alpha=alpha,
            )
        else:
            from models.common.lora import apply_lora_to_last_blocks_mlp
            apply_lora_to_last_blocks_mlp(
                extractor.encoder,
                last_n_blocks=last_blocks,
                rank=rank,
                alpha=alpha,
            )

    missing, unexpected = extractor.encoder.load_state_dict(state, strict=False)
    extractor.eval()
    return {"missing": len(missing), "unexpected": len(unexpected)}


__all__ = ["load_encoder_weights_from_pt"]
