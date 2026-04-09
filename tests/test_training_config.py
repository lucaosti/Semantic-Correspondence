"""Tests for training config dataclasses."""

from __future__ import annotations

from training.config import EarlyStoppingConfig, FinetuneConfig, LoRAConfig, TrainPaths


def test_finetune_config_defaults():
    c = FinetuneConfig()
    assert c.backbone == "dinov2_vitb14"
    assert c.batch_size == 20
    assert c.last_blocks == 2
    assert c.dino_layer_indices == 4
    assert c.precision == "auto"


def test_lora_config_defaults():
    c = LoRAConfig()
    assert c.rank == 8
    assert c.alpha == 16.0
    assert c.target == "mlp"
    assert c.dino_layer_indices == 4
    assert c.precision == "auto"


def test_early_stopping_config():
    c = EarlyStoppingConfig(patience=7, mode="min")
    assert c.patience == 7
    assert c.mode == "min"


def test_train_paths():
    p = TrainPaths(spair_root="/data/SPair-71k")
    assert p.spair_root == "/data/SPair-71k"
