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


def test_config_matches_train_argparse_defaults():
    """Verify training/config.py defaults stay in sync with train.py argparse defaults."""
    import sys
    from scripts.train import parse_args

    old_argv = sys.argv
    sys.argv = ["train.py", "--mode", "finetune"]
    args = parse_args()
    sys.argv = old_argv

    ft = FinetuneConfig()
    es = EarlyStoppingConfig()
    lora = LoRAConfig()

    assert args.batch_size == ft.batch_size
    assert args.epochs == ft.max_epochs
    assert args.last_blocks == ft.last_blocks
    assert args.weight_decay == ft.weight_decay
    assert args.layer_indices == ft.dino_layer_indices
    assert args.patience == es.patience
    assert abs(args.min_delta - es.min_delta) < 1e-9
    assert args.rank == lora.rank
    assert abs(args.alpha - lora.alpha) < 1e-9
