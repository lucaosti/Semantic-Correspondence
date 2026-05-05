"""Tests for runtime config helpers in scripts.run_pipeline."""

from __future__ import annotations

import pytest

import scripts.run_pipeline as rp


def test_normalize_precision_token():
    assert rp._normalize_precision_token("auto") == "auto"
    assert rp._normalize_precision_token("BF16") == "bf16"
    assert rp._normalize_precision_token(" fp16 ") == "fp16"
    with pytest.raises(ValueError):
        rp._normalize_precision_token("mixed")


def test_parse_backbone_int_map_success():
    out = rp._parse_backbone_int_map(
        "finetune.batch_size_by_backbone",
        {"dinov2_vitb14": 32, "sam_vit_b": 4},
    )
    assert out == {"dinov2_vitb14": 32, "sam_vit_b": 4}


def test_parse_backbone_int_map_rejects_invalid_values():
    with pytest.raises(ValueError):
        rp._parse_backbone_int_map("x", {"unknown_backbone": 4})
    with pytest.raises(ValueError):
        rp._parse_backbone_int_map("x", {"dinov2_vitb14": 0})


def test_resolve_batch_size_prefers_backbone_override(monkeypatch):
    monkeypatch.setattr(rp, "FT_BATCH_SIZE", 10)
    monkeypatch.setattr(rp, "LORA_BATCH_SIZE", 12)
    monkeypatch.setattr(rp, "FT_BATCH_SIZE_BY_BACKBONE", {"dinov3_vitb16": 24})
    monkeypatch.setattr(rp, "LORA_BATCH_SIZE_BY_BACKBONE", {"sam_vit_b": 6})

    assert rp._resolve_batch_size("finetune", "dinov3_vitb16") == 24
    assert rp._resolve_batch_size("finetune", "dinov2_vitb14") == 10
    assert rp._resolve_batch_size("lora", "sam_vit_b") == 6
    assert rp._resolve_batch_size("lora", "dinov2_vitb14") == 12


def test_resolve_batch_size_unknown_stage():
    with pytest.raises(ValueError):
        rp._resolve_batch_size("invalid", "dinov2_vitb14")


def test_resolve_image_hw_uses_backbone_override(monkeypatch):
    monkeypatch.setattr(rp, "IMAGE_SIZE_BY_BACKBONE", {"dinov2_vitb14": (518, 518), "sam_vit_b": (512, 512)})
    monkeypatch.setattr(rp, "IMAGE_HEIGHT", 784)
    monkeypatch.setattr(rp, "IMAGE_WIDTH", 784)
    assert rp._resolve_image_hw("dinov2_vitb14") == (518, 518)
    assert rp._resolve_image_hw("sam_vit_b") == (512, 512)


def test_resolve_image_hw_falls_back_to_global(monkeypatch):
    monkeypatch.setattr(rp, "IMAGE_SIZE_BY_BACKBONE", {})
    monkeypatch.setattr(rp, "IMAGE_HEIGHT", 784)
    monkeypatch.setattr(rp, "IMAGE_WIDTH", 784)
    assert rp._resolve_image_hw("dinov2_vitb14") == (784, 784)
    assert rp._resolve_image_hw("unknown_backbone") == (784, 784)


def _write_yaml(tmp_path, data):
    import yaml

    p = tmp_path / "cfg.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return p


def test_apply_yaml_reads_runtime_compile(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "COMPILE", False)
    cfg = _write_yaml(tmp_path, {"runtime": {"compile": True}})
    rp._apply_pipeline_yaml(cfg)
    assert rp.COMPILE is True


def test_apply_yaml_compile_default_false(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "COMPILE", False)
    cfg = _write_yaml(tmp_path, {"runtime": {}})
    rp._apply_pipeline_yaml(cfg)
    assert rp.COMPILE is False


def test_apply_yaml_resume_save_interval(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "RESUME_SAVE_INTERVAL", 50)
    cfg = _write_yaml(tmp_path, {"runtime": {"resume_save_interval": 200}})
    rp._apply_pipeline_yaml(cfg)
    assert rp.RESUME_SAVE_INTERVAL == 200


def test_fingerprint_includes_compile(monkeypatch):
    monkeypatch.setattr(rp, "COMPILE", False)
    fp_no_compile = rp._fingerprint_payload()
    monkeypatch.setattr(rp, "COMPILE", True)
    fp_with_compile = rp._fingerprint_payload()
    assert fp_no_compile["COMPILE"] != fp_with_compile["COMPILE"]


def test_fingerprint_payload_is_serializable():
    """The payload feeds fingerprint_from_config (json.dumps); must contain only basic types."""
    import json

    payload = rp._fingerprint_payload()
    json.dumps(payload, sort_keys=True)


def test_triplet_bool_validates_length():
    assert rp._triplet_bool("X", [True, False, True]) == (True, False, True)
    assert rp._triplet_bool("X", (1, 0, 1)) == (True, False, True)
    with pytest.raises(ValueError):
        rp._triplet_bool("X", [True, False])
    with pytest.raises(ValueError):
        rp._triplet_bool("X", "not-a-list")


def test_validate_triplet_length():
    rp._validate_triplet("OK", (True, False, True))
    with pytest.raises(ValueError):
        rp._validate_triplet("BAD", (True, False))


def test_apply_yaml_image_size_by_backbone(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "IMAGE_SIZE_BY_BACKBONE", {})
    cfg = _write_yaml(tmp_path, {
        "runtime": {
            "image_size_by_backbone": {
                "dinov2_vitb14": [518, 518],
                "sam_vit_b": [512, 512],
            }
        }
    })
    rp._apply_pipeline_yaml(cfg)
    assert rp.IMAGE_SIZE_BY_BACKBONE["dinov2_vitb14"] == (518, 518)
    assert rp.IMAGE_SIZE_BY_BACKBONE["sam_vit_b"] == (512, 512)


def test_apply_yaml_image_size_rejects_unknown_backbone(tmp_path):
    cfg = _write_yaml(tmp_path, {
        "runtime": {"image_size_by_backbone": {"foo_backbone": [256, 256]}}
    })
    with pytest.raises(ValueError):
        rp._apply_pipeline_yaml(cfg)


def test_apply_yaml_image_size_rejects_bad_pair(tmp_path):
    cfg = _write_yaml(tmp_path, {
        "runtime": {"image_size_by_backbone": {"dinov2_vitb14": [256]}}
    })
    with pytest.raises(ValueError):
        rp._apply_pipeline_yaml(cfg)


def test_apply_yaml_workflow_toggles_triplets(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "TRAIN_FINETUNE", (True, True, True))
    monkeypatch.setattr(rp, "TRAIN_LORA", (True, True, True))
    monkeypatch.setattr(rp, "RUN_EVAL_BASELINE", (True, True, True))
    cfg = _write_yaml(tmp_path, {
        "workflow_toggles": {
            "train_finetune": [True, False, True],
            "train_lora": [False, False, True],
            "run_eval_baseline": [True, True, False],
            "pipeline_resume": False,
            "run_export_metrics_tables": True,
        }
    })
    rp._apply_pipeline_yaml(cfg)
    assert rp.TRAIN_FINETUNE == (True, False, True)
    assert rp.TRAIN_LORA == (False, False, True)
    assert rp.RUN_EVAL_BASELINE == (True, True, False)
    assert rp.PIPELINE_RESUME is False
    assert rp.RUN_EXPORT_METRICS_TABLES is True


def test_apply_yaml_workflow_toggle_bad_triplet_raises(tmp_path):
    cfg = _write_yaml(tmp_path, {
        "workflow_toggles": {"train_finetune": [True, False]}
    })
    with pytest.raises(ValueError):
        rp._apply_pipeline_yaml(cfg)


def test_apply_yaml_paths_section(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "SPAIR_ROOT", None)
    monkeypatch.setattr(rp, "CHECKPOINT_DIR", "checkpoints")
    cfg = _write_yaml(tmp_path, {
        "paths": {"spair_root": "/data/SPair-71k", "checkpoint_dir": "/ckpts"}
    })
    rp._apply_pipeline_yaml(cfg)
    assert rp.SPAIR_ROOT == "/data/SPair-71k"
    assert rp.CHECKPOINT_DIR == "/ckpts"


def test_apply_yaml_runtime_num_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "NUM_WORKERS", None)
    rp._apply_pipeline_yaml(_write_yaml(tmp_path, {"runtime": {"num_workers": 4}}))
    assert rp.NUM_WORKERS == 4
    rp._apply_pipeline_yaml(_write_yaml(tmp_path, {"runtime": {"num_workers": -1}}))
    assert rp.NUM_WORKERS is None


def test_apply_yaml_finetune_block(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "FT_LR", 5e-5)
    monkeypatch.setattr(rp, "FT_WEIGHT_DECAY", 0.01)
    monkeypatch.setattr(rp, "FT_EPOCHS", 50)
    monkeypatch.setattr(rp, "FT_PATIENCE", 7)
    monkeypatch.setattr(rp, "LAST_BLOCKS_LIST", [1, 2, 4])
    cfg = _write_yaml(tmp_path, {
        "finetune": {
            "lr": 1e-4, "weight_decay": 0.05, "epochs": 30, "patience": 5,
            "last_blocks": 2,
        }
    })
    rp._apply_pipeline_yaml(cfg)
    assert rp.FT_LR == 1e-4
    assert rp.FT_WEIGHT_DECAY == 0.05
    assert rp.FT_EPOCHS == 30
    assert rp.FT_PATIENCE == 5
    assert rp.LAST_BLOCKS_LIST == [2]


def test_apply_yaml_lora_block(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "LORA_LR", 1e-3)
    monkeypatch.setattr(rp, "LORA_RANK", 8)
    monkeypatch.setattr(rp, "LORA_ALPHA", 16.0)
    monkeypatch.setattr(rp, "LORA_LAST_BLOCKS", 2)
    cfg = _write_yaml(tmp_path, {
        "lora": {"lr": 5e-4, "rank": 16, "alpha": 32.0, "last_blocks": 4}
    })
    rp._apply_pipeline_yaml(cfg)
    assert rp.LORA_LR == 5e-4
    assert rp.LORA_RANK == 16
    assert rp.LORA_ALPHA == 32.0
    assert rp.LORA_LAST_BLOCKS == 4


def test_apply_yaml_invalid_precision_raises(tmp_path):
    cfg = _write_yaml(tmp_path, {"runtime": {"precision": "double"}})
    with pytest.raises(ValueError):
        rp._apply_pipeline_yaml(cfg)


def test_apply_yaml_eval_alphas_and_wsa(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "EVAL_ALPHAS", (0.05, 0.1, 0.2))
    monkeypatch.setattr(rp, "WSA_WINDOW", 5)
    monkeypatch.setattr(rp, "WSA_TEMPERATURE", 1.0)
    cfg = _write_yaml(tmp_path, {
        "runtime": {"alphas": [0.1, 0.2], "wsa_window": 7, "wsa_temperature": 0.5}
    })
    rp._apply_pipeline_yaml(cfg)
    assert rp.EVAL_ALPHAS == (0.1, 0.2)
    assert rp.WSA_WINDOW == 7
    assert rp.WSA_TEMPERATURE == 0.5
