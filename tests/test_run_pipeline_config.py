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
