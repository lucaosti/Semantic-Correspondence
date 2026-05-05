"""Tests for pipeline resume fingerprinting (no PyTorch)."""

from __future__ import annotations

from pathlib import Path

from utils.pipeline_state import (
    fingerprint_from_config,
    is_step_done,
    load_state,
    save_state,
    should_reset_from_env,
    state_path,
)


def test_fingerprint_stable_for_same_config():
    cfg = {"a": 1, "b": [True, False]}
    assert fingerprint_from_config(cfg) == fingerprint_from_config(cfg)


def test_fingerprint_changes_when_config_changes():
    h1 = fingerprint_from_config({"epochs": 50})
    h2 = fingerprint_from_config({"epochs": 51})
    assert h1 != h2


def test_save_load_roundtrip(tmp_path: Path):
    repo = tmp_path / "repo"
    st = {"fingerprint": "abc", "completed": ["verify_dataset"]}
    save_state(repo, st)
    path = state_path(repo)
    assert path.is_file()
    loaded = load_state(repo)
    assert loaded is not None
    assert loaded["fingerprint"] == "abc"
    assert loaded["completed"] == ["verify_dataset"]


def test_is_step_done():
    assert is_step_done(["a", "b"], "a") is True
    assert is_step_done(["a", "b"], "c") is False


def test_should_reset_from_env(monkeypatch):
    monkeypatch.delenv("SEMANTIC_CORRESPONDENCE_PIPELINE_RESET", raising=False)
    assert should_reset_from_env() is False
    monkeypatch.setenv("SEMANTIC_CORRESPONDENCE_PIPELINE_RESET", "1")
    assert should_reset_from_env() is True
