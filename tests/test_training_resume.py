"""Tests for resume + compile helpers in scripts._training_common."""

from __future__ import annotations


import torch
import torch.nn as nn

from scripts import _training_common as tc
from training.early_stopping import EarlyStopping


def test_save_resume_atomic_writes_file_and_no_tmp(tmp_path):
    p = tmp_path / "resume.pt"
    payload = {"model": {"w": torch.zeros(2)}, "epoch": 3, "best_val": 0.42}
    tc.save_resume_atomic(str(p), payload)
    assert p.is_file()
    assert not (tmp_path / "resume.pt.tmp").exists()
    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert blob["epoch"] == 3
    assert blob["best_val"] == 0.42


def test_stopper_dict_roundtrip():
    s = EarlyStopping(patience=4, min_delta=0.01, mode="min")
    s.best_value = 0.123
    s.num_bad_epochs = 2
    s.best_epoch = 5
    blob = tc._stopper_to_dict(s)
    assert blob["patience"] == 4
    assert blob["min_delta"] == 0.01
    assert blob["mode"] == "min"
    assert blob["best_value"] == 0.123
    assert blob["num_bad_epochs"] == 2
    assert blob["best_epoch"] == 5

    s2 = EarlyStopping(patience=1, min_delta=0.0, mode="max")
    tc._apply_stopper_from_blob(s2, blob)
    assert s2.patience == 4
    assert s2.min_delta == 0.01
    assert s2.mode == "min"
    assert s2.best_value == 0.123
    assert s2.num_bad_epochs == 2
    assert s2.best_epoch == 5


def test_load_training_resume_no_arg_returns_defaults():
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stopper = EarlyStopping(patience=2)
    full, batch, best = tc.load_training_resume(
        None, model=model, optimizer=opt, device=torch.device("cpu"),
        stopper=stopper, grad_scaler=None, script_tag="test", epochs=10,
    )
    assert full == 0
    assert batch == 0
    assert best == float("inf")


def test_load_training_resume_missing_file_returns_defaults(tmp_path):
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stopper = EarlyStopping(patience=2)
    full, batch, best = tc.load_training_resume(
        str(tmp_path / "nope.pt"), model=model, optimizer=opt,
        device=torch.device("cpu"), stopper=stopper, grad_scaler=None,
        script_tag="test", epochs=10,
    )
    assert full == 0
    assert batch == 0
    assert best == float("inf")


def test_load_training_resume_training_complete_skips_loop(tmp_path):
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stopper = EarlyStopping(patience=2)
    p = tmp_path / "done.pt"
    torch.save({"training_complete": True, "best_val": 0.05}, p)
    full, batch, best = tc.load_training_resume(
        str(p), model=model, optimizer=opt, device=torch.device("cpu"),
        stopper=stopper, grad_scaler=None, script_tag="test", epochs=42,
    )
    assert full == 42  # epochs is returned to indicate "skip the loop"
    assert batch == 0
    assert best == 0.05


def test_load_training_resume_mid_epoch_payload(tmp_path):
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stopper = EarlyStopping(patience=2)
    p = tmp_path / "mid.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": 3,
        "full_epochs_done": 3,
        "batch_in_epoch": 17,
        "best_val": 0.7,
        "stopper": tc._stopper_to_dict(stopper),
    }, p)
    full, batch, best = tc.load_training_resume(
        str(p), model=model, optimizer=opt, device=torch.device("cpu"),
        stopper=stopper, grad_scaler=None, script_tag="test", epochs=50,
    )
    assert full == 3
    assert batch == 17
    assert best == 0.7


def test_load_training_resume_legacy_epoch_only(tmp_path):
    """Old checkpoints without full_epochs_done: derive from epoch + 1."""
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stopper = EarlyStopping(patience=2)
    p = tmp_path / "old.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": 5,
        "best_val": 0.3,
        "stopper": tc._stopper_to_dict(stopper),
    }, p)
    full, batch, best = tc.load_training_resume(
        str(p), model=model, optimizer=opt, device=torch.device("cpu"),
        stopper=stopper, grad_scaler=None, script_tag="test", epochs=50,
    )
    assert full == 6
    assert batch == 0
    assert best == 0.3


def test_maybe_compile_model_noop_when_not_requested():
    m = nn.Linear(4, 4)
    out = tc.maybe_compile_model(m, torch.device("cpu"), requested=False, script_tag="t")
    assert out is m


def test_maybe_compile_model_noop_on_cpu():
    m = nn.Linear(4, 4)
    out = tc.maybe_compile_model(m, torch.device("cpu"), requested=True, script_tag="t")
    assert out is m


def test_maybe_compile_model_noop_on_mps():
    m = nn.Linear(4, 4)
    out = tc.maybe_compile_model(m, torch.device("mps"), requested=True, script_tag="t")
    assert out is m


def test_save_resume_atomic_overwrites_existing(tmp_path):
    p = tmp_path / "r.pt"
    tc.save_resume_atomic(str(p), {"v": 1})
    tc.save_resume_atomic(str(p), {"v": 2})
    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert blob["v"] == 2
