"""Tests for utils.hardware: device/worker/precision adaptive helpers."""

from __future__ import annotations

import os

import torch

from utils import hardware as hw


def test_recommended_dataloader_workers_cuda(monkeypatch):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 16)
    n = hw.recommended_dataloader_workers(accelerator="cuda")
    # CUDA: floor 8, ceil min(cap, n-1) = min(64, 15)
    assert n >= 8
    assert n <= 64


def test_recommended_dataloader_workers_mps_returns_one():
    assert hw.recommended_dataloader_workers(accelerator="mps") == 1


def test_recommended_dataloader_workers_cpu(monkeypatch):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(hw.sys, "platform", "linux")
    n = hw.recommended_dataloader_workers(accelerator="cpu")
    assert 2 <= n <= 16


def test_recommended_dataloader_workers_low_cpu_returns_zero(monkeypatch):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 1)
    assert hw.recommended_dataloader_workers(accelerator="cuda") == 0


def test_recommended_dataloader_workers_windows(monkeypatch):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    n = hw.recommended_dataloader_workers(accelerator="cpu")
    assert 2 <= n <= 4


def test_recommended_dataloader_workers_darwin_returns_one(monkeypatch):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(hw.sys, "platform", "darwin")
    # darwin without explicit accelerator -> mps branch
    assert hw.recommended_dataloader_workers(accelerator="cpu") == 1


def test_recommended_device_str_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert hw.recommended_device_str() == "cuda"


def test_recommended_device_str_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _NoMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(torch.backends, "mps", _NoMps, raising=False)
    assert hw.recommended_device_str() == "cpu"


def test_resolve_device_str_passthrough():
    assert hw.resolve_device_str("cpu") == "cpu"
    assert hw.resolve_device_str("cuda") == "cuda"


def test_resolve_device_str_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    class _NoMps:
        @staticmethod
        def is_available() -> bool:
            return False

    monkeypatch.setattr(torch.backends, "mps", _NoMps, raising=False)
    assert hw.resolve_device_str(None) == "cpu"
    assert hw.resolve_device_str("") == "cpu"


def test_resolve_num_workers_explicit():
    assert hw.resolve_num_workers(0, accelerator="cuda") == 0
    assert hw.resolve_num_workers(4, accelerator="cuda") == 4


def test_resolve_num_workers_negative_uses_recommended():
    n = hw.resolve_num_workers(-1, accelerator="mps")
    assert n == 1


def test_pin_memory_for():
    assert hw.pin_memory_for(torch.device("cuda")) is True
    assert hw.pin_memory_for(torch.device("cpu")) is False
    assert hw.pin_memory_for(torch.device("mps")) is False


def test_dataloader_extra_kwargs_no_workers():
    assert hw.dataloader_extra_kwargs(0) == {}
    assert hw.dataloader_extra_kwargs(0, for_device="cuda") == {}


def test_dataloader_extra_kwargs_cuda(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.delenv("SEMANTIC_CORRESPONDENCE_PREFETCH_CAP", raising=False)
    kw = hw.dataloader_extra_kwargs(8, for_device="cuda")
    assert kw["persistent_workers"] is True
    assert kw["prefetch_factor"] >= 4


def test_dataloader_extra_kwargs_cpu_smaller_prefetch(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.delenv("SEMANTIC_CORRESPONDENCE_PREFETCH_CAP", raising=False)
    kw = hw.dataloader_extra_kwargs(4, for_device="cpu")
    assert kw["persistent_workers"] is True
    assert 2 <= kw["prefetch_factor"] <= 24


def test_dataloader_extra_kwargs_windows_no_persistent(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "win32")
    monkeypatch.delenv("SEMANTIC_CORRESPONDENCE_PREFETCH_CAP", raising=False)
    kw = hw.dataloader_extra_kwargs(4, for_device="cuda")
    assert kw["persistent_workers"] is False


def test_dataloader_extra_kwargs_respects_env_cap(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setenv("SEMANTIC_CORRESPONDENCE_PREFETCH_CAP", "5")
    kw = hw.dataloader_extra_kwargs(8, for_device="cuda")
    assert kw["prefetch_factor"] <= 8  # capped


def test_cpu_dataloader_worker_init_sets_blas_envs(monkeypatch):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.delenv(v, raising=False)
    hw.cpu_dataloader_worker_init(0)
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"


def test_loader_worker_init_for_device_cpu_with_workers():
    fn = hw.loader_worker_init_for_device("cpu", num_workers=4)
    assert callable(fn)


def test_loader_worker_init_for_device_no_workers_returns_none():
    assert hw.loader_worker_init_for_device("cpu", num_workers=0) is None
    assert hw.loader_worker_init_for_device("cuda", num_workers=4) is None


def test_maybe_tune_threads_for_cpu_device_noop_on_non_cpu():
    # Should not raise on cuda/mps and should not touch torch threads.
    hw.maybe_tune_threads_for_cpu_device("cuda", dataloader_workers=4)
    hw.maybe_tune_threads_for_cpu_device("mps", dataloader_workers=1)


def test_maybe_tune_threads_for_cpu_device_cpu_does_not_raise(monkeypatch):
    monkeypatch.setattr(hw.os, "cpu_count", lambda: 4)
    hw.maybe_tune_threads_for_cpu_device("cpu", dataloader_workers=0)
    hw.maybe_tune_threads_for_cpu_device("cpu", dataloader_workers=2)


def test_apply_accelerator_throughput_tweaks_cpu_does_not_raise():
    hw.apply_accelerator_throughput_tweaks(torch.device("cpu"))


def test_apply_accelerator_throughput_tweaks_cuda_does_not_raise(monkeypatch):
    # The function checks attributes lazily; calling with cuda device on a non-CUDA box
    # exercises the `if device.type == "cuda"` branch but skips the actual cuda assignments
    # because torch.backends.cuda may still be present. The call must not raise.
    hw.apply_accelerator_throughput_tweaks(torch.device("cuda"))
