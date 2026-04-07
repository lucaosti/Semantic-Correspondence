"""Tests for mixed-precision policy resolution in shared training helpers."""

from __future__ import annotations

import pytest
import torch

from scripts._training_common import resolve_precision_mode


def test_resolve_precision_auto_cpu():
    assert resolve_precision_mode("auto", device=torch.device("cpu")) == "fp32"


def test_resolve_precision_forced_non_cuda_falls_back():
    assert resolve_precision_mode("bf16", device=torch.device("cpu")) == "fp32"
    assert resolve_precision_mode("fp16", device=torch.device("cpu")) == "fp32"


def test_resolve_precision_explicit_fp32():
    assert resolve_precision_mode("fp32", device=torch.device("cpu")) == "fp32"


def test_resolve_precision_rejects_invalid_token():
    with pytest.raises(ValueError):
        resolve_precision_mode("half", device=torch.device("cpu"))
