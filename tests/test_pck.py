"""Tests for bbox-normalized PCK helpers."""

from __future__ import annotations

import torch

from evaluation.pck import mean_pck, pck_distance


def test_pck_distance_all_correct_when_identical():
    pred = torch.tensor([[0.0, 0.0], [10.0, 20.0]], dtype=torch.float32)
    gt = pred.clone()
    thr = torch.tensor(100.0)
    correct, valid = pck_distance(pred, gt, thr, alpha=0.1)
    assert valid.all()
    assert correct.all()


def test_pck_distance_ignores_invalid_gt():
    pred = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    gt = torch.tensor([[0.0, 0.0], [-2.0, -2.0]], dtype=torch.float32)
    thr = torch.tensor(50.0)
    correct, valid = pck_distance(pred, gt, thr, alpha=0.1, invalid_value=-2.0)
    assert valid.sum().item() == 1
    assert correct[0].item() is True


def test_mean_pck_zero_when_no_valid():
    pred = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    gt = torch.tensor([[-2.0, -2.0]], dtype=torch.float32)
    thr = torch.tensor(1.0)
    m = mean_pck(pred, gt, thr, alpha=0.1)
    assert m.item() == 0.0
