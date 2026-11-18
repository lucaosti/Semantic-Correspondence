"""Regression tests for the numerical fixes: loss, PCK frame, WSA and gradient accumulation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.dataset import scale_keypoints_xy
from evaluation.experiment_runner import _to_original_frame
from models.common.window_soft_argmax import refine_predictions_window_soft_argmax
from scripts import _training_common as tc
from training.early_stopping import EarlyStopping
from training.engine import correspondence_gaussian_loss


# ---------------------------------------------------------------------------
# Gaussian CE loss
# ---------------------------------------------------------------------------


class _OneHotExtractor(nn.Module):
    """Returns a fixed 4x4 grid whose 16 cells have orthonormal (one-hot) features."""

    def forward(self, x):
        feats = torch.eye(16).T.reshape(1, 16, 4, 4).expand(x.shape[0], -1, -1, -1)
        return feats, {}


def _cell_centre(i: int) -> float:
    return 4.0 * i + 1.5  # 16-px image, 4 cells: centre of cell i in pixels


def _loss_batch(src_cell, tgt_cell):
    kp = lambda c: torch.tensor([[[_cell_centre(c[0]), _cell_centre(c[1])]]])  # noqa: E731
    return {
        "src_img": torch.zeros(1, 3, 16, 16),
        "tgt_img": torch.zeros(1, 3, 16, 16),
        "src_kps": kp(src_cell),
        "tgt_kps": kp(tgt_cell),
        "n_valid_keypoints": torch.tensor([[1]]),
    }


def test_loss_is_lowest_when_target_is_the_matching_cell():
    ext = _OneHotExtractor()
    right = correspondence_gaussian_loss(ext, _loss_batch((1, 2), (1, 2)))
    wrong = correspondence_gaussian_loss(ext, _loss_batch((1, 2), (3, 0)))
    assert right < wrong


def test_loss_temperature_makes_the_loss_discriminative():
    # Regression: without a temperature, softmax over cosines in [-1, 1] stays nearly flat,
    # so a right and a wrong match get almost the same loss.
    ext = _OneHotExtractor()

    def gap(t):
        right = correspondence_gaussian_loss(ext, _loss_batch((1, 2), (1, 2)), temperature=t)
        wrong = correspondence_gaussian_loss(ext, _loss_batch((1, 2), (3, 0)), temperature=t)
        return float(wrong - right)

    assert gap(1.0) < 1.0 < gap(0.07)


def test_loss_ignores_padded_keypoints():
    ext = _OneHotExtractor()
    b = _loss_batch((1, 2), (1, 2))
    padded = dict(b)
    padded["src_kps"] = torch.cat([b["src_kps"], torch.full((1, 1, 2), -2.0)], dim=1)
    padded["tgt_kps"] = torch.cat([b["tgt_kps"], torch.full((1, 1, 2), -2.0)], dim=1)
    assert torch.allclose(correspondence_gaussian_loss(ext, b), correspondence_gaussian_loss(ext, padded))


# ---------------------------------------------------------------------------
# PCK frame and window soft-argmax
# ---------------------------------------------------------------------------


def test_predictions_map_back_to_original_frame_exactly():
    orig = torch.tensor([[0.0, 0.0], [499.0, 332.0], [250.3, 100.7]])
    resized = scale_keypoints_xy(orig, (500, 333), (518, 518))
    back = _to_original_frame(resized.unsqueeze(0), (518, 518), torch.tensor([[500.0, 333.0]]))
    assert torch.allclose(back[0], orig, atol=1e-3)


def test_wsa_symmetric_peak_returns_peak_cell_centre():
    sims = torch.full((1, 5, 5), -1.0)
    sims[0, 2, 2] = 1.0
    sims[0, 2, 1] = sims[0, 2, 3] = sims[0, 1, 2] = sims[0, 3, 2] = 0.5
    out = refine_predictions_window_soft_argmax(sims, (50, 50), window_size=3, temperature=0.05)
    assert torch.allclose(out[0], torch.tensor([24.5, 24.5]), atol=1e-4)  # centre of cell 2


def test_wsa_two_equal_peaks_return_their_midpoint():
    sims = torch.full((1, 1, 6), -1.0)
    sims[0, 0, 2] = sims[0, 0, 3] = 1.0
    out = refine_predictions_window_soft_argmax(sims, (10, 60), window_size=3, temperature=0.01)
    assert out[0, 0].item() == pytest.approx(29.5, abs=1e-3)  # halfway between cells 2 and 3


def test_wsa_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        refine_predictions_window_soft_argmax(torch.zeros(1, 3, 3), (3, 3), temperature=0.0)


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------


class _Pairs(Dataset):
    def __init__(self, n: int):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, generator=g)
        self.y = torch.randn(n, 1, generator=g)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"src_img": self.x[i], "y": self.y[i]}


def _collate(items):
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def _train(tmp_path, tag, *, n_items, batch_size, accumulation_steps):
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ds = _Pairs(n_items)
    tc.run_gaussian_training_loop(
        model=model, optimizer=opt, device=torch.device("cpu"), train_ds=ds,
        val_loader=DataLoader(ds, batch_size=3, collate_fn=_collate),
        step_loss=lambda b: ((model(b["src_img"]) - b["y"]) ** 2).mean(),
        stopper=EarlyStopping(patience=10), resume_path=str(tmp_path / f"{tag}.pt"),
        epochs=2, batch_size=batch_size, num_workers=0, dl_kw={}, worker_init_fn=None,
        pin_memory=False, log_batch_interval=0, resume_save_interval=1, resume_arg=None,
        precision="fp32", script_tag=tag, collate_fn=_collate,
        save_best_checkpoint=lambda e, v: None, accumulation_steps=accumulation_steps,
    )
    return torch.cat([p.detach().flatten() for p in model.parameters()])


@pytest.mark.parametrize("n_items", [8, 10])
def test_accumulation_matches_large_batch(tmp_path, n_items):
    # 2 x 2 accumulated micro-batches must equal one batch of 4, also when the epoch has a
    # trailing incomplete group (10 items -> 5 micro-batches, the 5th is skipped).
    big = _train(tmp_path, "big", n_items=n_items, batch_size=4, accumulation_steps=1)
    acc = _train(tmp_path, "acc", n_items=n_items, batch_size=2, accumulation_steps=2)
    assert torch.allclose(big, acc, atol=1e-6)
