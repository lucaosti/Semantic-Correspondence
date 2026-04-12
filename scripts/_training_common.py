"""
Shared helpers for training scripts (``train_finetune.py``, ``train_lora.py``).

Centralizes duplicated logic so that changes propagate consistently.
"""

from __future__ import annotations

import json as _json
from contextlib import nullcontext
import os
from itertools import islice
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dinov2.backbone import build_dinov2_vit_b14
from models.dinov3.backbone import build_dinov3_vit_b16
from models.sam.backbone import build_sam_vit_b_image_encoder
from training.early_stopping import EarlyStopping

TRAIN_SHUFFLE_SEED_BASE: int = 90210
"""Per-epoch shuffle seed so mid-epoch resume skips the same batch order."""

_ALLOWED_PRECISION = ("auto", "fp32", "bf16", "fp16")


def save_resume_atomic(path: str, payload: dict[str, Any]) -> None:
    """Write a resume checkpoint via atomic rename to avoid half-written files."""
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def build_backbone(
    name: str,
    *,
    dinov2_weights: Optional[str] = None,
    dinov3_weights: Optional[str] = None,
    sam_checkpoint: Optional[str] = None,
) -> nn.Module:
    """Instantiate a pretrained backbone by name string."""
    if name == "dinov2_vitb14":
        return build_dinov2_vit_b14(pretrained=dinov2_weights is None, weights_path=dinov2_weights)
    if name == "dinov3_vitb16":
        return build_dinov3_vit_b16(pretrained=dinov3_weights is None, weights_path=dinov3_weights)
    if name == "sam_vit_b":
        return build_sam_vit_b_image_encoder(checkpoint_path=sam_checkpoint)
    raise ValueError(f"Unknown backbone: {name!r}")


def _torch_load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _stopper_to_dict(stopper: EarlyStopping) -> dict[str, Any]:
    return {
        "patience": stopper.patience,
        "min_delta": stopper.min_delta,
        "mode": stopper.mode,
        "best_value": stopper.best_value,
        "num_bad_epochs": stopper.num_bad_epochs,
        "best_epoch": stopper.best_epoch,
    }


def _apply_stopper_from_blob(stopper: EarlyStopping, s: dict[str, Any]) -> None:
    stopper.best_value = s.get("best_value")
    stopper.num_bad_epochs = int(s.get("num_bad_epochs", 0))
    stopper.best_epoch = int(s.get("best_epoch", -1))
    stopper.patience = int(s.get("patience", stopper.patience))
    stopper.min_delta = float(s.get("min_delta", stopper.min_delta))
    stopper.mode = s.get("mode", stopper.mode)  # type: ignore[assignment]


def load_training_resume(
    resume_path_arg: Optional[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stopper: EarlyStopping,
    grad_scaler: Optional[torch.cuda.amp.GradScaler],
    script_tag: str,
    epochs: int = 0,
) -> tuple[int, int, float]:
    """
    Load ``*_resume.pt`` if ``resume_path_arg`` exists.

    If the checkpoint carries ``training_complete=True`` the training loop will
    be skipped entirely (returns ``epochs`` as ``full_epochs_done``).

    Returns
    -------
    full_epochs_done, batch_in_epoch, best_val
    """
    if not resume_path_arg:
        return 0, 0, float("inf")
    if not os.path.isfile(resume_path_arg):
        print(
            f"{script_tag}: WARNING --resume file missing ({resume_path_arg}); starting from scratch.",
            flush=True,
        )
        return 0, 0, float("inf")

    blob = _torch_load_checkpoint(resume_path_arg, device)

    # Fast path: prior run finished; skip the training loop.
    if blob.get("training_complete"):
        best_val = float(blob.get("best_val", float("inf")))
        print(
            f"{script_tag}: RESUME training already complete "
            f"(file={resume_path_arg} best_val={best_val:.6f}); skipping loop.",
            flush=True,
        )
        return epochs, 0, best_val

    model.load_state_dict(blob["model"], strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    if grad_scaler is not None and "grad_scaler" in blob:
        grad_scaler.load_state_dict(blob["grad_scaler"])
    best_val = float(blob.get("best_val", float("inf")))
    _apply_stopper_from_blob(stopper, blob.get("stopper") or {})
    if "full_epochs_done" in blob:
        full_epochs_done = int(blob["full_epochs_done"])
        batch_in_epoch = int(blob.get("batch_in_epoch", 0))
    else:
        full_epochs_done = int(blob["epoch"]) + 1
        batch_in_epoch = 0
    print(
        f"{script_tag}: RESUME full_epochs_done={full_epochs_done} batch_in_epoch={batch_in_epoch} "
        f"(file={resume_path_arg} best_val={best_val})",
        flush=True,
    )
    if batch_in_epoch:
        print(
            f"{script_tag}: mid-epoch resume; skipping first {batch_in_epoch} batches",
            flush=True,
        )
    return full_epochs_done, batch_in_epoch, best_val


def resolve_precision_mode(requested: str, *, device: torch.device) -> str:
    token = str(requested).strip().lower()
    if token not in _ALLOWED_PRECISION:
        allowed = ", ".join(_ALLOWED_PRECISION)
        raise ValueError(f"precision must be one of: {allowed}. Got {requested!r}.")

    if token == "auto":
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            if props.major >= 8 and torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp16"
        return "fp32"

    if token in ("bf16", "fp16") and device.type != "cuda":
        print(
            f"WARNING: requested precision={token} on device={device.type}; falling back to fp32.",
            flush=True,
        )
        return "fp32"

    return token


def _autocast_ctx(device: torch.device, precision_mode: str):
    if device.type != "cuda" or precision_mode == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision_mode == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_gaussian_training_loop(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_ds: Any,
    val_loader: DataLoader,
    step_loss: Callable[[dict[str, Any]], torch.Tensor],
    stopper: EarlyStopping,
    resume_path: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    dl_kw: dict[str, Any],
    worker_init_fn: Any,
    pin_memory: bool,
    log_batch_interval: int,
    resume_save_interval: int,
    resume_arg: Optional[str],
    precision: str,
    script_tag: str,
    collate_fn: Any,
    extra_resume_payload: Optional[dict[str, Any]] = None,
    save_best_checkpoint: Callable[[int, float], None],
    epoch_log_suffix: Optional[Callable[[], str]] = None,
    log_preamble_extra: str = "",
    history_path: Optional[str] = None,
) -> None:
    """
    Shared train/val loop with mid-epoch resume, periodic resume saves, early stopping.

    ``save_best_checkpoint(epoch, val_loss)`` is called every epoch after validation when
    ``val_loss`` improves the caller-tracked best (caller should update best inside the callback).
    """
    extra_payload = dict(extra_resume_payload) if extra_resume_payload else {}
    precision_mode = resolve_precision_mode(precision, device=device)
    use_grad_scaler = device.type == "cuda" and precision_mode == "fp16"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    full_epochs_done, batch_in_epoch, best_val = load_training_resume(
        resume_arg,
        model=model,
        optimizer=optimizer,
        device=device,
        stopper=stopper,
        grad_scaler=grad_scaler,
        script_tag=script_tag,
        epochs=epochs,
    )

    extra = f" {log_preamble_extra}" if log_preamble_extra else " "
    print(
        f"{script_tag}: device={device}{extra}epochs={epochs} batch_size={batch_size} "
        f"num_workers={num_workers} resume_save_interval={resume_save_interval} "
        f"precision={precision_mode}",
        flush=True,
    )

    resume_skip_remaining = batch_in_epoch
    for epoch in range(full_epochs_done, epochs):
        skip_batches = resume_skip_remaining
        resume_skip_remaining = 0
        gen = torch.Generator()
        gen.manual_seed(TRAIN_SHUFFLE_SEED_BASE + epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            generator=gen,
            worker_init_fn=worker_init_fn,
            **dl_kw,
        )
        n_train_batches = len(train_loader)
        print(f"epoch {epoch + 1}/{epochs} (train batches: {n_train_batches})", flush=True)
        model.train()
        total_loss = 0.0
        n_batches = 0
        train_it = iter(train_loader)
        if skip_batches:
            train_it = islice(train_it, skip_batches, None)
        for batch_idx, batch in enumerate(train_it, start=skip_batches):
            if log_batch_interval and batch_idx % log_batch_interval == 0:
                print(f"  batch {batch_idx}/{n_train_batches}", flush=True)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, precision_mode):
                loss = step_loss(batch)
            if not torch.isfinite(loss):
                continue
            if use_grad_scaler:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            done_in_epoch = batch_idx + 1
            iv = resume_save_interval
            if iv > 0 and (done_in_epoch % iv == 0 or done_in_epoch == n_train_batches):
                payload: dict[str, Any] = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "full_epochs_done": epoch,
                    "batch_in_epoch": done_in_epoch,
                    "best_val": best_val,
                    "stopper": _stopper_to_dict(stopper),
                }
                if use_grad_scaler:
                    payload["grad_scaler"] = grad_scaler.state_dict()
                payload.update(extra_payload)
                save_resume_atomic(resume_path, payload)
                print(
                    f"{script_tag}: saved resume checkpoint (batch_in_epoch={done_in_epoch}/{n_train_batches}) -> {resume_path}",
                    flush=True,
                )
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with _autocast_ctx(device, precision_mode):
                    loss = step_loss(batch)
                if torch.isfinite(loss):
                    val_loss += float(loss.cpu())
                    vn += 1
        val_loss = val_loss / max(vn, 1)

        suffix = epoch_log_suffix() if epoch_log_suffix else ""
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}{suffix}")

        if history_path is not None:
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
            }
            with open(history_path, "a", encoding="utf-8") as _hf:
                _hf.write(_json.dumps(record) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            save_best_checkpoint(epoch, val_loss)

        end_payload: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "full_epochs_done": epoch + 1,
            "batch_in_epoch": 0,
            "best_val": best_val,
            "stopper": _stopper_to_dict(stopper),
        }
        if use_grad_scaler:
            end_payload["grad_scaler"] = grad_scaler.state_dict()
        end_payload.update(extra_payload)
        save_resume_atomic(resume_path, end_payload)

        if stopper.step(val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    # Mark the checkpoint as complete so that if the pipeline re-runs this stage
    # (e.g. due to a Colab disconnect before mark_step_done) the training loop is
    # skipped and the stage completes instantly without re-training from scratch.
    completion_payload: dict[str, Any] = {
        "training_complete": True,
        "best_val": best_val,
        "full_epochs_done": epochs,
        "batch_in_epoch": 0,
    }
    completion_payload.update(extra_payload)
    save_resume_atomic(resume_path, completion_payload)
    print("Done.")


__all__ = [
    "TRAIN_SHUFFLE_SEED_BASE",
    "build_backbone",
    "load_training_resume",
    "resolve_precision_mode",
    "run_gaussian_training_loop",
    "save_resume_atomic",
]
