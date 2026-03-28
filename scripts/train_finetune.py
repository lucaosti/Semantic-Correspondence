#!/usr/bin/env python3
"""
Fine-tune the last transformer blocks on SPair-71k (Task 2).

This script is intentionally minimal: it wires the dataset, backbone, optimizer, Gaussian CE
loss, and early stopping. You still need official backbones on ``PYTHONPATH``.

When this training runs via the orchestrated pipeline (`scripts/run_pipeline.py`), the
pipeline passes explicit ``--epochs`` / ``--patience`` values, overriding this script's
standalone defaults (see `documentation.md`, §8.2).

**Splits:** train on ``train`` (``trn.txt``), select hyperparameters / early-stop on ``val``,
report on ``test`` (run a separate eval script).
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import islice
from typing import Any, Optional

# Per-epoch shuffle must be reproducible so mid-epoch resume skips the same batch order.
_TRAIN_SHUFFLE_SEED_BASE = 90210

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.paths import resolve_spair_root
from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn
from models.dinov2.backbone import build_dinov2_vit_b14
from models.dinov3.backbone import build_dinov3_vit_b16
from models.sam.backbone import build_sam_vit_b_image_encoder
from training.config import EarlyStoppingConfig
from training.early_stopping import EarlyStopping
from training.engine import correspondence_gaussian_loss_dino_vit, correspondence_gaussian_loss_sam
from training.unfreeze import collect_trainable_parameter_groups, freeze_all, unfreeze_last_transformer_blocks
from utils.hardware import (
    apply_accelerator_throughput_tweaks,
    dataloader_extra_kwargs,
    loader_worker_init_for_device,
    maybe_tune_threads_for_cpu_device,
    pin_memory_for,
    resolve_device_str,
    resolve_num_workers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune backbone last blocks on SPair-71k (Task 2).")
    p.add_argument("--spair-root", type=str, default=None)
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov3_vitb16", "sam_vit_b"],
    )
    p.add_argument("--dinov2-weights", type=str, default=None)
    p.add_argument("--dinov3-weights", type=str, default=None)
    p.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to official SAM ViT-B weights (required for sam_vit_b unless you hack pretrained=False).",
    )
    p.add_argument("--last-blocks", type=int, default=2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Pairs per optimizer step (Gaussian loss averages over the batch).",
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto from CPU count and OS).",
    )
    p.add_argument("--preprocess", type=str, default="FIXED_RESIZE")
    p.add_argument("--height", type=int, default=784, help="Must be divisible by backbone patch size (e.g. 784 for ViT-B/14 and /16).")
    p.add_argument("--width", type=int, default=784)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--log-batch-interval",
        type=int,
        default=2500,
        help="Print training batch progress every N steps (0 = only epoch summaries).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Load *_resume.pt (model + optimizer + epoch + early-stopping state) and continue training.",
    )
    p.add_argument(
        "--resume-save-interval",
        type=int,
        default=100,
        help="Save *_resume.pt every N training batches within an epoch (0 = only at end of each epoch).",
    )
    return p.parse_args()


def _save_resume_atomic(path: str, payload: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _build_backbone(
    name: str,
    *,
    d2: Optional[str],
    d3: Optional[str],
    sam_ckpt: Optional[str],
) -> nn.Module:
    if name == "dinov2_vitb14":
        return build_dinov2_vit_b14(pretrained=d2 is None, weights_path=d2)
    if name == "dinov3_vitb16":
        return build_dinov3_vit_b16(pretrained=d3 is None, weights_path=d3)
    if name == "sam_vit_b":
        return build_sam_vit_b_image_encoder(checkpoint_path=sam_ckpt)
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1.", file=sys.stderr)
        return 2
    root = resolve_spair_root(args.spair_root)
    if not os.path.isdir(root):
        print(f"ERROR: SPair-71k not found at: {root}", file=sys.stderr)
        return 2

    if args.backbone == "sam_vit_b" and args.sam_checkpoint is None:
        print("ERROR: --sam-checkpoint is required for backbone sam_vit_b.", file=sys.stderr)
        return 2

    device = torch.device(resolve_device_str(args.device))
    num_workers = resolve_num_workers(args.num_workers, accelerator=device.type)
    maybe_tune_threads_for_cpu_device(device.type, dataloader_workers=num_workers)
    model = _build_backbone(
        args.backbone,
        d2=args.dinov2_weights,
        d3=args.dinov3_weights,
        sam_ckpt=args.sam_checkpoint,
    ).to(device)
    apply_accelerator_throughput_tweaks(device)

    freeze_all(model)
    unfreeze_last_transformer_blocks(model, n_blocks=args.last_blocks)
    opt = torch.optim.AdamW(collect_trainable_parameter_groups(model, base_lr=args.lr), weight_decay=args.weight_decay)

    mode = PreprocessMode[args.preprocess.strip().upper()]
    train_ds = SPair71kPairDataset(
        spair_root=root,
        split="train",
        preprocess=mode,
        output_size_hw=(args.height, args.width),
        normalize=True,
        photometric_augment=None,
    )
    val_ds = SPair71kPairDataset(
        spair_root=root,
        split="val",
        preprocess=mode,
        output_size_hw=(args.height, args.width),
        normalize=True,
        photometric_augment=None,
    )

    dl_kw = dataloader_extra_kwargs(num_workers, for_device=device.type)
    _winit = loader_worker_init_for_device(device.type, num_workers)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=spair_collate_fn,
        pin_memory=pin_memory_for(device),
        worker_init_fn=_winit,
        **dl_kw,
    )

    es_cfg = EarlyStoppingConfig(patience=args.patience, mode="min")
    stopper = EarlyStopping(patience=es_cfg.patience, mode=es_cfg.mode)

    best_val = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(
        args.checkpoint_dir, f"{args.backbone}_lastblocks{args.last_blocks}_resume.pt"
    )
    full_epochs_done = 0
    batch_in_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            try:
                blob = torch.load(args.resume, map_location=device, weights_only=False)
            except TypeError:
                blob = torch.load(args.resume, map_location=device)
            model.load_state_dict(blob["model"], strict=True)
            opt.load_state_dict(blob["optimizer"])
            best_val = float(blob.get("best_val", float("inf")))
            s = blob.get("stopper") or {}
            stopper.best_value = s.get("best_value")
            stopper.num_bad_epochs = int(s.get("num_bad_epochs", 0))
            stopper.best_epoch = int(s.get("best_epoch", -1))
            stopper.patience = int(s.get("patience", stopper.patience))
            stopper.min_delta = float(s.get("min_delta", stopper.min_delta))
            stopper.mode = s.get("mode", stopper.mode)  # type: ignore[assignment]
            if "full_epochs_done" in blob:
                full_epochs_done = int(blob["full_epochs_done"])
                batch_in_epoch = int(blob.get("batch_in_epoch", 0))
            else:
                full_epochs_done = int(blob["epoch"]) + 1
                batch_in_epoch = 0
            print(
                f"train_finetune: RESUME full_epochs_done={full_epochs_done} batch_in_epoch={batch_in_epoch} "
                f"(file={args.resume} best_val={best_val})",
                flush=True,
            )
            if batch_in_epoch:
                print(
                    f"train_finetune: mid-epoch resume; skipping first {batch_in_epoch} batches "
                    f"of epoch {full_epochs_done + 1}/{args.epochs} (1-based)",
                    flush=True,
                )
        else:
            print(
                f"train_finetune: WARNING --resume file missing ({args.resume}); starting from scratch.",
                flush=True,
            )

    print(
        f"train_finetune: device={device} backbone={args.backbone} "
        f"epochs={args.epochs} batch_size={args.batch_size} "
        f"num_workers={num_workers} resume_save_interval={args.resume_save_interval}",
        flush=True,
    )

    layer_indices = 4
    use_sam = args.backbone == "sam_vit_b"

    def _step_loss(batch_tensors: dict) -> torch.Tensor:
        if use_sam:
            return correspondence_gaussian_loss_sam(model, batch_tensors)
        return correspondence_gaussian_loss_dino_vit(model, batch_tensors, layer_indices=layer_indices)

    resume_skip_remaining = batch_in_epoch
    for epoch in range(full_epochs_done, args.epochs):
        skip_batches = resume_skip_remaining
        resume_skip_remaining = 0
        gen = torch.Generator()
        gen.manual_seed(_TRAIN_SHUFFLE_SEED_BASE + epoch)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=spair_collate_fn,
            pin_memory=pin_memory_for(device),
            generator=gen,
            worker_init_fn=_winit,
            **dl_kw,
        )
        n_train_batches = len(train_loader)
        print(f"epoch {epoch + 1}/{args.epochs} (train batches: {n_train_batches})", flush=True)
        model.train()
        total_loss = 0.0
        n_batches = 0
        train_it = iter(train_loader)
        if skip_batches:
            train_it = islice(train_it, skip_batches, None)
        for batch_idx, batch in enumerate(train_it, start=skip_batches):
            if args.log_batch_interval and batch_idx % args.log_batch_interval == 0:
                print(f"  batch {batch_idx}/{n_train_batches}", flush=True)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = _step_loss(batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            done_in_epoch = batch_idx + 1
            iv = args.resume_save_interval
            if iv > 0 and (done_in_epoch % iv == 0 or done_in_epoch == n_train_batches):
                _save_resume_atomic(
                    resume_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch,
                        "full_epochs_done": epoch,
                        "batch_in_epoch": done_in_epoch,
                        "best_val": best_val,
                        "stopper": {
                            "patience": stopper.patience,
                            "min_delta": stopper.min_delta,
                            "mode": stopper.mode,
                            "best_value": stopper.best_value,
                            "num_bad_epochs": stopper.num_bad_epochs,
                            "best_epoch": stopper.best_epoch,
                        },
                    },
                )
                print(
                    f"train_finetune: saved resume checkpoint (batch_in_epoch={done_in_epoch}/{n_train_batches}) -> {resume_path}",
                    flush=True,
                )
        train_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                loss = _step_loss(batch)
                if torch.isfinite(loss):
                    val_loss += float(loss.cpu())
                    vn += 1
        val_loss = val_loss / max(vn, 1)

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(args.checkpoint_dir, f"{args.backbone}_lastblocks{args.last_blocks}_best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt)

        _save_resume_atomic(
            resume_path,
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "full_epochs_done": epoch + 1,
                "batch_in_epoch": 0,
                "best_val": best_val,
                "stopper": {
                    "patience": stopper.patience,
                    "min_delta": stopper.min_delta,
                    "mode": stopper.mode,
                    "best_value": stopper.best_value,
                    "num_bad_epochs": stopper.num_bad_epochs,
                    "best_epoch": stopper.best_epoch,
                },
            },
        )

        if stopper.step(val_loss, epoch):
            print(f"Early stopping at epoch {epoch}")
            break

    if os.path.isfile(resume_path):
        os.remove(resume_path)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
