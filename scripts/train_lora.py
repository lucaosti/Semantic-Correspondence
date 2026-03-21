#!/usr/bin/env python3
"""
LoRA fine-tuning on SPair-71k (Task 4, parameter-efficient adaptation).

Only LoRA parameters on the last blocks' MLP linear layers are optimized by default
(see :func:`models.common.lora.apply_lora_to_last_blocks_mlp`).

**Splits:** train on ``train``, early-stop / monitor on ``val``, report on ``test`` via ``scripts/eval_baseline.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn
from data.paths import resolve_spair_root
from models.dinov2.backbone import build_dinov2_vit_b14
from models.dinov3.backbone import build_dinov3_vit_b16
from models.sam.backbone import build_sam_vit_b_image_encoder
from models.common.lora import (
    apply_lora_to_last_blocks_mlp,
    apply_lora_to_last_blocks_mlp_sam,
    lora_trainable_parameters,
)
from training.early_stopping import EarlyStopping
from training.engine import correspondence_gaussian_loss_dino_vit, correspondence_gaussian_loss_sam
from training.unfreeze import freeze_all
from utils.hardware import (
    dataloader_extra_kwargs,
    maybe_tune_threads_for_cpu_device,
    pin_memory_for,
    resolve_device_str,
    resolve_num_workers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning (Task 4) on ViT-B backbones.")
    p.add_argument("--spair-root", type=str, default=None)
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov3_vitb16", "sam_vit_b"],
    )
    p.add_argument("--dinov2-weights", type=str, default=None)
    p.add_argument("--dinov3-weights", type=str, default=None)
    p.add_argument("--sam-checkpoint", type=str, default=None, help="Required for sam_vit_b.")
    p.add_argument("--last-blocks", type=int, default=2, help="How many terminal blocks receive LoRA on MLP linears.")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    p.add_argument("--epochs", type=int, default=2, help="Short schedules are typical for PEFT on SPair-71k.")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Must be 1 (current Gaussian loss expects a single pair per step).",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto from CPU count and OS).",
    )
    p.add_argument("--preprocess", type=str, default="FIXED_RESIZE")
    p.add_argument("--height", type=int, default=784, help="Must be divisible by backbone patch size (e.g. 784 for ViT-B/14 and /16).")
    p.add_argument("--width", type=int, default=784)
    p.add_argument("--patience", type=int, default=3)
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
    return p.parse_args()


def _build_backbone(name: str, *, d2, d3, sam_ckpt) -> nn.Module:
    if name == "dinov2_vitb14":
        return build_dinov2_vit_b14(pretrained=d2 is None, weights_path=d2)
    if name == "dinov3_vitb16":
        return build_dinov3_vit_b16(pretrained=d3 is None, weights_path=d3)
    if name == "sam_vit_b":
        return build_sam_vit_b_image_encoder(checkpoint_path=sam_ckpt)
    raise ValueError(name)


def main() -> int:
    args = parse_args()
    if args.batch_size != 1:
        print("ERROR: --batch-size must be 1 for the Gaussian loss helpers.", file=sys.stderr)
        return 2
    root = resolve_spair_root(args.spair_root)
    if not os.path.isdir(root):
        print(f"ERROR: SPair-71k not found at: {root}", file=sys.stderr)
        print("Extract the dataset into <repo>/data/SPair-71k (see data/SPair-71k/README.md).", file=sys.stderr)
        return 2

    if args.backbone == "sam_vit_b" and args.sam_checkpoint is None:
        print("ERROR: --sam-checkpoint is required for backbone sam_vit_b.", file=sys.stderr)
        return 2

    device = torch.device(resolve_device_str(args.device))
    maybe_tune_threads_for_cpu_device(device.type)
    num_workers = resolve_num_workers(args.num_workers)
    model = _build_backbone(
        args.backbone,
        d2=args.dinov2_weights,
        d3=args.dinov3_weights,
        sam_ckpt=args.sam_checkpoint,
    ).to(device)

    freeze_all(model)
    if args.backbone == "sam_vit_b":
        lora_params = apply_lora_to_last_blocks_mlp_sam(
            model,
            last_n_blocks=args.last_blocks,
            rank=args.rank,
            alpha=args.alpha,
        )
    else:
        lora_params = apply_lora_to_last_blocks_mlp(
            model,
            last_n_blocks=args.last_blocks,
            rank=args.rank,
            alpha=args.alpha,
        )
    for p in lora_params:
        p.requires_grad = True

    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)

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

    dl_kw = dataloader_extra_kwargs(num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=spair_collate_fn,
        pin_memory=pin_memory_for(device),
        **dl_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=spair_collate_fn,
        pin_memory=pin_memory_for(device),
        **dl_kw,
    )

    layer_indices = 4
    use_sam = args.backbone == "sam_vit_b"

    def _step_loss(batch_tensors: dict) -> torch.Tensor:
        if use_sam:
            return correspondence_gaussian_loss_sam(model, batch_tensors)
        return correspondence_gaussian_loss_dino_vit(model, batch_tensors, layer_indices=layer_indices)

    stopper = EarlyStopping(patience=args.patience, mode="min")
    best_val = float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(
        args.checkpoint_dir, f"{args.backbone}_lora_r{args.rank}_resume.pt"
    )
    start_epoch = -1
    if args.resume:
        if os.path.isfile(args.resume):
            try:
                blob = torch.load(args.resume, map_location=device, weights_only=False)
            except TypeError:
                blob = torch.load(args.resume, map_location=device)
            model.load_state_dict(blob["model"], strict=True)
            opt.load_state_dict(blob["optimizer"])
            start_epoch = int(blob["epoch"])
            best_val = float(blob.get("best_val", float("inf")))
            s = blob.get("stopper") or {}
            stopper.best_value = s.get("best_value")
            stopper.num_bad_epochs = int(s.get("num_bad_epochs", 0))
            stopper.best_epoch = int(s.get("best_epoch", -1))
            stopper.patience = int(s.get("patience", stopper.patience))
            stopper.min_delta = float(s.get("min_delta", stopper.min_delta))
            stopper.mode = s.get("mode", stopper.mode)  # type: ignore[assignment]
            print(
                f"train_lora: RESUME from epoch {start_epoch + 1}/{args.epochs} "
                f"(file={args.resume} best_val={best_val})",
                flush=True,
            )
        else:
            print(
                f"train_lora: WARNING --resume file missing ({args.resume}); starting from scratch.",
                flush=True,
            )

    n_train_batches = len(train_loader)
    print(
        f"train_lora: device={device} backbone={args.backbone} "
        f"epochs={args.epochs} batches_per_epoch={n_train_batches} batch_size={args.batch_size} "
        f"num_workers={num_workers}",
        flush=True,
    )

    for epoch in range(start_epoch + 1, args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs} (train batches: {n_train_batches})", flush=True)
        model.train()
        total = 0.0
        n_batches = 0
        for batch_idx, batch in enumerate(train_loader):
            if args.log_batch_interval and batch_idx % args.log_batch_interval == 0:
                print(f"  batch {batch_idx}/{n_train_batches}", flush=True)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            loss = _step_loss(batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            n_batches += 1
        train_loss = total / max(n_batches, 1)

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

        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} | lora_params={len(lora_trainable_parameters(model))}")

        if val_loss < best_val:
            best_val = val_loss
            path = os.path.join(args.checkpoint_dir, f"{args.backbone}_lora_r{args.rank}_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "lora": {"rank": args.rank, "alpha": args.alpha, "last_blocks": args.last_blocks},
                },
                path,
            )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "stopper": {
                    "patience": stopper.patience,
                    "min_delta": stopper.min_delta,
                    "mode": stopper.mode,
                    "best_value": stopper.best_value,
                    "num_bad_epochs": stopper.num_bad_epochs,
                    "best_epoch": stopper.best_epoch,
                },
                "lora": {"rank": args.rank, "alpha": args.alpha, "last_blocks": args.last_blocks},
            },
            resume_path,
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
