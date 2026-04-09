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

import torch
from torch.utils.data import DataLoader

from data.dataset import PreprocessMode, SPair71kPairDataset, build_photometric_pair_transform, spair_collate_fn
from data.paths import resolve_spair_root
from scripts._training_common import build_backbone, run_gaussian_training_loop
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
        default=20,
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
    p.add_argument("--height", type=int, default=784, help="Input height; must be divisible by patch size. Recommended: DINOv2→518, DINOv3/SAM→512. Fallback: 784.")
    p.add_argument("--width", type=int, default=784)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument(
        "--layer-indices",
        type=int,
        default=4,
        help="Intermediate ViT layer for DINO feature extraction (ignored for sam_vit_b).",
    )
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="Training precision policy. 'auto' picks bf16/fp16 on CUDA and fp32 otherwise.",
    )
    p.add_argument(
        "--log-batch-interval",
        type=int,
        default=100,
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
    model = build_backbone(
        args.backbone,
        dinov2_weights=args.dinov2_weights,
        dinov3_weights=args.dinov3_weights,
        sam_checkpoint=args.sam_checkpoint,
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
        photometric_augment=build_photometric_pair_transform(),
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

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(
        args.checkpoint_dir, f"{args.backbone}_lastblocks{args.last_blocks}_resume.pt"
    )

    layer_indices = args.layer_indices
    use_sam = args.backbone == "sam_vit_b"

    def _step_loss(batch_tensors: dict) -> torch.Tensor:
        if use_sam:
            return correspondence_gaussian_loss_sam(model, batch_tensors)
        return correspondence_gaussian_loss_dino_vit(model, batch_tensors, layer_indices=layer_indices)

    def _save_best(epoch: int, val_loss: float) -> None:
        ckpt = os.path.join(args.checkpoint_dir, f"{args.backbone}_lastblocks{args.last_blocks}_best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt)

    run_gaussian_training_loop(
        model=model,
        optimizer=opt,
        device=device,
        train_ds=train_ds,
        val_loader=val_loader,
        step_loss=_step_loss,
        stopper=stopper,
        resume_path=resume_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=num_workers,
        dl_kw=dl_kw,
        worker_init_fn=_winit,
        pin_memory=pin_memory_for(device),
        log_batch_interval=args.log_batch_interval,
        resume_save_interval=args.resume_save_interval,
        resume_arg=args.resume,
        precision=args.precision,
        script_tag="train_finetune",
        collate_fn=spair_collate_fn,
        extra_resume_payload=None,
        save_best_checkpoint=_save_best,
        epoch_log_suffix=None,
        log_preamble_extra=f"backbone={args.backbone} ",
        history_path=os.path.join(
            args.checkpoint_dir,
            f"{args.backbone}_ft_lb{args.last_blocks}_history.jsonl",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
