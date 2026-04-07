#!/usr/bin/env python3
"""
LoRA fine-tuning on SPair-71k (Task 4, parameter-efficient adaptation).

Only LoRA parameters on the last blocks' MLP linear layers are optimized by default
(see :func:`models.common.lora.apply_lora_to_last_blocks_mlp`).

When this training runs via the orchestrated pipeline (`scripts/run_pipeline.py`), the
pipeline passes explicit ``--epochs`` / ``--patience`` values, overriding this script's
standalone defaults (see `documentation.md`, §8.2).

**Splits:** train on ``train``, early-stop / monitor on ``val``, report on ``test`` via ``scripts/run_pipeline.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from data.dataset import PreprocessMode, SPair71kPairDataset, build_photometric_pair_transform, spair_collate_fn
from data.paths import resolve_spair_root
from models.common.lora import (
    apply_lora_to_last_blocks_mlp,
    apply_lora_to_last_blocks_mlp_sam,
    lora_trainable_parameters,
)
from scripts._training_common import build_backbone, run_gaussian_training_loop
from training.early_stopping import EarlyStopping
from training.engine import correspondence_gaussian_loss_dino_vit, correspondence_gaussian_loss_sam
from training.unfreeze import freeze_all, unfreeze_parameters
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
        default=100,
        help="Pairs per optimizer step (Gaussian loss averages over the batch).",
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
    p.add_argument(
        "--layer-indices",
        type=int,
        default=4,
        help="Intermediate ViT layer for DINO feature extraction (ignored for sam_vit_b).",
    )
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


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1.", file=sys.stderr)
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
    unfreeze_parameters(lora_params)

    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)

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

    layer_indices = args.layer_indices
    use_sam = args.backbone == "sam_vit_b"

    def _step_loss(batch_tensors: dict) -> torch.Tensor:
        if use_sam:
            return correspondence_gaussian_loss_sam(model, batch_tensors)
        return correspondence_gaussian_loss_dino_vit(model, batch_tensors, layer_indices=layer_indices)

    stopper = EarlyStopping(patience=args.patience, mode="min")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    resume_path = os.path.join(
        args.checkpoint_dir, f"{args.backbone}_lora_r{args.rank}_resume.pt"
    )
    lora_meta = {"rank": args.rank, "alpha": args.alpha, "last_blocks": args.last_blocks}

    def _save_best(epoch: int, val_loss: float) -> None:
        path = os.path.join(args.checkpoint_dir, f"{args.backbone}_lora_r{args.rank}_best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "lora": lora_meta,
            },
            path,
        )

    def _epoch_suffix() -> str:
        return f" | lora_params={len(lora_trainable_parameters(model))}"

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
        script_tag="train_lora",
        collate_fn=spair_collate_fn,
        extra_resume_payload={"lora": lora_meta},
        save_best_checkpoint=_save_best,
        epoch_log_suffix=_epoch_suffix,
        log_preamble_extra=f"backbone={args.backbone} ",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
