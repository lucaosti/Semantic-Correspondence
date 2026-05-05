#!/usr/bin/env python3
"""Unified training entry point for SPair-71k.

``--mode finetune``: fine-tune the last N transformer blocks (Task 2).
``--mode lora``: LoRA adapters on the last N blocks' MLP linears (Task 4).

Split discipline: train on ``train``, early-stop / monitor on ``val``; report ``test`` via
``scripts/run_pipeline.py``.

Architectural note
------------------
Both modes operate through :class:`DenseFeatureExtractor`. This eliminates the previous
"two paths" split between training (which called the per-backbone extraction functions
directly) and evaluation (which used the wrapper), so a default change in either side
cannot silently desync the two paths anymore.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

from data.dataset import (
    PreprocessMode,
    SPair71kPairDataset,
    build_photometric_pair_transform,
    spair_collate_fn,
)
from data.paths import resolve_spair_root
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from models.common.lora import (
    apply_lora_to_last_blocks_mlp,
    apply_lora_to_last_blocks_mlp_sam,
    lora_trainable_parameters,
)
from scripts._training_common import maybe_compile_model, run_gaussian_training_loop
from training.early_stopping import EarlyStopping
from training.engine import correspondence_gaussian_loss
from training.unfreeze import (
    collect_trainable_parameter_groups,
    freeze_all,
    unfreeze_last_transformer_blocks,
    unfreeze_parameters,
)
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
    p = argparse.ArgumentParser(description="Train on SPair-71k (finetune or LoRA).")
    p.add_argument("--mode", choices=["finetune", "lora"], required=True)
    p.add_argument("--spair-root", type=str, default=None)
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "dinov3_vitb16", "sam_vit_b"],
    )
    p.add_argument("--dinov2-weights", type=str, default=None)
    p.add_argument("--dinov3-weights", type=str, default=None)
    p.add_argument("--sam-checkpoint", type=str, default=None)
    p.add_argument("--last-blocks", type=int, default=2)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (lora mode only).")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (lora mode only).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 5e-5 for finetune, 1e-3 for LoRA.")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="AdamW weight decay (finetune only; LoRA uses 0).")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--preprocess", type=str, default="FIXED_RESIZE")
    p.add_argument("--height", type=int, default=784)
    p.add_argument("--width", type=int, default=784)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--min-delta", type=float, default=0.0,
                   help="Early-stopping tolerance on validation loss. An epoch counts "
                        "as improvement only if val_loss < best - min_delta. "
                        "0.0 (default) = strict: any improvement resets the counter.")
    p.add_argument("--layer-indices", type=int, default=4)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--precision", type=str, default="auto",
                   choices=["auto", "fp32", "bf16", "fp16"])
    p.add_argument("--log-batch-interval", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--resume-save-interval", type=int, default=100)
    p.add_argument("--compile", action="store_true",
                   help="Try torch.compile on CUDA Ampere+ (silently no-op elsewhere).")
    p.add_argument("--accumulation-steps", type=int, default=1,
                   help="Gradient accumulation: optimizer step every N micro-batches. "
                        "Effective batch = batch_size * accumulation_steps.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1.", file=sys.stderr)
        return 2
    if args.accumulation_steps < 1:
        print("ERROR: --accumulation-steps must be >= 1.", file=sys.stderr)
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

    # Build the unified feature extractor (wraps the backbone). The training step calls it
    # directly; the underlying encoder is `extractor.encoder`, on which we apply
    # freezing / unfreezing / LoRA injection.
    extractor_cfg = DenseExtractorConfig(
        name=BackboneName(args.backbone),
        dinov2_weights_path=args.dinov2_weights,
        dinov3_weights_path=args.dinov3_weights,
        sam_checkpoint_path=args.sam_checkpoint,
        dino_layer_indices=args.layer_indices,
    )
    extractor = DenseFeatureExtractor(extractor_cfg, freeze=True).to(device)
    encoder = extractor.encoder
    apply_accelerator_throughput_tweaks(device)

    freeze_all(encoder)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    fused_opt = device.type == "cuda"

    if args.mode == "finetune":
        unfreeze_last_transformer_blocks(encoder, n_blocks=args.last_blocks)
        lr = args.lr if args.lr is not None else 5e-5
        opt = torch.optim.AdamW(
            collect_trainable_parameter_groups(encoder, base_lr=lr),
            weight_decay=args.weight_decay,
            fused=fused_opt,
        )
        tag = f"{args.backbone}_lastblocks{args.last_blocks}"
        history_name = f"{args.backbone}_ft_lb{args.last_blocks}_history.jsonl"
        extra_resume_payload = None
        epoch_log_suffix = None
        script_tag = "train_finetune"

        def _save_best(epoch: int, val_loss: float) -> None:
            ckpt = os.path.join(args.checkpoint_dir, f"{tag}_best.pt")
            torch.save(
                {"model": encoder.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt
            )

    else:  # lora
        if args.backbone == "sam_vit_b":
            lora_params = apply_lora_to_last_blocks_mlp_sam(
                encoder, last_n_blocks=args.last_blocks, rank=args.rank, alpha=args.alpha,
            )
        else:
            lora_params = apply_lora_to_last_blocks_mlp(
                encoder, last_n_blocks=args.last_blocks, rank=args.rank, alpha=args.alpha,
            )
        unfreeze_parameters(lora_params)
        # LoRA injection mutates the encoder modules in-place (replaces the relevant
        # ``nn.Linear`` layers) and adds new parameters; move them to the same device.
        extractor.to(device)
        lr = args.lr if args.lr is not None else 1e-3
        opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0, fused=fused_opt)

        tag = f"{args.backbone}_lora_r{args.rank}"
        history_name = f"{tag}_history.jsonl"
        lora_meta = {"rank": args.rank, "alpha": args.alpha, "last_blocks": args.last_blocks}
        extra_resume_payload = {"lora": lora_meta}
        script_tag = "train_lora"

        def _save_best(epoch: int, val_loss: float) -> None:
            path = os.path.join(args.checkpoint_dir, f"{tag}_best.pt")
            torch.save(
                {
                    "model": encoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "lora": lora_meta,
                },
                path,
            )

        def _epoch_suffix() -> str:
            return f" | lora_params={len(lora_trainable_parameters(encoder))}"

        epoch_log_suffix = _epoch_suffix

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

    stopper = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode="min")
    resume_path = os.path.join(args.checkpoint_dir, f"{tag}_resume.pt")

    # Compile (CUDA Ampere+ only). Keep `extractor` as the uncompiled module so state_dict / save /
    # resume use clean keys; route the forward through the compiled wrapper inside the closure.
    fwd_extractor = maybe_compile_model(extractor, device, requested=args.compile, script_tag=script_tag)

    def _step_loss(batch_tensors: dict) -> torch.Tensor:
        return correspondence_gaussian_loss(fwd_extractor, batch_tensors)

    run_gaussian_training_loop(
        model=extractor,
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
        script_tag=script_tag,
        collate_fn=spair_collate_fn,
        extra_resume_payload=extra_resume_payload,
        save_best_checkpoint=_save_best,
        epoch_log_suffix=epoch_log_suffix,
        log_preamble_extra=f"backbone={args.backbone} ",
        history_path=os.path.join(args.checkpoint_dir, history_name),
        accumulation_steps=args.accumulation_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
