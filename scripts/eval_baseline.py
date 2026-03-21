#!/usr/bin/env python3
"""
Evaluate the training-free baseline (cosine + argmax) on SPair-71k.

**Final metrics must use split ``test``.** Use ``val`` only for development / sanity checks.

Examples
--------
.. code-block:: bash

   export SPAIR_ROOT=/path/to/SPair-71k
   python scripts/eval_baseline.py --backbone dinov2_vitb14 --split test --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from data.paths import resolve_spair_root
from evaluation.baseline_eval import build_eval_dataloader, evaluate_spair_loader
from evaluation.checkpoint_loader import load_encoder_weights_from_pt
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from utils.hardware import (
    maybe_tune_threads_for_cpu_device,
    pin_memory_for,
    resolve_device_str,
    resolve_num_workers,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPair-71k baseline evaluation (Task 1 / Task 3 WSArgmax).")
    p.add_argument("--spair-root", type=str, default=None, help="Path to SPair-71k (else SPAIR_ROOT / DATASET_ROOT).")
    p.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=[x.value for x in BackboneName],
        help="Foundation backbone (Base variants).",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split.")
    p.add_argument("--dinov2-weights", type=str, default=None, help="Optional local DINOv2 checkpoint path.")
    p.add_argument("--dinov3-weights", type=str, default=None, help="Optional local DINOv3 checkpoint path.")
    p.add_argument("--sam-checkpoint", type=str, default=None, help="Optional SAM ViT-B checkpoint path.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional .pt from train_finetune/train_lora (loads into backbone encoder).",
    )
    p.add_argument("--preprocess", type=str, default="FIXED_RESIZE", help="PreprocessMode enum name (e.g. FIXED_RESIZE).")
    p.add_argument("--height", type=int, default=784)
    p.add_argument("--width", type=int, default=784)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    p.add_argument("--window-soft-argmax", action="store_true", help="Apply inference-only window soft-argmax (Task 3).")
    p.add_argument("--wsa-window", type=int, default=5)
    p.add_argument("--wsa-temperature", type=float, default=1.0)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1 = auto from CPU count and OS).",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, evaluate only the first N pairs (debug).")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, cuda, or mps (default: auto — cuda if available, else mps, else cpu).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = resolve_spair_root(args.spair_root)

    if not os.path.isdir(root):
        print(f"ERROR: SPair-71k not found at: {root}", file=sys.stderr)
        print("Set SPAIR_ROOT to the SPair-71k folder, or DATASET_ROOT to its parent.", file=sys.stderr)
        return 2

    device = torch.device(resolve_device_str(args.device))
    maybe_tune_threads_for_cpu_device(device.type)
    num_workers = resolve_num_workers(args.num_workers)
    pin_mem = pin_memory_for(device)

    cfg = DenseExtractorConfig(
        name=BackboneName(args.backbone),
        dinov2_weights_path=args.dinov2_weights,
        dinov3_weights_path=args.dinov3_weights,
        sam_checkpoint_path=args.sam_checkpoint,
    )
    extractor = DenseFeatureExtractor(cfg, freeze=True)

    if args.checkpoint:
        info = load_encoder_weights_from_pt(extractor, args.checkpoint, map_location="cpu")
        if info["missing"] or info["unexpected"]:
            print(
                f"Checkpoint load: missing={info['missing']} unexpected={info['unexpected']}",
                file=sys.stderr,
            )

    loader = build_eval_dataloader(
        root,
        split=args.split,
        batch_size=1,
        num_workers=num_workers,
        preprocess=args.preprocess,
        output_size_hw=(args.height, args.width),
        pin_memory=pin_mem,
    )
    if args.limit and args.limit > 0:
        from torch.utils.data import DataLoader, Subset

        from data.dataset import spair_collate_fn

        n = min(args.limit, len(loader.dataset))
        loader = DataLoader(
            Subset(loader.dataset, list(range(n))),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=spair_collate_fn,
            pin_memory=pin_mem,
        )

    metrics = evaluate_spair_loader(
        loader,
        extractor,
        alphas=args.alphas,
        use_window_soft_argmax=args.window_soft_argmax,
        wsa_window=args.wsa_window,
        wsa_temperature=args.wsa_temperature,
        device=device,
    )
    print(f"SPair root: {root}")
    print(f"Backbone: {args.backbone}")
    print(f"Split: {args.split}")
    print(f"WSArgmax: {args.window_soft_argmax}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
