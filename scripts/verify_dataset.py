#!/usr/bin/env python3
"""
Verify that SPair-71k is present and readable (layout, images, one sample load).

Exit code ``0`` on success, ``2`` if paths are missing or loading fails.

Examples
--------
.. code-block:: bash

   python scripts/verify_dataset.py
   python scripts/verify_dataset.py --spair-root /path/to/SPair-71k
"""

from __future__ import annotations

import argparse
import os
import sys

from data.dataset import PreprocessMode, SPair71kPairDataset
from data.paths import resolve_spair_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify SPair-71k installation and dataset loader.")
    p.add_argument("--spair-root", type=str, default=None, help="Override SPAIR_ROOT / default repo path.")
    return p.parse_args()


def _require(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def main() -> int:
    args = parse_args()
    root = resolve_spair_root(args.spair_root)
    layout = os.path.join(root, "Layout", "large")
    images = os.path.join(root, "JPEGImages")
    ann = os.path.join(root, "PairAnnotation")

    try:
        _require(root, "SPair root")
        _require(layout, "Layout/large (split lists)")
        for name in ("trn.txt", "val.txt", "test.txt"):
            _require(os.path.join(layout, name), name)
        _require(images, "JPEGImages")
        _require(ann, "PairAnnotation")
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Unpack SPair-71k under data/SPair-71k/ or set SPAIR_ROOT.", file=sys.stderr)
        return 2

    for split in ("train", "val", "test"):
        ds = SPair71kPairDataset(
            spair_root=root,
            split=split,
            preprocess=PreprocessMode.FIXED_RESIZE,
            output_size_hw=(784, 784),
            normalize=True,
            photometric_augment=None,
        )
        n = len(ds)
        if n <= 0:
            print(f"ERROR: split {split!r} has zero pairs.", file=sys.stderr)
            return 2
        sample = ds[0]
        assert tuple(sample["src_img"].shape) == (3, 784, 784)
        print(f"OK  split={split:<5}  pairs={n:6d}  sample src tensor {tuple(sample['src_img'].shape)}")

    print(f"SPair root: {root}")
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
