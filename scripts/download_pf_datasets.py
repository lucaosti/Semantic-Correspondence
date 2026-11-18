#!/usr/bin/env python3
"""Download PF-Willow and PF-Pascal (test pairs) under ``data/``.

Usage (from repository root)::

    python scripts/download_pf_datasets.py [--datasets pf_willow pf_pascal] [--out-dir DIR]

Sources: the official Proposal Flow archives (Ham et al., CVPR 2016 / TPAMI 2018) and the
standard test-pair lists of CNNGeometric / WeakAlign (Rocco et al.), also used by NC-Net,
DHPF and CATs. Result::

    <out_dir>/PF-Willow/test_pairs_pf.csv            + PF-dataset/<class>/*.png
    <out_dir>/PF-Pascal/test_pairs_pf_pascal.csv     + PF-dataset-PASCAL/JPEGImages/*.jpg

The archives are extracted unchanged, so the image paths stored in the CSVs resolve
relative to each dataset root. Existing datasets are left untouched.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
_WILLOW = "http://www.di.ens.fr/willow/research"
DATASETS = {
    "pf_willow": (
        "PF-Willow",
        f"{_WILLOW}/proposalflow/dataset/PF-dataset.zip",
        f"{_WILLOW}/cnngeometric/other_resources/test_pairs_pf.csv",
    ),
    "pf_pascal": (
        "PF-Pascal",
        f"{_WILLOW}/proposalflow/dataset/PF-dataset-PASCAL.zip",
        f"{_WILLOW}/cnngeometric/other_resources/test_pairs_pf_pascal.csv",
    ),
}


def _download(url: str, dest: Path) -> None:
    print(f"  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp, tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=dest.parent
    ) as tmp:
        while chunk := resp.read(1 << 20):
            tmp.write(chunk)
    Path(tmp.name).replace(dest)


def prepare(name: str, out_dir: Path) -> None:
    folder, zip_url, csv_url = DATASETS[name]
    root = out_dir / folder
    csv_path = root / csv_url.rsplit("/", 1)[1]
    if csv_path.is_file():
        print(f"{folder}: already present at {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "_archive.zip"
    print(f"{folder}: downloading")
    _download(zip_url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    zip_path.unlink()
    _download(csv_url, csv_path)  # last: its presence marks a complete dataset
    print(f"{folder}: ready at {root}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="*", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    for name in args.datasets:
        prepare(name, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
