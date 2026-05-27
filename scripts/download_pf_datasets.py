#!/usr/bin/env python3
"""Download and prepare PF-Willow and PF-Pascal datasets under ``data/``.

Usage (from repository root)::

    python scripts/download_pf_datasets.py
    python scripts/download_pf_datasets.py --datasets pf_willow
    python scripts/download_pf_datasets.py --datasets pf_pascal
    python scripts/download_pf_datasets.py --out-dir /path/to/datasets

The script writes datasets to ``<out_dir>/PF-Willow/`` and
``<out_dir>/PF-Pascal/`` respectively.

PF-Willow
---------
Downloaded from the official WILLOW release (Ham et al., CVPR 2016):
  http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip

The original zip contains JPEG images organised in 10 sub-category folders and a
``test_pairs_pf.mat`` annotation file.  If ``scipy`` is available the script
converts the MATLAB file to the CSV format expected by
:class:`data.pf_dataset.PFWillowPairDataset`.  Without ``scipy`` the raw zip is
still extracted and the user can convert manually (see message printed at the end).

PF-Pascal
---------
Annotation CSV files are downloaded from the CATs repository
(Cho et al., NeurIPS 2021), which hosts the community-standard split files
compatible with the HPF / CATs / SFNet evaluation protocol.

The Pascal VOC 2011 JPEG images are NOT downloaded automatically because the
dataset requires manual registration at the Pascal VOC challenge website.
The script prints clear instructions on where to place the images.

Expected final layout
---------------------
::

    data/PF-Willow/
    ├── car(G)/   car(M)/   car(S)/
    ├── dog(M)/   dog(S)/   duck(S)/
    ├── face(S)/  motorbike(S)/
    ├── winebottle(M)/  winebottle(S)/
    └── test_pairs_pf.csv

    data/PF-Pascal/
    ├── JPEGImages/<category>/…  ← place VOC 2011 images here
    └── annotations/
        ├── trn_pairs.csv
        ├── val_pairs.csv
        └── test_pairs.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Download helpers (shared with download_pretrained_weights.py pattern)
# ---------------------------------------------------------------------------

_PF_WILLOW_ZIP_URL = (
    "http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip"
)

# CATs repository raw URLs for PF-Pascal annotation CSVs.
_CATS_BASE = (
    "https://raw.githubusercontent.com/SunghwanHong/"
    "Cost-Aggregation-transformers/main/data/PF-PASCAL"
)
_PF_PASCAL_CSV_URLS = {
    "trn_pairs.csv": f"{_CATS_BASE}/trn_pairs.csv",
    "val_pairs.csv": f"{_CATS_BASE}/val_pairs.csv",
    "test_pairs.csv": f"{_CATS_BASE}/test_pairs.csv",
}


def _download_url(url: str, dest: Path, *, timeout: int = 120) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Semantic-Correspondence downloader/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if int(getattr(resp, "status", 200)) >= 400:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dest.parent)) as tmp:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
            tmp_path = Path(tmp.name)
    os.replace(tmp_path, dest)


def _download_with_fallbacks(urls: Sequence[str], dest: Path) -> None:
    errors: List[str] = []
    for url in urls:
        try:
            print(f"  Trying: {url}")
            _download_url(url, dest)
            print(f"  Saved:  {dest}")
            return
        except Exception as exc:
            errors.append(f"  - {url}: {exc}")
    raise RuntimeError("All download URLs failed:\n" + "\n".join(errors))


# ---------------------------------------------------------------------------
# MATLAB → CSV conversion for PF-Willow
# ---------------------------------------------------------------------------


def _mat_to_kps_json(arr: "np.ndarray") -> str:  # type: ignore[name-defined]
    """Convert a (2, N) or (N, 2) numpy array to a JSON list of [x, y] pairs."""
    import numpy as np  # noqa: PLC0415

    a = np.asarray(arr, dtype=float)
    if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] != 2:
        # (2, N) → transpose to (N, 2)
        a = a.T
    pts = [[float(a[i, 0]), float(a[i, 1])] for i in range(a.shape[0])]
    return json.dumps(pts)


def _convert_pf_willow_mat(mat_path: Path, out_csv: Path) -> None:
    """Convert ``test_pairs_pf.mat`` → ``test_pairs_pf.csv``."""
    try:
        from scipy.io import loadmat  # type: ignore[import-untyped]
        import numpy as np  # noqa: PLC0415
    except ImportError:
        print(
            "WARNING: scipy is not installed — cannot convert test_pairs_pf.mat.\n"
            "  Install scipy (pip install scipy) and re-run, or convert manually.\n"
            f"  The raw MATLAB file is at: {mat_path}"
        )
        return

    mat = loadmat(str(mat_path), squeeze_me=True)
    src_names = mat["src_imnames"]
    trg_names = mat["trg_imnames"]
    src_kps_all = mat["src_kps"]
    trg_kps_all = mat["trg_kps"]

    import csv as _csv  # noqa: PLC0415

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["src_imname", "trg_imname", "src_kps", "trg_kps", "n_kps"])
        n = len(src_names)
        for i in range(n):
            src_nm = str(src_names[i]).strip()
            trg_nm = str(trg_names[i]).strip()
            src_kp_arr = np.asarray(src_kps_all[i], dtype=float)
            trg_kp_arr = np.asarray(trg_kps_all[i], dtype=float)
            src_json = _mat_to_kps_json(src_kp_arr)
            trg_json = _mat_to_kps_json(trg_kp_arr)
            n_kps = json.loads(src_json).__len__()
            writer.writerow([src_nm, trg_nm, src_json, trg_json, n_kps])
    print(f"  Converted: {out_csv} ({n} pairs)")


# ---------------------------------------------------------------------------
# PF-Willow download
# ---------------------------------------------------------------------------


def download_pf_willow(out_dir: Path) -> None:
    """Download and prepare PF-Willow under ``out_dir/PF-Willow/``."""
    dest_dir = out_dir / "PF-Willow"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = dest_dir / "test_pairs_pf.csv"
    if csv_path.is_file():
        print(f"PF-Willow: annotation CSV already present at {csv_path}")
        return

    zip_path = dest_dir / "_pf_dataset.zip"
    if not zip_path.is_file():
        print(f"PF-Willow: downloading from {_PF_WILLOW_ZIP_URL} …")
        try:
            _download_with_fallbacks([_PF_WILLOW_ZIP_URL], zip_path)
        except RuntimeError as exc:
            print(
                f"ERROR: could not download PF-Willow.\n{exc}\n\n"
                "Manual alternative:\n"
                f"  1. Download: {_PF_WILLOW_ZIP_URL}\n"
                f"  2. Extract to: {dest_dir}/\n"
                f"  3. Run this script again to convert the .mat file to CSV."
            )
            return
    else:
        print(f"PF-Willow: using cached zip {zip_path}")

    print("PF-Willow: extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    # The zip typically unpacks into a subdirectory; flatten if needed.
    extracted_roots = [p for p in dest_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(extracted_roots) == 1:
        inner = extracted_roots[0]
        for item in inner.iterdir():
            target = dest_dir / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        inner.rmdir()

    zip_path.unlink(missing_ok=True)

    mat_path = dest_dir / "test_pairs_pf.mat"
    if mat_path.is_file() and not csv_path.is_file():
        print("PF-Willow: converting test_pairs_pf.mat → test_pairs_pf.csv …")
        _convert_pf_willow_mat(mat_path, csv_path)

    if csv_path.is_file():
        print(f"PF-Willow ready at {dest_dir}/")
    else:
        print(
            f"PF-Willow images extracted to {dest_dir}/\n"
            "  To complete setup: install scipy and re-run this script to generate\n"
            "  test_pairs_pf.csv, or convert the .mat file manually."
        )


# ---------------------------------------------------------------------------
# PF-Pascal download
# ---------------------------------------------------------------------------


def download_pf_pascal(out_dir: Path) -> None:
    """Download PF-Pascal annotation CSVs under ``out_dir/PF-Pascal/``."""
    dest_dir = out_dir / "PF-Pascal"
    ann_dir = dest_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    all_present = all((ann_dir / fn).is_file() for fn in _PF_PASCAL_CSV_URLS)
    if all_present:
        print(f"PF-Pascal: annotation CSVs already present at {ann_dir}/")
    else:
        print("PF-Pascal: downloading annotation CSVs from CATs repository …")
        for filename, url in _PF_PASCAL_CSV_URLS.items():
            csv_path = ann_dir / filename
            if csv_path.is_file():
                print(f"  Already present: {filename}")
                continue
            try:
                _download_with_fallbacks([url], csv_path)
            except RuntimeError as exc:
                print(
                    f"  ERROR downloading {filename}: {exc}\n"
                    f"  Manual alternative: download from\n  {url}\n"
                    f"  and place at {csv_path}"
                )

    img_dir = dest_dir / "JPEGImages"
    if not img_dir.is_dir() or not any(img_dir.iterdir()):
        print(
            "\nPF-Pascal images (Pascal VOC 2011) are NOT downloaded automatically.\n"
            "Follow these steps:\n"
            "  1. Register and download VOC 2011 from:\n"
            "       http://host.robots.ox.ac.uk/pascal/VOC/voc2011/\n"
            "  2. Extract so that JPEG images are accessible as:\n"
            f"       {dest_dir}/JPEGImages/<category>/<image>.jpg\n"
            "     or equivalently set  PF_PASCAL_ROOT  to point at a folder that\n"
            "     already contains  JPEGImages/  and  annotations/."
        )
    else:
        print(f"PF-Pascal ready at {dest_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_ALL_DATASETS = ["pf_willow", "pf_pascal"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare PF-Willow and PF-Pascal datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        metavar="DATASET",
        help=(
            f"Which datasets to prepare. Choices: {', '.join(_ALL_DATASETS)}. "
            "Default: both."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Parent directory for dataset folders (default: <repo>/data/).",
    )
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    datasets = args.datasets if args.datasets is not None else _ALL_DATASETS
    unknown = [d for d in datasets if d not in _ALL_DATASETS]
    if unknown:
        print(
            f"ERROR: unknown datasets: {unknown}. Allowed: {_ALL_DATASETS}",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "pf_willow" in datasets:
        download_pf_willow(out_dir)

    if "pf_pascal" in datasets:
        download_pf_pascal(out_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
