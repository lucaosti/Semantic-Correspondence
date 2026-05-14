#!/usr/bin/env python3
"""
Download official pretrained weights for all supported backbone variants into ``checkpoints/``.

Backbones:
  DINOv2  ViT-S/14, ViT-B/14, ViT-L/14  — Meta CDN (public, no auth)
  DINOv3  ViT-B/16                        — Meta CDN + HuggingFace mirrors (SHA256 verified)
  DINOv3  ViT-S/16, ViT-L/16             — Meta CDN (auth may be required); set env vars
                                             DINOV3_VITS16_WEIGHTS_URL / DINOV3_VITL16_WEIGHTS_URL
                                             to override with the approved URL.
  SAM     ViT-B, ViT-L                    — Meta CDN + HuggingFace mirrors

Files are skipped if already present. Existing files are never overwritten.

Usage (from repository root)::

   python scripts/download_pretrained_weights.py
   python scripts/download_pretrained_weights.py --variants dinov2_vits14 dinov2_vitb14 sam_vit_b
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

# SHA256 for files where an official checksum is published / community-verified.
_D3_VITB16_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_http_url(url: str, dest: Path, *, expected_sha256: Optional[str] = None) -> None:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Semantic-Correspondence downloader/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        if int(getattr(resp, "status", 200)) >= 400:
            raise RuntimeError(f"HTTP error while downloading {url}")
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dest.parent)) as tmp:
            while True:
                data = resp.read(1024 * 1024)
                if not data:
                    break
                tmp.write(data)
            tmp_path = Path(tmp.name)
    try:
        if expected_sha256 is not None:
            got = _sha256_file(tmp_path)
            if got.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {url}: expected {expected_sha256}, got {got}"
                )
        os.replace(tmp_path, dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _download_with_fallbacks(
    *,
    urls: Sequence[str],
    dest: Path,
    expected_sha256: Optional[str] = None,
) -> None:
    errors: List[str] = []
    for url in urls:
        try:
            print(f"  Trying: {url}")
            _download_http_url(url, dest, expected_sha256=expected_sha256)
            print(f"  Downloaded: {dest.name}")
            return
        except Exception as exc:
            errors.append(f"  - {url}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"All download URLs failed:\n{joined}")


def _download_dinov2(ckpt: Path, variant: str) -> None:
    """Download a DINOv2 variant (vits14 / vitb14 / vitl14)."""
    import torch
    from models.dinov2.hub_loader import _DINOV2_BASE_URL, _make_dinov2_model_name

    arch_map = {"dinov2_vits14": "vit_small", "dinov2_vitb14": "vit_base", "dinov2_vitl14": "vit_large"}
    arch = arch_map[variant]
    mn = _make_dinov2_model_name(arch, 14, 0)
    url = f"{_DINOV2_BASE_URL}/{mn}/{mn}_pretrain.pth"
    dest = ckpt / f"{mn}_pretrain.pth"
    if dest.is_file():
        print(f"Already present: {dest.name}")
        return
    print(f"Downloading {variant} ...")
    torch.hub.download_url_to_file(url, str(dest))
    print(f"Downloaded: {dest.name}")


def _download_dinov3_vitb16(ckpt: Path) -> None:
    """Download DINOv3 ViT-B/16 with SHA256 verification and HuggingFace fallbacks."""
    from models.dinov3.hub_loader import Weights, _make_dinov3_vit_model_url

    dest = ckpt / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    if dest.is_file():
        print(f"Already present: {dest.name}")
        got_sha = _sha256_file(dest)
        if got_sha.lower() != _D3_VITB16_SHA256.lower():
            raise RuntimeError(
                f"Existing DINOv3 ViT-B/16 file hash mismatch for {dest}: {got_sha}. "
                "Delete the file and rerun."
            )
        return
    url_primary = _make_dinov3_vit_model_url(
        patch_size=16, compact_arch_name="vitb", version=None,
        weights=Weights.LVD1689M, hash="73cec8be",
    )
    d3_override = os.environ.get("DINOV3_WEIGHTS_URL", "").strip()
    urls = ([d3_override] if d3_override else []) + [
        url_primary,
        "https://huggingface.co/jaychempan/dinov3/resolve/main/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?download=true",
        "https://huggingface.co/REPA-E/iREPA-collections/resolve/ece5c3539c805644084db6fc299d190a8eab73d8/pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?download=true",
        "https://huggingface.co/XavierJiezou/co2s-models/resolve/main/pretrained/dinov3_vitb16_pretrain_lvd1689m.pth?download=true",
    ]
    print("Downloading DINOv3 ViT-B/16 (with fallbacks + SHA256 check)...")
    _download_with_fallbacks(urls=urls, dest=dest, expected_sha256=_D3_VITB16_SHA256)


def _download_dinov3_size_variant(ckpt: Path, variant: str) -> None:
    """Download DINOv3 ViT-S/16 or ViT-L/16.

    These checkpoints require approved access from Meta.  Use env vars
    ``DINOV3_VITS16_WEIGHTS_URL`` / ``DINOV3_VITL16_WEIGHTS_URL`` to supply
    the approved download URL; the file is then stored under ``checkpoints/``.
    """
    env_key_map = {
        "dinov3_vits16": "DINOV3_VITS16_WEIGHTS_URL",
        "dinov3_vitl16": "DINOV3_VITL16_WEIGHTS_URL",
    }
    filename_map = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m.pth",
    }
    env_key = env_key_map[variant]
    dest = ckpt / filename_map[variant]
    if dest.is_file():
        print(f"Already present: {dest.name}")
        return
    url_override = os.environ.get(env_key, "").strip()
    if not url_override:
        print(
            f"SKIP {variant}: checkpoint not present and {env_key} not set.\n"
            f"  Request access at https://github.com/facebookresearch/dinov3 and set:\n"
            f"    export {env_key}=<approved_url>\n"
            f"  then rerun this script."
        )
        return
    print(f"Downloading {variant} from {env_key} ...")
    _download_with_fallbacks(urls=[url_override], dest=dest)


def _download_sam(ckpt: Path, variant: str) -> None:
    """Download SAM ViT-B or ViT-L."""
    urls_map = {
        "sam_vit_b": [
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "https://huggingface.co/scenario-labs/sam_vit/resolve/1c5c33ad1abe579854e3e2a6228568026aea6758/sam_vit_b_01ec64.pth?download=true",
            "https://huggingface.co/fofr/comfyui/resolve/76574336fbb61a96825dbe0b41bda2f2ec214084/sams/sam_vit_b_01ec64.pth?download=true",
        ],
        "sam_vit_l": [
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_l_0b3195.pth?download=true",
            "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/sams/sam_vit_l_0b3195.pth?download=true",
        ],
    }
    filenames = {"sam_vit_b": "sam_vit_b_01ec64.pth", "sam_vit_l": "sam_vit_l_0b3195.pth"}
    dest = ckpt / filenames[variant]
    if dest.is_file():
        print(f"Already present: {dest.name}")
        return
    print(f"Downloading {variant} (with fallbacks)...")
    _download_with_fallbacks(urls=urls_map[variant], dest=dest)


_ALL_VARIANTS = [
    "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14",
    "dinov3_vitb16", "dinov3_vits16", "dinov3_vitl16",
    "sam_vit_b", "sam_vit_l",
]


def download_variant(ckpt: Path, variant: str) -> None:
    """Download a single backbone variant into ``ckpt/``."""
    if variant.startswith("dinov2"):
        _download_dinov2(ckpt, variant)
    elif variant == "dinov3_vitb16":
        _download_dinov3_vitb16(ckpt)
    elif variant in ("dinov3_vits16", "dinov3_vitl16"):
        _download_dinov3_size_variant(ckpt, variant)
    elif variant.startswith("sam"):
        _download_sam(ckpt, variant)
    else:
        print(f"WARNING: unknown variant {variant!r}, skipping.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download pretrained backbone weights.")
    parser.add_argument(
        "--variants", nargs="*", default=None,
        metavar="BACKBONE",
        help=(
            "Which backbone variants to download. "
            f"Choices: {', '.join(_ALL_VARIANTS)}. "
            "Default: dinov2_vitb14 dinov3_vitb16 sam_vit_b (original set)."
        ),
    )
    args = parser.parse_args(argv)

    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    variants = args.variants if args.variants is not None else ["dinov2_vitb14", "dinov3_vitb16", "sam_vit_b"]

    unknown = [v for v in variants if v not in _ALL_VARIANTS]
    if unknown:
        print(f"ERROR: unknown variants: {unknown}. Allowed: {_ALL_VARIANTS}", file=sys.stderr)
        return 2

    for v in variants:
        download_variant(ckpt, v)

    print(f"\nAll requested weights under {ckpt}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
