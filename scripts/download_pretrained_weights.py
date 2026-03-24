#!/usr/bin/env python3
"""
Download official pretrained weights for DINOv2 ViT-B/14, DINOv3 ViT-B/16, and SAM ViT-B
into ``checkpoints/``.

URLs match ``models/dinov2/hub_loader.py`` and ``models/dinov3/hub_loader.py`` (same as
``torch.hub.load_state_dict_from_url`` used during training). SAM uses
``scripts/download_sam_vit_b.sh``.

Usage (from repository root)::

   python scripts/download_pretrained_weights.py
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
D3_EXPECTED_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_http_url(url: str, dest: Path, *, expected_sha256: str | None = None) -> None:
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
    expected_sha256: str | None = None,
) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            print(f"Trying: {url}")
            _download_http_url(url, dest, expected_sha256=expected_sha256)
            print(f"Downloaded: {dest}")
            return
        except Exception as exc:
            errors.append(f"- {url}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"All download URLs failed:\n{joined}")


def main() -> int:
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch

    from models.dinov2.hub_loader import _DINOV2_BASE_URL, _make_dinov2_model_name
    from models.dinov3.hub_loader import Weights, _make_dinov3_vit_model_url

    ckpt = ROOT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    mb = _make_dinov2_model_name("vit_base", 14, 0)
    url_d2 = f"{_DINOV2_BASE_URL}/{mb}/{mb}_pretrain.pth"
    dest_d2 = ckpt / "dinov2_vitb14_pretrain.pth"
    if not dest_d2.is_file():
        print(f"Downloading DINOv2: {url_d2}")
        torch.hub.download_url_to_file(url_d2, str(dest_d2))
    else:
        print(f"Already present: {dest_d2}")

    url_d3 = _make_dinov3_vit_model_url(
        patch_size=16,
        compact_arch_name="vitb",
        version=None,
        weights=Weights.LVD1689M,
        hash="73cec8be",
    )
    dest_d3 = ckpt / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    if not dest_d3.is_file():
        print("Downloading DINOv3 (with URL fallbacks)...")
        d3_url_override = os.environ.get("DINOV3_WEIGHTS_URL", "").strip()
        fallback_urls = [
            url_d3,
            # Official model repository (if the .pth artifact is available to your account).
            "https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m/resolve/main/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?download=true",
            # Community mirrors with the same known SHA256.
            "https://huggingface.co/REPA-E/iREPA-collections/resolve/ece5c3539c805644084db6fc299d190a8eab73d8/pretrained_models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?download=true",
            "https://huggingface.co/XavierJiezou/co2s-models/resolve/main/pretrained/dinov3_vitb16_pretrain_lvd1689m.pth?download=true",
        ]
        urls = ([d3_url_override] if d3_url_override else []) + fallback_urls
        _download_with_fallbacks(urls=urls, dest=dest_d3, expected_sha256=D3_EXPECTED_SHA256)
    else:
        print(f"Already present: {dest_d3}")
        got_sha = _sha256_file(dest_d3)
        if got_sha.lower() != D3_EXPECTED_SHA256.lower():
            raise RuntimeError(
                f"Existing DINOv3 file hash mismatch for {dest_d3}: {got_sha}. "
                "Delete the file and rerun download_pretrained_weights.py."
            )

    dest_sam = ckpt / "sam_vit_b_01ec64.pth"
    if not dest_sam.is_file():
        print("Downloading SAM (with URL fallbacks)...")
        sam_urls = [
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "https://huggingface.co/scenario-labs/sam_vit/resolve/1c5c33ad1abe579854e3e2a6228568026aea6758/sam_vit_b_01ec64.pth?download=true",
            "https://huggingface.co/fofr/comfyui/resolve/76574336fbb61a96825dbe0b41bda2f2ec214084/sams/sam_vit_b_01ec64.pth?download=true"
        ]
        _download_with_fallbacks(urls=sam_urls, dest=dest_sam)
    else:
        print(f"Already present: {dest_sam}")

    print(f"All pretrained weights under {ckpt}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
