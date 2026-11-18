"""PF-Willow and PF-Pascal test pairs for cross-dataset PCK evaluation.

Both use the standard pair lists distributed with CNNGeometric / WeakAlign (also used by
NC-Net, DHPF and CATs) and the official Proposal Flow image archives, prepared by
``scripts/download_pf_datasets.py``::

    <pf_willow_root>/test_pairs_pf.csv          900 pairs, 10 keypoints each
    <pf_willow_root>/PF-dataset/<class>/*.png
    <pf_pascal_root>/test_pairs_pf_pascal.csv   299 pairs
    <pf_pascal_root>/PF-dataset-PASCAL/JPEGImages/*.jpg

Keypoints are MATLAB (1-based) pixel coordinates and are shifted to the 0-based
pixel-centre convention used everywhere else. Items follow the schema of
:class:`data.dataset.SPair71kPairDataset`.

PCK normalisation (standard protocol, e.g. DHPF / CATs), in the **original** target frame:

* **PF-Willow** — ``alpha_bbox``: ``max(W, H)`` of the tightest box around the target
  keypoints (PF-Willow has no object boxes).
* **PF-Pascal** — ``alpha_img``: ``max(W, H)`` of the target image.
"""

from __future__ import annotations

import csv
import functools
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataset import (
    INVALID_KP_COORD,
    MAX_KEYPOINTS,
    PreprocessMode,
    _apply_normalize,
    build_imagenet_normalize,
    pad_keypoints_to_max,
    preprocess_pair_images_and_keypoints,
)

PF_PASCAL_CATEGORIES: Tuple[str, ...] = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)

PF_WILLOW_CSV = "test_pairs_pf.csv"
PF_PASCAL_CSV = "test_pairs_pf_pascal.csv"


def _xy(xs: List[float], ys: List[float]) -> torch.Tensor:
    """MATLAB 1-based ``(x, y)`` lists -> ``(N, 2)`` 0-based tensor."""
    if len(xs) != len(ys):
        raise ValueError(f"x/y length mismatch: {len(xs)} vs {len(ys)}")
    return torch.tensor([xs, ys], dtype=torch.float32).T - 1.0


def _read_willow_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            col = lambda p: [float(r[f"{p}{i}"]) for i in range(1, 11)]  # noqa: E731
            src, tgt = r["imageA"].strip(), r["imageB"].strip()
            rows.append({
                "src": src, "tgt": tgt, "category": src.split("/")[1],
                "src_kps": _xy(col("XA"), col("YA")), "tgt_kps": _xy(col("XB"), col("YB")),
            })
    return rows


def _read_pascal_rows(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            col = lambda k: [float(v) for v in r[k].split(";")]  # noqa: E731
            rows.append({
                "src": r["source_image"].strip(), "tgt": r["target_image"].strip(),
                "category": PF_PASCAL_CATEGORIES[int(r["class"]) - 1],
                "src_kps": _xy(col("XA"), col("YA")), "tgt_kps": _xy(col("XB"), col("YB")),
            })
    return rows


class _PFPairDataset(Dataset):
    csv_name: str = ""

    def __init__(
        self,
        root: Union[str, os.PathLike],
        split: str = "test",
        preprocess: PreprocessMode = PreprocessMode.FIXED_RESIZE,
        output_size_hw: Optional[Tuple[int, int]] = (784, 784),
        patch_size: int = 14,
        normalize: bool = True,
        photometric_augment: Any = None,
    ) -> None:
        super().__init__()
        if str(split) != "test":
            raise ValueError(f"{type(self).__name__} only provides the standard test split, got {split!r}")
        if preprocess != PreprocessMode.FIXED_RESIZE or output_size_hw is None:
            raise ValueError("Only FIXED_RESIZE with an explicit output size is supported.")
        if photometric_augment is not None:
            raise ValueError("PF datasets are evaluation-only.")
        self.root = os.path.abspath(os.fspath(root))
        self.output_size_hw = output_size_hw
        self.patch_size = int(patch_size)
        csv_path = os.path.join(self.root, self.csv_name)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{csv_path} not found. Run  python scripts/download_pf_datasets.py")
        self._records = self._read_rows(csv_path)
        for rec in self._records:
            for key in ("src", "tgt"):
                p = os.path.join(self.root, rec[key])
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing image: {p}")
        self._categories = sorted({r["category"] for r in self._records})
        self._maybe_normalize = functools.partial(
            _apply_normalize, normalize=build_imagenet_normalize() if normalize else None
        )
        self._to_tensor = transforms.ToTensor()

    @staticmethod
    def _read_rows(csv_path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _threshold(self, tgt_kps: torch.Tensor, tgt_wh: Tuple[int, int]) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]
        src_pil = Image.open(os.path.join(self.root, rec["src"])).convert("RGB")
        tgt_pil = Image.open(os.path.join(self.root, rec["tgt"])).convert("RGB")
        src_wh, tgt_wh = src_pil.size, tgt_pil.size
        src_p, tgt_p, src_k, tgt_k, _ = preprocess_pair_images_and_keypoints(
            src_pil, tgt_pil, rec["src_kps"].clone(), rec["tgt_kps"].clone(),
            mode=PreprocessMode.FIXED_RESIZE, patch_size=self.patch_size,
            fixed_size_hw=self.output_size_hw,
        )
        src_k_pad, n_src = pad_keypoints_to_max(src_k, MAX_KEYPOINTS, INVALID_KP_COORD)
        tgt_k_pad, n_tgt = pad_keypoints_to_max(tgt_k, MAX_KEYPOINTS, INVALID_KP_COORD)
        tgt_k_orig_pad, _ = pad_keypoints_to_max(rec["tgt_kps"], MAX_KEYPOINTS, INVALID_KP_COORD)
        if n_src != n_tgt:
            raise ValueError(f"{rec['src']}|{rec['tgt']}: {n_src} vs {n_tgt} keypoints")
        kp_ids = torch.full((MAX_KEYPOINTS,), -1, dtype=torch.int64)
        kp_ids[:n_src] = torch.arange(n_src)
        zero = torch.zeros(1, dtype=torch.float32)
        return {
            "pair_id_str": f"{rec['src']}|{rec['tgt']}",
            "category": rec["category"],
            "category_id": torch.tensor([self._categories.index(rec["category"])], dtype=torch.int64),
            "src_img": self._maybe_normalize(self._to_tensor(src_p)),
            "tgt_img": self._maybe_normalize(self._to_tensor(tgt_p)),
            "src_kps": src_k_pad,
            "tgt_kps": tgt_k_pad,
            "tgt_kps_orig": tgt_k_orig_pad,
            "kp_ids": kp_ids,
            "n_valid_keypoints": torch.tensor([n_src], dtype=torch.int64),
            "src_bbox": torch.zeros(4, dtype=torch.float32),
            "tgt_bbox": torch.zeros(4, dtype=torch.float32),
            "pck_threshold_bbox": self._threshold(rec["tgt_kps"], tgt_wh),
            "src_imsize": torch.tensor(src_wh, dtype=torch.float32),
            "tgt_imsize": torch.tensor(tgt_wh, dtype=torch.float32),
            "viewpoint_variation": zero, "scale_variation": zero,
            "truncation": zero, "occlusion": zero,
            "src_kp_names": [""] * MAX_KEYPOINTS,
        }


class PFWillowPairDataset(_PFPairDataset):
    """PF-Willow test pairs; PCK normalised by the target keypoint bounding box."""

    csv_name = PF_WILLOW_CSV
    _read_rows = staticmethod(_read_willow_rows)

    def _threshold(self, tgt_kps, tgt_wh):
        extent = tgt_kps.max(dim=0).values - tgt_kps.min(dim=0).values
        return extent.max().clamp_min(1e-6)


class PFPascalPairDataset(_PFPairDataset):
    """PF-Pascal test pairs; PCK normalised by the target image size."""

    csv_name = PF_PASCAL_CSV
    _read_rows = staticmethod(_read_pascal_rows)

    def _threshold(self, tgt_kps, tgt_wh):
        return torch.tensor(float(max(tgt_wh)), dtype=torch.float32)


__all__ = ["PF_PASCAL_CATEGORIES", "PFPascalPairDataset", "PFWillowPairDataset"]
