"""PF-Willow and PF-Pascal image-pair datasets for semantic correspondence evaluation.

Both datasets are evaluation benchmarks; neither has a standard training split used by
this project.  They follow the same output schema as :class:`data.dataset.SPair71kPairDataset`
so they can be dropped into the existing PCK evaluation runner without any changes.

PCK normalisation
-----------------
* **PF-Willow** — image-size-based: ``threshold = alpha * max(out_H, out_W)``.
  Stored in ``pck_threshold_bbox`` as ``max(out_H, out_W)`` (a constant for all images
  after fixed-resize preprocessing).
* **PF-Pascal** — bbox-based: ``threshold = alpha * max(bbox_W, bbox_H)``.
  Identical normalisation to SPair-71k.

Expected on-disk layout
-----------------------
PF-Willow::

    <pf_willow_root>/
    ├── car(G)/   car(M)/   car(S)/
    ├── dog(M)/   dog(S)/
    ├── duck(S)/  face(S)/
    ├── motorbike(S)/
    ├── winebottle(M)/  winebottle(S)/
    └── test_pairs_pf.csv

``test_pairs_pf.csv`` columns: ``src_imname``, ``trg_imname``, ``src_kps``, ``trg_kps``,
``n_kps``.  ``*_kps`` are JSON-encoded arrays of ``[x, y]`` pairs (pixel coordinates) or
flat lists ``[x1, y1, x2, y2, ...]``.

PF-Pascal::

    <pf_pascal_root>/
    ├── JPEGImages/<category>/…
    └── annotations/
        ├── trn_pairs.csv
        ├── val_pairs.csv
        └── test_pairs.csv

CSV columns: ``src_imname``, ``trg_imname``, ``src_kps``, ``trg_kps``,
``src_bbox``, ``trg_bbox``, ``category``.  Bboxes are ``[x1, y1, x2, y2]``; keypoints
are JSON arrays of ``[x, y]`` pairs.

Run ``python scripts/download_pf_datasets.py`` to acquire both datasets.
"""

from __future__ import annotations

import ast
import csv
import functools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.dataset import (
    INVALID_KP_COORD,
    MAX_KEYPOINTS,
    PreprocessMode,
    SplitSpec,
    _apply_normalize,
    build_imagenet_normalize,
    normalize_split_name,
    pad_keypoints_to_max,
    preprocess_pair_images_and_keypoints,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

PF_WILLOW_CATEGORIES: Tuple[str, ...] = (
    "car(G)",
    "car(M)",
    "car(S)",
    "dog(M)",
    "dog(S)",
    "duck(S)",
    "face(S)",
    "motorbike(S)",
    "winebottle(M)",
    "winebottle(S)",
)

PF_PASCAL_CATEGORIES: Tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

# CSV split filename for each logical split key.
_PF_PASCAL_CSV: Dict[str, str] = {
    "train": "trn_pairs.csv",
    "val": "val_pairs.csv",
    "test": "test_pairs.csv",
}


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------


def _parse_kps_field(field: str) -> torch.Tensor:
    """Parse a CSV keypoint field into a ``(N, 2)`` float32 tensor of ``(x, y)`` pairs.

    Accepts both JSON arrays of ``[x, y]`` pairs and flat coordinate lists
    ``[x1, y1, x2, y2, ...]``.
    """
    data = ast.literal_eval(field.strip())
    if not data:
        return torch.zeros((0, 2), dtype=torch.float32)
    first = data[0]
    if isinstance(first, (int, float)):
        flat = [float(v) for v in data]
        if len(flat) % 2 != 0:
            raise ValueError(
                f"Flat keypoint list has odd length ({len(flat)}): {field!r}"
            )
        pts = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
    else:
        pts = [[float(p[0]), float(p[1])] for p in data]
    return torch.tensor(pts, dtype=torch.float32)


def _parse_bbox_field(field: str) -> torch.Tensor:
    """Parse a CSV bounding-box field into a ``(4,)`` float32 tensor ``[x1, y1, x2, y2]``."""
    data = ast.literal_eval(field.strip())
    flat = [float(x) for x in data]
    if len(flat) != 4:
        raise ValueError(
            f"Expected 4 bbox values, got {len(flat)}: {field!r}"
        )
    return torch.tensor(flat, dtype=torch.float32)


def _scale_bbox_xyxy(
    bbox: torch.Tensor,
    orig_wh: Tuple[int, int],
    new_wh: Tuple[int, int],
) -> torch.Tensor:
    """Scale a ``(4,)`` bbox ``[x1, y1, x2, y2]`` after image resize.

    Both ``orig_wh`` and ``new_wh`` are in ``(width, height)`` order (PIL convention).
    """
    ow, oh = float(orig_wh[0]), float(orig_wh[1])
    nw, nh = float(new_wh[0]), float(new_wh[1])
    out = bbox.clone().view(4)
    out[0::2] *= nw / ow
    out[1::2] *= nh / oh
    return out


# ---------------------------------------------------------------------------
# PF-Willow dataset
# ---------------------------------------------------------------------------


class PFWillowPairDataset(Dataset):
    """PF-Willow image-pair dataset for semantic correspondence evaluation.

    PF-Willow contains ~900 test pairs across 10 object sub-categories.  There is
    no official train/val split; the entire dataset is used as a test benchmark.
    Passing ``split="train"`` or ``split="val"`` is accepted (normalised) but still
    loads ``test_pairs_pf.csv``.

    **PCK normalisation:** ``threshold = alpha * max(out_H, out_W)`` (image-size-based).
    The ``pck_threshold_bbox`` field stores ``max(out_H, out_W)``, a constant for all
    images under fixed-resize preprocessing.
    """

    def __init__(
        self,
        pf_willow_root: Union[str, os.PathLike],
        split: Union[str, SplitSpec] = "test",
        preprocess: PreprocessMode = PreprocessMode.FIXED_RESIZE,
        output_size_hw: Optional[Tuple[int, int]] = (784, 784),
        patch_size: int = 14,
        normalize: bool = True,
        photometric_augment: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        super().__init__()
        if preprocess != PreprocessMode.FIXED_RESIZE:
            raise ValueError(f"Unsupported preprocess mode: {preprocess}")
        if output_size_hw is None:
            raise ValueError("output_size_hw is required.")

        self.root = os.path.abspath(os.fspath(pf_willow_root))
        self.split = normalize_split_name(split)
        self.preprocess = preprocess
        self.output_size_hw = output_size_hw
        self.patch_size = int(patch_size)
        self.normalize = bool(normalize)
        self.photometric_augment = photometric_augment

        csv_path = os.path.join(self.root, "test_pairs_pf.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"PF-Willow annotation file not found: {csv_path}\n"
                "Run  python scripts/download_pf_datasets.py  to acquire the dataset."
            )

        self._records: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_kps = _parse_kps_field(row["src_kps"])
                trg_kps = _parse_kps_field(row["trg_kps"])
                src_rel = row["src_imname"].strip()
                trg_rel = row["trg_imname"].strip()
                src_path = os.path.join(self.root, src_rel)
                trg_path = os.path.join(self.root, trg_rel)
                for p in (src_path, trg_path):
                    if not os.path.isfile(p):
                        raise FileNotFoundError(f"Missing PF-Willow image: {p}")
                # Category is the top-level directory component of the image path.
                category = src_rel.split("/")[0].split("\\")[0]
                self._records.append({
                    "src_path": src_path,
                    "trg_path": trg_path,
                    "src_kps": src_kps,
                    "trg_kps": trg_kps,
                    "category": category,
                    "pair_id": f"{src_rel}|{trg_rel}",
                })

        to_tensor = transforms.ToTensor()
        norm = build_imagenet_normalize() if self.normalize else None
        self._to_tensor = to_tensor
        self._maybe_normalize = functools.partial(_apply_normalize, normalize=norm)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]
        src_pil = Image.open(rec["src_path"]).convert("RGB")
        tgt_pil = Image.open(rec["trg_path"]).convert("RGB")
        src_orig_wh = src_pil.size
        tgt_orig_wh = tgt_pil.size

        src_p2, tgt_p2, src_k2, tgt_k2, _ = preprocess_pair_images_and_keypoints(
            src_pil,
            tgt_pil,
            rec["src_kps"].clone(),
            rec["trg_kps"].clone(),
            mode=self.preprocess,
            patch_size=self.patch_size,
            fixed_size_hw=self.output_size_hw,
        )

        src_t = self._to_tensor(src_p2)
        tgt_t = self._to_tensor(tgt_p2)
        if self.photometric_augment is not None:
            src_t, tgt_t = self.photometric_augment(src_t, tgt_t)
        src_t = self._maybe_normalize(src_t)
        tgt_t = self._maybe_normalize(tgt_t)

        src_k_pad, n_src = pad_keypoints_to_max(src_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        tgt_k_pad, n_tgt = pad_keypoints_to_max(tgt_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        n_valid = int(min(n_src, n_tgt))

        out_h, out_w = int(self.output_size_hw[0]), int(self.output_size_hw[1])
        pck_threshold_bbox = torch.tensor(float(max(out_h, out_w)), dtype=torch.float32)

        kp_ids_pad = torch.full((MAX_KEYPOINTS,), -1, dtype=torch.int64)
        if n_valid > 0:
            kp_ids_pad[:n_valid] = torch.arange(n_valid, dtype=torch.int64)

        try:
            cat_id = PF_WILLOW_CATEGORIES.index(rec["category"])
        except ValueError:
            cat_id = 0

        return {
            "pair_id_str": rec["pair_id"],
            "category": rec["category"],
            "category_id": torch.tensor([cat_id], dtype=torch.int64),
            "src_img": src_t,
            "tgt_img": tgt_t,
            "src_kps": src_k_pad,
            "tgt_kps": tgt_k_pad,
            "kp_ids": kp_ids_pad,
            "n_valid_keypoints": torch.tensor([n_valid], dtype=torch.int64),
            "src_bbox": torch.zeros(4, dtype=torch.float32),
            "tgt_bbox": torch.zeros(4, dtype=torch.float32),
            "pck_threshold_bbox": pck_threshold_bbox,
            "src_imsize": torch.tensor(
                [float(src_orig_wh[0]), float(src_orig_wh[1])], dtype=torch.float32
            ),
            "tgt_imsize": torch.tensor(
                [float(tgt_orig_wh[0]), float(tgt_orig_wh[1])], dtype=torch.float32
            ),
            "viewpoint_variation": torch.zeros(1, dtype=torch.float32),
            "scale_variation": torch.zeros(1, dtype=torch.float32),
            "truncation": torch.zeros(1, dtype=torch.float32),
            "occlusion": torch.zeros(1, dtype=torch.float32),
            "src_kp_names": [""] * MAX_KEYPOINTS,
        }


# ---------------------------------------------------------------------------
# PF-Pascal dataset
# ---------------------------------------------------------------------------


class PFPascalPairDataset(Dataset):
    """PF-Pascal image-pair dataset for semantic correspondence evaluation.

    PF-Pascal contains ~1 300 pairs across 20 Pascal VOC categories with train /
    val / test splits.

    **PCK normalisation:** ``threshold = alpha * max(bbox_W, bbox_H)`` (bbox-based),
    identical to SPair-71k.
    """

    def __init__(
        self,
        pf_pascal_root: Union[str, os.PathLike],
        split: Union[str, SplitSpec] = "test",
        preprocess: PreprocessMode = PreprocessMode.FIXED_RESIZE,
        output_size_hw: Optional[Tuple[int, int]] = (784, 784),
        patch_size: int = 14,
        normalize: bool = True,
        photometric_augment: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
    ) -> None:
        super().__init__()
        if preprocess != PreprocessMode.FIXED_RESIZE:
            raise ValueError(f"Unsupported preprocess mode: {preprocess}")
        if output_size_hw is None:
            raise ValueError("output_size_hw is required.")

        self.root = os.path.abspath(os.fspath(pf_pascal_root))
        self.split = normalize_split_name(split)
        self.preprocess = preprocess
        self.output_size_hw = output_size_hw
        self.patch_size = int(patch_size)
        self.normalize = bool(normalize)
        self.photometric_augment = photometric_augment

        split_key = self.split.value  # "train", "val", or "test"
        csv_filename = _PF_PASCAL_CSV[split_key]
        csv_path = os.path.join(self.root, "annotations", csv_filename)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"PF-Pascal annotation file not found: {csv_path}\n"
                "Run  python scripts/download_pf_datasets.py  to acquire the dataset."
            )

        self._records: List[Dict[str, Any]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src_kps = _parse_kps_field(row["src_kps"])
                trg_kps = _parse_kps_field(row["trg_kps"])
                src_bbox = _parse_bbox_field(row["src_bbox"])
                trg_bbox = _parse_bbox_field(row["trg_bbox"])
                src_rel = row["src_imname"].strip()
                trg_rel = row["trg_imname"].strip()
                src_path = os.path.join(self.root, src_rel)
                trg_path = os.path.join(self.root, trg_rel)
                for p in (src_path, trg_path):
                    if not os.path.isfile(p):
                        raise FileNotFoundError(f"Missing PF-Pascal image: {p}")
                self._records.append({
                    "src_path": src_path,
                    "trg_path": trg_path,
                    "src_kps": src_kps,
                    "trg_kps": trg_kps,
                    "src_bbox": src_bbox,
                    "trg_bbox": trg_bbox,
                    "category": row["category"].strip(),
                    "pair_id": f"{src_rel}|{trg_rel}",
                })

        to_tensor = transforms.ToTensor()
        norm = build_imagenet_normalize() if self.normalize else None
        self._to_tensor = to_tensor
        self._maybe_normalize = functools.partial(_apply_normalize, normalize=norm)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]
        src_pil = Image.open(rec["src_path"]).convert("RGB")
        tgt_pil = Image.open(rec["trg_path"]).convert("RGB")
        src_orig_wh = src_pil.size
        tgt_orig_wh = tgt_pil.size

        src_p2, tgt_p2, src_k2, tgt_k2, _ = preprocess_pair_images_and_keypoints(
            src_pil,
            tgt_pil,
            rec["src_kps"].clone(),
            rec["trg_kps"].clone(),
            mode=self.preprocess,
            patch_size=self.patch_size,
            fixed_size_hw=self.output_size_hw,
        )

        src_t = self._to_tensor(src_p2)
        tgt_t = self._to_tensor(tgt_p2)
        if self.photometric_augment is not None:
            src_t, tgt_t = self.photometric_augment(src_t, tgt_t)
        src_t = self._maybe_normalize(src_t)
        tgt_t = self._maybe_normalize(tgt_t)

        out_w, out_h = int(self.output_size_hw[1]), int(self.output_size_hw[0])
        src_bbox = _scale_bbox_xyxy(rec["src_bbox"], src_orig_wh, (out_w, out_h))
        tgt_bbox = _scale_bbox_xyxy(rec["trg_bbox"], tgt_orig_wh, (out_w, out_h))

        src_k_pad, n_src = pad_keypoints_to_max(src_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        tgt_k_pad, n_tgt = pad_keypoints_to_max(tgt_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        n_valid = int(min(n_src, n_tgt))

        tw = tgt_bbox[2] - tgt_bbox[0]
        th = tgt_bbox[3] - tgt_bbox[1]
        pck_threshold_bbox = torch.max(tw, th).clamp_min(1e-6)

        kp_ids_pad = torch.full((MAX_KEYPOINTS,), -1, dtype=torch.int64)
        if n_valid > 0:
            kp_ids_pad[:n_valid] = torch.arange(n_valid, dtype=torch.int64)

        try:
            cat_id = PF_PASCAL_CATEGORIES.index(rec["category"])
        except ValueError:
            cat_id = 0

        return {
            "pair_id_str": rec["pair_id"],
            "category": rec["category"],
            "category_id": torch.tensor([cat_id], dtype=torch.int64),
            "src_img": src_t,
            "tgt_img": tgt_t,
            "src_kps": src_k_pad,
            "tgt_kps": tgt_k_pad,
            "kp_ids": kp_ids_pad,
            "n_valid_keypoints": torch.tensor([n_valid], dtype=torch.int64),
            "src_bbox": src_bbox,
            "tgt_bbox": tgt_bbox,
            "pck_threshold_bbox": pck_threshold_bbox,
            "src_imsize": torch.tensor(
                [float(src_orig_wh[0]), float(src_orig_wh[1])], dtype=torch.float32
            ),
            "tgt_imsize": torch.tensor(
                [float(tgt_orig_wh[0]), float(tgt_orig_wh[1])], dtype=torch.float32
            ),
            "viewpoint_variation": torch.zeros(1, dtype=torch.float32),
            "scale_variation": torch.zeros(1, dtype=torch.float32),
            "truncation": torch.zeros(1, dtype=torch.float32),
            "occlusion": torch.zeros(1, dtype=torch.float32),
            "src_kp_names": [""] * MAX_KEYPOINTS,
        }


__all__ = [
    "PF_PASCAL_CATEGORIES",
    "PF_WILLOW_CATEGORIES",
    "PFPascalPairDataset",
    "PFWillowPairDataset",
    "_parse_bbox_field",
    "_parse_kps_field",
]
