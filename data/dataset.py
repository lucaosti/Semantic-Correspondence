"""
SPair-71k dataset utilities and PyTorch dataset for semantic correspondence.

This module follows the on-disk layout used by common benchmarks and tooling
(e.g., SD4Match): images under ``JPEGImages/<category>/``, pair lists under
``Layout/large/{trn,val,test}.txt``, and per-pair JSON annotations under
``PairAnnotation/<split>/``.

**Split policy (project constraint):**

- ``train`` (file ``trn.txt``): training only.
- ``val``: validation / hyperparameter tuning and early stopping.
- ``test``: **final evaluation only** - do not use for model selection.

All symbols, docstrings, and comments are in English by project convention.
"""

from __future__ import annotations

import functools
import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPAIR_CATEGORIES: Tuple[str, ...] = (
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
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "train",
    "tvmonitor",
)

# Maximum number of keypoint slots used in SPair-style padding (matches common benchmarks).
MAX_KEYPOINTS: int = 20

# Sentinel used for padded (invalid) keypoint coordinates in tensor outputs.
INVALID_KP_COORD: float = -2.0


class SplitSpec(str, Enum):
    """
    Dataset split identifiers exposed by this project.

    ``TRAIN`` maps to the SPair filename ``trn.txt`` (not ``train.txt``).
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class PreprocessMode(str, Enum):
    """Image preprocessing strategy for ViT backbones (patch grids).

    Only ``FIXED_RESIZE`` is supported by the project pipeline. The pipeline pins each
    backbone's input size to a patch-aligned multiple via
    ``IMAGE_SIZE_BY_BACKBONE`` in ``scripts/run_pipeline.py`` (e.g. 518 = 37x14 for
    DINOv2, 512 = 32x16 for DINOv3 / SAM-internal-frame), so the simpler resize is
    sufficient and the legacy letterbox / scale-longest variants have been removed.
    """

    FIXED_RESIZE = "fixed_resize"


# ---------------------------------------------------------------------------
# Split / path helpers
# ---------------------------------------------------------------------------


def normalize_split_name(split: Union[str, SplitSpec]) -> SplitSpec:
    """Normalize a user-provided split string (``train``/``trn``/``val``/``test``)."""
    if isinstance(split, SplitSpec):
        return split
    s = str(split).strip().lower()
    if s in {"trn", "train"}:
        return SplitSpec.TRAIN
    if s == "val":
        return SplitSpec.VAL
    if s == "test":
        return SplitSpec.TEST
    raise ValueError(
        f"Unknown split {split!r}. Expected one of: train, trn, val, test."
    )


def spair_split_filename(split: Union[str, SplitSpec]) -> str:
    """Return the SPair on-disk list filename (``trn.txt`` / ``val.txt`` / ``test.txt``)."""
    spec = normalize_split_name(split)
    return "trn.txt" if spec == SplitSpec.TRAIN else f"{spec.value}.txt"


def default_spair_root(dataset_root: Union[str, os.PathLike[str]]) -> str:
    """Return ``<dataset_root>/SPair-71k``."""
    return os.path.join(os.path.abspath(os.fspath(dataset_root)), "SPair-71k")


@dataclass(frozen=True)
class SPairPaths:
    """Resolved filesystem paths for a SPair-71k installation."""

    root: str
    layout_dir: str
    images_dir: str
    pair_ann_dir: str

    @staticmethod
    def from_root(spair_root: Union[str, os.PathLike[str]]) -> "SPairPaths":
        root = os.path.abspath(os.fspath(spair_root))
        return SPairPaths(
            root=root,
            layout_dir=os.path.join(root, "Layout", "large"),
            images_dir=os.path.join(root, "JPEGImages"),
            pair_ann_dir=os.path.join(root, "PairAnnotation"),
        )


def parse_spair_pair_line(line: str) -> Tuple[str, str, str, str]:
    """Parse ``<pair_index>-<src_stem>-<tgt_stem>:<category>``."""
    text = line.strip()
    if ":" not in text:
        raise ValueError(f"Malformed SPair pair line (missing ':'): {line!r}")
    head, category = text.rsplit(":", 1)
    parts = head.split("-", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed SPair pair line (expected 3 '-' groups in head): {line!r}")
    pair_index, src_stem, tgt_stem = parts
    return pair_index, src_stem, tgt_stem, category


def read_split_pair_ids(split_file: Union[str, os.PathLike[str]]) -> List[str]:
    """Read pair identifiers (one per line) from a SPair split list file."""
    path = os.path.abspath(os.fspath(split_file))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Split file not found: {path}")
    pair_ids: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line:
                pair_ids.append(line)
    return pair_ids


def load_pair_annotation_json(annotation_path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    """Load a SPair pair annotation JSON file."""
    path = os.path.abspath(os.fspath(annotation_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Geometry: keypoints, resizing
# ---------------------------------------------------------------------------


def _to_xy_keypoints(tensor_2x_n: torch.Tensor) -> torch.Tensor:
    """Convert keypoints from ``(2, N)`` to ``(N, 2)`` in ``(x, y)`` order."""
    if tensor_2x_n.ndim != 2 or tensor_2x_n.shape[0] != 2:
        raise ValueError(f"Expected a (2, N) keypoint tensor, got {tuple(tensor_2x_n.shape)}")
    return tensor_2x_n.transpose(0, 1).contiguous()


def pad_keypoints_to_max(
    kps_xy: torch.Tensor,
    max_points: int = MAX_KEYPOINTS,
    fill_value: float = INVALID_KP_COORD,
) -> Tuple[torch.Tensor, int]:
    """Pad/truncate keypoints to a fixed maximum count for batching.

    Returns a tuple ``(padded_tensor[(max_points, 2)], n_valid)``.
    """
    if kps_xy.ndim != 2 or kps_xy.shape[-1] != 2:
        raise ValueError(f"Expected (N, 2) keypoints, got {tuple(kps_xy.shape)}")
    n = int(kps_xy.shape[0])
    n_valid = min(n, int(max_points))
    out = torch.full((max_points, 2), float(fill_value), dtype=torch.float32)
    if n_valid > 0:
        out[:n_valid] = kps_xy[:n_valid].to(dtype=torch.float32)
    return out, n_valid


def scale_keypoints_xy(
    kps_xy: torch.Tensor,
    orig_size_xy: Tuple[int, int],
    new_size_xy: Tuple[int, int],
) -> torch.Tensor:
    """Scale keypoints after a pure resize from ``orig_size`` to ``new_size`` (PIL ``(W, H)``)."""
    ow, oh = orig_size_xy
    nw, nh = new_size_xy
    if ow <= 0 or oh <= 0:
        raise ValueError("Original width/height must be positive.")
    sx = float(nw) / float(ow)
    sy = float(nh) / float(oh)
    out = kps_xy.clone()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def pil_resize_bicubic(img: Image.Image, size_xy: Tuple[int, int]) -> Image.Image:
    """Resize a PIL image with bicubic interpolation (``(W, H)`` order)."""
    return img.resize((int(size_xy[0]), int(size_xy[1])), resample=Image.BICUBIC)


def preprocess_pair_images_and_keypoints(
    src_pil: Image.Image,
    tgt_pil: Image.Image,
    src_kps_xy: torch.Tensor,
    tgt_kps_xy: torch.Tensor,
    mode: PreprocessMode = PreprocessMode.FIXED_RESIZE,
    patch_size: int = 14,
    fixed_size_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Resize an image pair and rescale keypoints accordingly.

    Only ``FIXED_RESIZE`` is supported (see :class:`PreprocessMode`).
    """
    if mode != PreprocessMode.FIXED_RESIZE:
        raise ValueError(f"Unsupported preprocess mode: {mode}")
    if fixed_size_hw is None:
        raise ValueError("fixed_size_hw is required for FIXED_RESIZE.")

    src_pil = src_pil.convert("RGB")
    tgt_pil = tgt_pil.convert("RGB")

    out_h, out_w = int(fixed_size_hw[0]), int(fixed_size_hw[1])
    src_orig = src_pil.size  # (w, h)
    tgt_orig = tgt_pil.size

    src_new = pil_resize_bicubic(src_pil, (out_w, out_h))
    tgt_new = pil_resize_bicubic(tgt_pil, (out_w, out_h))
    src_k = scale_keypoints_xy(src_kps_xy, src_orig, (out_w, out_h))
    tgt_k = scale_keypoints_xy(tgt_kps_xy, tgt_orig, (out_w, out_h))
    meta: Dict[str, Any] = {
        "mode": str(mode),
        "patch_size": int(patch_size),
        "out_hw": (out_h, out_w),
    }
    return src_new, tgt_new, src_k, tgt_k, meta


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def build_imagenet_normalize() -> transforms.Normalize:
    """Build ImageNet normalization (torchvision convention)."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Normalize(mean=mean, std=std)


def build_photometric_pair_transform(
    brightness: Tuple[float, float] = (0.85, 1.15),
    contrast: Tuple[float, float] = (0.85, 1.15),
    saturation: Tuple[float, float] = (0.85, 1.15),
    hue: Tuple[float, float] = (-0.05, 0.05),
    p: float = 0.9,
    seed: Optional[int] = None,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Identical photometric jitter on both tensors of a pair (RGB in ``[0, 1]``).

    Geometric transforms are intentionally excluded — they would require coherent
    keypoint remapping. Returns a ``_PhotometricPairTransform`` (top-level callable so
    ``DataLoader(num_workers>0)`` pickles it under both fork and spawn start methods).
    """
    return _PhotometricPairTransform((brightness, contrast, saturation, hue), p, seed)


class _PhotometricPairTransform:
    """Picklable callable implementing :func:`build_photometric_pair_transform`."""

    __slots__ = ("ranges", "p", "seed")

    def __init__(
        self,
        ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        p: float,
        seed: Optional[int],
    ) -> None:
        self.ranges = ranges
        self.p = float(p)
        self.seed = seed

    def __call__(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        if rng.random() > self.p:
            return src, tgt
        vals = tuple(rng.uniform(*r) for r in self.ranges)
        return _photo_apply(src, *vals), _photo_apply(tgt, *vals)


def _photo_apply(t: torch.Tensor, b: float, c: float, s: float, h: float) -> torch.Tensor:
    # `t` is (3, H, W) in [0, 1]; TF accepts tensors so we skip the PIL round-trip.
    t = TF.adjust_brightness(t, b)
    t = TF.adjust_contrast(t, c)
    t = TF.adjust_saturation(t, s)
    t = TF.adjust_hue(t, h)
    return t.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PairRecord:
    pair_id: str
    category: str
    src_path: str
    tgt_path: str
    src_kps_xy: torch.Tensor
    tgt_kps_xy: torch.Tensor
    kp_ids: torch.Tensor
    src_bbox_xyxy: torch.Tensor
    tgt_bbox_xyxy: torch.Tensor
    viewpoint_variation: float
    scale_variation: float
    truncation: float
    occlusion: float


class SPair71kPairDataset(Dataset):
    """SPair-71k image-pair dataset for semantic correspondence (training and eval).

    Yields a dict with batched-friendly tensors, padded keypoints, the bbox-derived
    PCK threshold, and SPair difficulty flags. Only ``PreprocessMode.FIXED_RESIZE`` is
    supported by the project pipeline.
    """

    def __init__(
        self,
        spair_root: Union[str, os.PathLike[str]],
        split: Union[str, SplitSpec],
        category: Union[str, Literal["all"]] = "all",
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

        self.paths = SPairPaths.from_root(spair_root)
        self.split = normalize_split_name(split)
        self.split_file_stem = "trn" if self.split == SplitSpec.TRAIN else self.split.value
        self.category_filter: Optional[str] = None if str(category).lower() == "all" else str(category)
        if self.category_filter is not None and self.category_filter not in SPAIR_CATEGORIES:
            raise ValueError(
                f"Unknown category {self.category_filter!r}. "
                f"Expected one of {list(SPAIR_CATEGORIES)} or 'all'."
            )

        self.preprocess = preprocess
        self.output_size_hw = output_size_hw
        self.patch_size = int(patch_size)
        self.normalize = bool(normalize)
        self.photometric_augment = photometric_augment

        split_list_path = os.path.join(self.paths.layout_dir, spair_split_filename(self.split))
        pair_ids = read_split_pair_ids(split_list_path)
        if self.category_filter is not None:
            pair_ids = [pid for pid in pair_ids if self.category_filter in pid]

        self._categories: List[str] = sorted(
            d for d in os.listdir(self.paths.images_dir)
            if os.path.isdir(os.path.join(self.paths.images_dir, d))
        )
        if not self._categories:
            raise FileNotFoundError(f"No category folders found under: {self.paths.images_dir}")

        ann_split_dir = os.path.join(self.paths.pair_ann_dir, self.split_file_stem)
        if not os.path.isdir(ann_split_dir):
            raise FileNotFoundError(f"Pair annotation directory not found: {ann_split_dir}")

        self._records: List[_PairRecord] = []
        for pair_id in pair_ids:
            ann_path = os.path.join(ann_split_dir, f"{pair_id}.json")
            anno = load_pair_annotation_json(ann_path)
            cat = str(anno["category"])
            if self.category_filter is not None and cat != self.category_filter:
                continue

            _, src_stem, tgt_stem, _ = parse_spair_pair_line(pair_id)
            src_kps = torch.tensor(anno["src_kps"], dtype=torch.float32)
            tgt_kps = torch.tensor(anno["trg_kps"], dtype=torch.float32)
            src_kps_xy = src_kps if src_kps.ndim == 2 and src_kps.shape[1] == 2 else _to_xy_keypoints(src_kps)
            tgt_kps_xy = tgt_kps if tgt_kps.ndim == 2 and tgt_kps.shape[1] == 2 else _to_xy_keypoints(tgt_kps)

            src_bbox = torch.tensor(anno["src_bndbox"], dtype=torch.float32).view(-1)
            tgt_bbox = torch.tensor(anno["trg_bndbox"], dtype=torch.float32).view(-1)
            if src_bbox.numel() != 4 or tgt_bbox.numel() != 4:
                raise ValueError(f"Invalid bounding box tensors in {ann_path}")

            # SPair stores keypoint identifiers as strings (e.g. ``["0", "3"]``); coerce to int.
            kp_ids_raw = anno.get("kps_ids") or anno.get("kps_index") or list(range(int(src_kps_xy.shape[0])))
            kp_ids_t = torch.tensor([int(x) for x in kp_ids_raw], dtype=torch.int64).view(-1)

            src_path = os.path.join(self.paths.images_dir, cat, f"{src_stem}.jpg")
            tgt_path = os.path.join(self.paths.images_dir, cat, f"{tgt_stem}.jpg")
            for p in (src_path, tgt_path):
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing image: {p}")

            self._records.append(
                _PairRecord(
                    pair_id=pair_id,
                    category=cat,
                    src_path=src_path,
                    tgt_path=tgt_path,
                    src_kps_xy=src_kps_xy,
                    tgt_kps_xy=tgt_kps_xy,
                    kp_ids=kp_ids_t,
                    src_bbox_xyxy=src_bbox,
                    tgt_bbox_xyxy=tgt_bbox,
                    viewpoint_variation=float(anno.get("viewpoint_variation", 0)),
                    scale_variation=float(anno.get("scale_variation", 0)),
                    truncation=float(anno.get("truncation", 0)),
                    occlusion=float(anno.get("occlusion", 0)),
                )
            )

        to_tensor = transforms.ToTensor()
        norm = build_imagenet_normalize() if self.normalize else None
        self._to_tensor = to_tensor
        self._normalize = norm
        self._maybe_normalize = functools.partial(_apply_normalize, normalize=norm)

    def __len__(self) -> int:
        return len(self._records)

    def category_index(self, name: str) -> int:
        return self._categories.index(name)

    def _bbox_after_resize(
        self,
        bbox_xyxy: torch.Tensor,
        orig_wh: Tuple[int, int],
    ) -> torch.Tensor:
        ow, oh = orig_wh
        if self.output_size_hw is None:
            raise ValueError("output_size_hw is required.")
        out_h, out_w = int(self.output_size_hw[0]), int(self.output_size_hw[1])
        out = bbox_xyxy.clone().view(4)
        out[0::2] *= out_w / float(ow)
        out[1::2] *= out_h / float(oh)
        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]
        src_pil = Image.open(rec.src_path).convert("RGB")
        tgt_pil = Image.open(rec.tgt_path).convert("RGB")
        src_orig_wh = src_pil.size
        tgt_orig_wh = tgt_pil.size

        src_p2, tgt_p2, src_k2, tgt_k2, _ = preprocess_pair_images_and_keypoints(
            src_pil,
            tgt_pil,
            rec.src_kps_xy.clone(),
            rec.tgt_kps_xy.clone(),
            mode=self.preprocess,
            patch_size=self.patch_size,
            fixed_size_hw=self.output_size_hw,
        )

        # Tensor conversion + identical photometric jitter on both tensors of the pair.
        src_t = self._to_tensor(src_p2)
        tgt_t = self._to_tensor(tgt_p2)
        if self.photometric_augment is not None:
            src_t, tgt_t = self.photometric_augment(src_t, tgt_t)
        src_t = self._maybe_normalize(src_t)
        tgt_t = self._maybe_normalize(tgt_t)

        src_bbox = self._bbox_after_resize(rec.src_bbox_xyxy, src_orig_wh)
        tgt_bbox = self._bbox_after_resize(rec.tgt_bbox_xyxy, tgt_orig_wh)

        src_k_pad, n_src = pad_keypoints_to_max(src_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        tgt_k_pad, n_tgt = pad_keypoints_to_max(tgt_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        n_valid = int(min(n_src, n_tgt))

        # Bbox-based PCK tolerance (max side length) in **preprocessed target** pixel frame.
        tw = tgt_bbox[2] - tgt_bbox[0]
        th = tgt_bbox[3] - tgt_bbox[1]
        pck_threshold_bbox = torch.max(tw, th).clamp_min(1e-6)

        # Pad keypoint identifiers to MAX_KEYPOINTS as well; -1 marks padded slots.
        kp_ids_pad = torch.full((MAX_KEYPOINTS,), -1, dtype=torch.int64)
        if n_valid > 0:
            kp_ids_pad[:n_valid] = rec.kp_ids[:n_valid]

        return {
            "pair_id_str": rec.pair_id,
            "category": rec.category,
            "category_id": torch.tensor([self._categories.index(rec.category)], dtype=torch.int64),
            "src_img": src_t,
            "tgt_img": tgt_t,
            "src_kps": src_k_pad,
            "tgt_kps": tgt_k_pad,
            "kp_ids": kp_ids_pad,
            "n_valid_keypoints": torch.tensor([n_valid], dtype=torch.int64),
            "src_bbox": src_bbox,
            "tgt_bbox": tgt_bbox,
            "pck_threshold_bbox": pck_threshold_bbox,
            "src_imsize": torch.tensor([src_orig_wh[0], src_orig_wh[1]], dtype=torch.float32),
            "tgt_imsize": torch.tensor([tgt_orig_wh[0], tgt_orig_wh[1]], dtype=torch.float32),
            "viewpoint_variation": torch.tensor([rec.viewpoint_variation], dtype=torch.float32),
            "scale_variation": torch.tensor([rec.scale_variation], dtype=torch.float32),
            "truncation": torch.tensor([rec.truncation], dtype=torch.float32),
            "occlusion": torch.tensor([rec.occlusion], dtype=torch.float32),
        }


def _apply_normalize(
    t: torch.Tensor, *, normalize: Optional[transforms.Normalize]
) -> torch.Tensor:
    """Top-level helper so DataLoader workers can pickle the dataset."""
    return normalize(t) if normalize is not None else t


def spair_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensor fields; keep ``pair_id_str`` and ``category`` as ``list[str]``."""
    if len(batch) == 0:
        raise ValueError("Empty batch.")
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        if k in ("pair_id_str", "category"):
            out[k] = [b[k] for b in batch]
            continue
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
