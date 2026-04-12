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
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

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
    """
    Image preprocessing strategies relevant to ViT backbones (patch grids).

    - ``FIXED_RESIZE``: Resize to a fixed ``(height, width)`` (may distort aspect ratio
      if ``height != width`` unless you set them from the same scaled size).
    - ``LETTERBOX_PATCH_GRID``: Letterbox (preserve aspect ratio) into a target box
      whose height and width are **multiples of ``patch_size``**, then map keypoints
      through the same scale and padding transform.
    - ``SCALE_LONGEST_ROUND``: Scale so the longest side equals ``longest_side``, then
      **independently** round height and width to the nearest multiple of
      ``patch_size`` (small aspect-ratio change; simple and patch-aligned).
    """

    FIXED_RESIZE = "fixed_resize"
    LETTERBOX_PATCH_GRID = "letterbox_patch_grid"
    SCALE_LONGEST_ROUND = "scale_longest_round"


# ---------------------------------------------------------------------------
# Split / path helpers
# ---------------------------------------------------------------------------


def normalize_split_name(split: Union[str, SplitSpec]) -> SplitSpec:
    """
    Normalize a user-provided split string to :class:`SplitSpec`.

    Accepts aliases such as ``"trn"`` for training.

    Parameters
    ----------
    split:
        One of ``train``, ``trn``, ``val``, ``test`` (case-insensitive).

    Returns
    -------
    SplitSpec
        Canonical enum value.

    Raises
    ------
    ValueError
        If ``split`` is not a recognized split name.
    """
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
    """
    Return the SPair on-disk list filename (without directory) for a logical split.

    SPair stores the training list as ``trn.txt``; this function maps the logical
    ``train`` split to that filename.

    Parameters
    ----------
    split:
        Logical split (``train`` / ``val`` / ``test``).

    Returns
    -------
    str
        One of ``trn.txt``, ``val.txt``, ``test.txt``.
    """
    spec = normalize_split_name(split)
    return "trn.txt" if spec == SplitSpec.TRAIN else f"{spec.value}.txt"


def default_spair_root(dataset_root: Union[str, os.PathLike[str]]) -> str:
    """
    Return the conventional SPair-71k root directory under a user dataset root.

    Parameters
    ----------
    dataset_root:
        Path that **contains** the ``SPair-71k`` folder (not the folder itself).

    Returns
    -------
    str
        ``<dataset_root>/SPair-71k``.
    """
    return os.path.join(os.path.abspath(os.fspath(dataset_root)), "SPair-71k")


@dataclass(frozen=True)
class SPairPaths:
    """
    Resolved filesystem paths for a SPair-71k installation.

    Attributes
    ----------
    root:
        Path to the ``SPair-71k`` directory.
    layout_dir:
        Directory containing split pair-list files (``Layout/large``).
    images_dir:
        Directory containing ``JPEGImages``.
    pair_ann_dir:
        Directory containing per-split pair annotation folders (``PairAnnotation``).
    """

    root: str
    layout_dir: str
    images_dir: str
    pair_ann_dir: str

    @staticmethod
    def from_root(spair_root: Union[str, os.PathLike[str]]) -> "SPairPaths":
        """
        Build path bundle from the ``SPair-71k`` root directory.

        Parameters
        ----------
        spair_root:
            Absolute or relative path to the extracted ``SPair-71k`` folder.

        Returns
        -------
        SPairPaths
            Standard subpaths under ``spair_root``.
        """
        root = os.path.abspath(os.fspath(spair_root))
        return SPairPaths(
            root=root,
            layout_dir=os.path.join(root, "Layout", "large"),
            images_dir=os.path.join(root, "JPEGImages"),
            pair_ann_dir=os.path.join(root, "PairAnnotation"),
        )


def parse_spair_pair_line(line: str) -> Tuple[str, str, str, str]:
    """
    Parse a single pair id line from SPair split files.

    The expected format (as used by SD4Match / SPair tooling) is::

        <pair_index>-<src_stem>-<tgt_stem>:<category>

    Example::

        0-2008_006481-2008_007698:aeroplane

    Parameters
    ----------
    line:
        Raw text line (typically stripped).

    Returns
    -------
    tuple[str, str, str, str]
        ``(pair_index, src_stem, tgt_stem, category)``.

    Raises
    ------
    ValueError
        If the line does not match the expected pattern.
    """
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
    """
    Read pair identifiers (one per line) from a SPair split list file.

    Blank lines are skipped. Each non-empty line is a **pair id string** used as the
    basename for the JSON annotation file in ``PairAnnotation/<split>/``.

    Parameters
    ----------
    split_file:
        Path to ``trn.txt``, ``val.txt``, or ``test.txt``.

    Returns
    -------
    list[str]
        Pair id strings in file order.
    """
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
    """
    Load a SPair pair annotation JSON file.

    Parameters
    ----------
    annotation_path:
        Path to ``<pair_id>.json``.

    Returns
    -------
    dict[str, Any]
        Parsed JSON object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = os.path.abspath(os.fspath(annotation_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Geometry: keypoints, resizing, padding, patch rounding
# ---------------------------------------------------------------------------


def round_side_to_patch_multiple(size: int, patch_size: int) -> int:
    """
    Round a spatial side length to the nearest multiple of ``patch_size``.

    This follows the common ViT practice of using input heights/widths divisible by
    the patch size (e.g., rounding 768 -> 770 for patch size 14).

    Parameters
    ----------
    size:
        Side length in pixels (height or width).
    patch_size:
        ViT patch size (e.g., 14 or 16).

    Returns
    -------
    int
        Rounded size, at least ``patch_size``.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    rounded = int(round(float(size) / float(patch_size)) * float(patch_size))
    return max(patch_size, rounded)


def _to_xy_keypoints(tensor_2x_n: torch.Tensor) -> torch.Tensor:
    """
    Convert keypoints from shape ``(2, N)`` to ``(N, 2)`` in ``(x, y)`` order.

    Parameters
    ----------
    tensor_2x_n:
        Tensor with rows ``[x1..xN], [y1..yN]``.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(N, 2)``.
    """
    if tensor_2x_n.ndim != 2 or tensor_2x_n.shape[0] != 2:
        raise ValueError(f"Expected a (2, N) keypoint tensor, got {tuple(tensor_2x_n.shape)}")
    return tensor_2x_n.transpose(0, 1).contiguous()


def pad_keypoints_to_max(
    kps_xy: torch.Tensor,
    max_points: int = MAX_KEYPOINTS,
    fill_value: float = INVALID_KP_COORD,
) -> Tuple[torch.Tensor, int]:
    """
    Pad/truncate keypoints to a fixed maximum count for batching.

    Parameters
    ----------
    kps_xy:
        ``(N, 2)`` keypoints in pixel coordinates.
    max_points:
        Maximum number of points to keep (extra points are truncated).
    fill_value:
        Coordinate value used for padded slots.

    Returns
    -------
    tuple[torch.Tensor, int]
        Padded tensor of shape ``(max_points, 2)`` and ``n_valid`` (<= max_points).
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
    """
    Scale keypoints after a pure resize from ``orig_size`` to ``new_size``.

    Parameters
    ----------
    kps_xy:
        ``(N, 2)`` keypoints in **original** pixel space. Invalid points should already
        be marked with sentinel coordinates (e.g., ``-2``) if you rely on masking;
        this function scales **all** rows identically.
    orig_size_xy:
        ``(width, height)`` of the original image (PIL order).
    new_size_xy:
        ``(width, height)`` after resizing.

    Returns
    -------
    torch.Tensor
        Scaled ``(N, 2)`` keypoints.
    """
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


def letterbox_affine(
    orig_size_xy: Tuple[int, int],
    out_size_xy: Tuple[int, int],
) -> Tuple[float, int, int, int, int]:
    """
    Compute letterboxing parameters mapping original image space to a canvas.

    The image is scaled uniformly by ``scale = min(out_w/ow, out_h/oh)``, resized to
    ``(round(ow*scale), round(oh*scale))``, and pasted at integer offset
    ``(pad_left, pad_top)`` into an ``out_w x out_h`` canvas.

    Parameters
    ----------
    orig_size_xy:
        ``(width, height)`` of the source image.
    out_size_xy:
        ``(width, height)`` of the output canvas.

    Returns
    -------
    tuple[float, int, int, int, int]
        ``(scale, pad_left, pad_top, new_w, new_h)`` where ``new_w/new_h`` are the
        resized content size before padding.
    """
    ow, oh = orig_size_xy
    out_w, out_h = out_size_xy
    if ow <= 0 or oh <= 0 or out_w <= 0 or out_h <= 0:
        raise ValueError("Image and output sizes must be positive.")
    scale = min(out_w / ow, out_h / oh)
    new_w = int(round(ow * scale))
    new_h = int(round(oh * scale))
    pad_left = (out_w - new_w) // 2
    pad_top = (out_h - new_h) // 2
    return float(scale), int(pad_left), int(pad_top), int(new_w), int(new_h)


def map_keypoints_after_letterbox(
    kps_xy: torch.Tensor,
    scale: float,
    pad_left: int,
    pad_top: int,
) -> torch.Tensor:
    """
    Map keypoints through the same letterbox transform as the image.

    Parameters
    ----------
    kps_xy:
        ``(N, 2)`` keypoints in **original** pixel coordinates.
    scale:
        Uniform scale applied to the image content.
    pad_left, pad_top:
        Integer paste offsets on the canvas.

    Returns
    -------
    torch.Tensor
        Transformed ``(N, 2)`` keypoints in output canvas coordinates.
    """
    out = kps_xy.clone()
    out[:, 0] = out[:, 0] * scale + float(pad_left)
    out[:, 1] = out[:, 1] * scale + float(pad_top)
    return out


def pil_resize_bicubic(img: Image.Image, size_xy: Tuple[int, int]) -> Image.Image:
    """
    Resize a PIL image with bicubic interpolation.

    Parameters
    ----------
    img:
        RGB PIL image.
    size_xy:
        Target ``(width, height)``.

    Returns
    -------
    PIL.Image.Image
        Resized image.
    """
    return img.resize((int(size_xy[0]), int(size_xy[1])), resample=Image.BICUBIC)


def preprocess_pair_images_and_keypoints(
    src_pil: Image.Image,
    tgt_pil: Image.Image,
    src_kps_xy: torch.Tensor,
    tgt_kps_xy: torch.Tensor,
    mode: PreprocessMode,
    patch_size: int = 14,
    fixed_size_hw: Optional[Tuple[int, int]] = None,
    longest_side: int = 784,
    letterbox_fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Apply a consistent geometric preprocessing policy to an image pair and keypoints.

    This function is the central place to keep **image/keypoint consistency** correct
    when resizing or padding.

    Parameters
    ----------
    src_pil, tgt_pil:
        Source/target RGB images.
    src_kps_xy, tgt_kps_xy:
        ``(N, 2)`` keypoints in each image's **original** pixel space.
    mode:
        Preprocessing strategy (see :class:`PreprocessMode`).
    patch_size:
        ViT patch size used for rounding in patch-aware modes.
    fixed_size_hw:
        Required if ``mode == FIXED_RESIZE``: ``(H, W)`` target size.
    longest_side:
        Used by ``SCALE_LONGEST_ROUND`` as the target longest edge before rounding.
    letterbox_fill:
        RGB fill color for letterbox margins.

    Returns
    -------
    tuple[PIL.Image.Image, PIL.Image.Image, torch.Tensor, torch.Tensor, dict]
        Preprocessed images, transformed keypoints, and a metadata dict (scale/pad).
    """
    src_pil = src_pil.convert("RGB")
    tgt_pil = tgt_pil.convert("RGB")

    src_orig = src_pil.size  # (w, h)
    tgt_orig = tgt_pil.size

    meta: Dict[str, Any] = {"mode": str(mode), "patch_size": int(patch_size)}

    if mode == PreprocessMode.FIXED_RESIZE:
        if fixed_size_hw is None:
            raise ValueError("fixed_size_hw must be provided for FIXED_RESIZE mode.")
        out_h, out_w = int(fixed_size_hw[0]), int(fixed_size_hw[1])
        src_new = pil_resize_bicubic(src_pil, (out_w, out_h))
        tgt_new = pil_resize_bicubic(tgt_pil, (out_w, out_h))
        src_k = scale_keypoints_xy(src_kps_xy, src_orig, (out_w, out_h))
        tgt_k = scale_keypoints_xy(tgt_kps_xy, tgt_orig, (out_w, out_h))
        meta.update({"out_hw": (out_h, out_w)})
        return src_new, tgt_new, src_k, tgt_k, meta

    if mode == PreprocessMode.SCALE_LONGEST_ROUND:
        # Per-image scaling (src/tgt may differ); keypoints follow their image.
        def _scale_round_one(
            im: Image.Image, kps: torch.Tensor
        ) -> Tuple[Image.Image, torch.Tensor, Dict[str, Any]]:
            ow, oh = im.size
            long_side = max(oh, ow)
            if long_side <= 0:
                raise ValueError("Invalid image size.")
            s = float(longest_side) / float(long_side)
            nh = int(round(oh * s))
            nw = int(round(ow * s))
            nh2 = round_side_to_patch_multiple(nh, patch_size)
            nw2 = round_side_to_patch_multiple(nw, patch_size)
            im2 = pil_resize_bicubic(im, (nw2, nh2))
            k2 = scale_keypoints_xy(kps, (ow, oh), (nw2, nh2))
            return im2, k2, {"scale": s, "out_hw": (nh2, nw2), "orig_wh": (ow, oh)}

        src_new, src_k, src_meta = _scale_round_one(src_pil, src_kps_xy)
        tgt_new, tgt_k, tgt_meta = _scale_round_one(tgt_pil, tgt_kps_xy)
        meta.update({"src": src_meta, "tgt": tgt_meta})
        return src_new, tgt_new, src_k, tgt_k, meta

    if mode == PreprocessMode.LETTERBOX_PATCH_GRID:
        if fixed_size_hw is None:
            raise ValueError(
                "fixed_size_hw must provide target (H, W) canvas for LETTERBOX_PATCH_GRID."
            )
        out_h, out_w = int(fixed_size_hw[0]), int(fixed_size_hw[1])
        out_w = round_side_to_patch_multiple(out_w, patch_size)
        out_h = round_side_to_patch_multiple(out_h, patch_size)

        def _letterbox_one(
            im: Image.Image, kps: torch.Tensor, orig_wh: Tuple[int, int]
        ) -> Tuple[Image.Image, torch.Tensor, Dict[str, Any]]:
            scale, pad_l, pad_t, nw, nh = letterbox_affine(orig_wh, (out_w, out_h))
            im_resized = pil_resize_bicubic(im, (nw, nh))
            canvas = Image.new("RGB", (out_w, out_h), letterbox_fill)
            canvas.paste(im_resized, (pad_l, pad_t))
            k2 = map_keypoints_after_letterbox(
                scale_keypoints_xy(kps, orig_wh, (nw, nh)), scale=1.0, pad_left=pad_l, pad_top=pad_t
            )
            # Note: scale_keypoints_xy already maps orig -> (nw,nh); paste offset adds padding.
            return canvas, k2, {
                "scale": scale,
                "pad_left": pad_l,
                "pad_top": pad_t,
                "content_wh": (nw, nh),
                "canvas_wh": (out_w, out_h),
            }

        src_new, src_k, src_lb = _letterbox_one(src_pil, src_kps_xy, src_orig)
        tgt_new, tgt_k, tgt_lb = _letterbox_one(tgt_pil, tgt_kps_xy, tgt_orig)
        meta.update({"src": src_lb, "tgt": tgt_lb, "out_hw": (out_h, out_w)})
        return src_new, tgt_new, src_k, tgt_k, meta

    raise ValueError(f"Unsupported preprocess mode: {mode}")


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


def _pil_to_tensor_and_maybe_normalize(
    pil_img: Image.Image,
    *,
    to_tensor: transforms.ToTensor,
    normalize: Optional[transforms.Normalize],
) -> torch.Tensor:
    """Top-level helper so ``DataLoader(num_workers>0)`` can pickle the dataset on macOS/Windows."""
    t = to_tensor(pil_img)
    return normalize(t) if normalize is not None else t


def build_imagenet_normalize() -> transforms.Normalize:
    """
    Build ImageNet normalization used by many ViT checkpoints.

    Returns
    -------
    torchvision.transforms.Normalize
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD).
    """
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
) -> Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]:
    """
    Build a photometric augmentation that applies **identical** jitter to both images.

    Geometric augmentations are **not** included here. If you add geometric transforms,
    you must update keypoints consistently (see project constraints).

    The returned callable is safe to use inside dataset ``__getitem__`` when you want
    the same random color jitter on source/target (a common practice for paired data).

    Parameters
    ----------
    brightness, contrast, saturation, hue:
        Ranges forwarded to ``torchvision.transforms.functional`` adjustment ops.
        Hue is handled via ``adjust_hue`` (input factor in ``[-0.5, 0.5]`` domain).
    p:
        Probability of applying the sampled jitter (otherwise returns originals).
    seed:
        Optional RNG seed for reproducibility (per call, unless you re-seed externally).

    Returns
    -------
    Callable[[PIL.Image, PIL.Image], tuple[PIL.Image, PIL.Image]]
        ``(src_aug, tgt_aug)``.
    """

    def _sample_uniform(r: Tuple[float, float], rng: random.Random) -> float:
        return rng.uniform(float(r[0]), float(r[1]))

    def _apply_pair(src: Image.Image, tgt: Image.Image) -> Tuple[Image.Image, Image.Image]:
        rng = random.Random(seed) if seed is not None else random.Random()
        if rng.random() > float(p):
            return src, tgt

        # Sample parameters once, apply to both images.
        b = _sample_uniform(brightness, rng)
        c = _sample_uniform(contrast, rng)
        s = _sample_uniform(saturation, rng)
        h = _sample_uniform(hue, rng)

        def _one(im: Image.Image) -> Image.Image:
            ten = TF.pil_to_tensor(im).float() / 255.0
            ten = TF.adjust_brightness(ten, b)
            ten = TF.adjust_contrast(ten, c)
            ten = TF.adjust_saturation(ten, s)
            ten = TF.adjust_hue(ten, h)
            ten = torch.clamp(ten, 0.0, 1.0)
            return TF.to_pil_image(ten)

        return _one(src), _one(tgt)

    return _apply_pair


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class SPair71kPairDataset(Dataset):
    """
    PyTorch dataset for SPair-71k **image pairs** with keypoint correspondences.

    The dataset yields dictionaries containing tensors ready for training or
    evaluation, while preserving the information needed for PCK computation
    (bounding boxes and original sizes), consistent with common implementations.

    Notes
    -----
    - Pair annotation JSON is expected to include at least: ``src_kps``, ``trg_kps``,
      ``src_bndbox``, ``trg_bndbox``, ``category`` (as in the public SPair annotations).
    - Invalid/padded keypoints are filled with :data:`INVALID_KP_COORD` in the returned
      tensors; use ``n_valid_keypoints`` to mask losses/metrics.
    """

    def __init__(
        self,
        spair_root: Union[str, os.PathLike[str]],
        split: Union[str, SplitSpec],
        category: Union[str, Literal["all"]] = "all",
        preprocess: PreprocessMode = PreprocessMode.FIXED_RESIZE,
        output_size_hw: Optional[Tuple[int, int]] = (784, 784),
        patch_size: int = 14,
        longest_side: int = 784,
        letterbox_fill: Tuple[int, int, int] = (0, 0, 0),
        normalize: bool = True,
        photometric_augment: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        spair_root:
            Path to the extracted ``SPair-71k`` directory.
        split:
            Logical split: ``train`` (``trn.txt``), ``val``, or ``test``.
        category:
            Pascal category name, or ``all`` to include every category.
        preprocess:
            Geometric preprocessing policy (see :class:`PreprocessMode`).
        output_size_hw:
            ``(H, W)`` used by ``FIXED_RESIZE`` and ``LETTERBOX_PATCH_GRID`` modes.
            For patch-aware letterboxing, sides are rounded to multiples of ``patch_size``.
        patch_size:
            ViT patch size (used for rounding in patch-aware modes).
        longest_side:
            Target longest side before patch rounding in ``SCALE_LONGEST_ROUND``.
        letterbox_fill:
            RGB fill for letterbox margins.
        normalize:
            If ``True``, apply ImageNet normalization to ``src_img`` / ``tgt_img``.
        photometric_augment:
            Optional callable ``(src_pil, tgt_pil) -> (src_pil2, tgt_pil2)`` applied
            **before** resizing / tensor conversion (typically jitter).
        """
        super().__init__()
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
        self.longest_side = int(longest_side)
        self.letterbox_fill = letterbox_fill
        self.normalize = bool(normalize)
        self.photometric_augment = photometric_augment

        split_list_path = os.path.join(self.paths.layout_dir, spair_split_filename(self.split))
        self._pair_ids = read_split_pair_ids(split_list_path)
        if self.category_filter is not None:
            self._pair_ids = [pid for pid in self._pair_ids if self.category_filter in pid]

        # Category folders under JPEGImages (sorted for stable index mapping).
        self._categories = sorted(
            [d for d in os.listdir(self.paths.images_dir) if os.path.isdir(os.path.join(self.paths.images_dir, d))]
        )
        if len(self._categories) == 0:
            raise FileNotFoundError(f"No category folders found under: {self.paths.images_dir}")

        ann_split_dir = os.path.join(self.paths.pair_ann_dir, self.split_file_stem)
        if not os.path.isdir(ann_split_dir):
            raise FileNotFoundError(f"Pair annotation directory not found: {ann_split_dir}")

        self._records: List[Dict[str, Any]] = []
        for pair_id in self._pair_ids:
            ann_path = os.path.join(ann_split_dir, f"{pair_id}.json")
            anno = load_pair_annotation_json(ann_path)
            cat = str(anno["category"])
            if self.category_filter is not None and cat != self.category_filter:
                continue

            _, src_stem, tgt_stem, _cat_from_line = parse_spair_pair_line(pair_id)
            src_name = f"{src_stem}.jpg"
            tgt_name = f"{tgt_stem}.jpg"

            src_kps = torch.tensor(anno["src_kps"], dtype=torch.float32)
            tgt_kps = torch.tensor(anno["trg_kps"], dtype=torch.float32)
            if src_kps.ndim != 2 or tgt_kps.ndim != 2:
                raise ValueError(f"Unexpected keypoint array shape in {ann_path}")
            # Public annotations are typically (N, 2); keep internal as (N, 2) xy.
            if src_kps.shape[1] == 2 and src_kps.shape[0] != 2:
                src_kps_xy = src_kps
            elif src_kps.shape[0] == 2:
                src_kps_xy = _to_xy_keypoints(src_kps)
            else:
                raise ValueError(f"Cannot interpret src_kps shape {tuple(src_kps.shape)} in {ann_path}")

            if tgt_kps.shape[1] == 2 and tgt_kps.shape[0] != 2:
                tgt_kps_xy = tgt_kps
            elif tgt_kps.shape[0] == 2:
                tgt_kps_xy = _to_xy_keypoints(tgt_kps)
            else:
                raise ValueError(f"Cannot interpret trg_kps shape {tuple(tgt_kps.shape)} in {ann_path}")

            src_bbox = torch.tensor(anno["src_bndbox"], dtype=torch.float32).view(-1)
            tgt_bbox = torch.tensor(anno["trg_bndbox"], dtype=torch.float32).view(-1)
            if src_bbox.numel() != 4 or tgt_bbox.numel() != 4:
                raise ValueError(f"Invalid bounding box tensors in {ann_path}")

            src_path = os.path.join(self.paths.images_dir, cat, src_name)
            tgt_path = os.path.join(self.paths.images_dir, cat, tgt_name)
            if not os.path.isfile(src_path):
                raise FileNotFoundError(f"Missing source image: {src_path}")
            if not os.path.isfile(tgt_path):
                raise FileNotFoundError(f"Missing target image: {tgt_path}")

            self._records.append(
                {
                    "pair_id": pair_id,
                    "category": cat,
                    "src_path": src_path,
                    "tgt_path": tgt_path,
                    "src_kps_xy": src_kps_xy,
                    "tgt_kps_xy": tgt_kps_xy,
                    "src_bbox_xyxy": src_bbox,
                    "tgt_bbox_xyxy": tgt_bbox,
                    "viewpoint_variation": torch.tensor(anno.get("viewpoint_variation", 0)).float(),
                    "scale_variation": torch.tensor(anno.get("scale_variation", 0)).float(),
                    "truncation": torch.tensor(anno.get("truncation", 0)).float(),
                    "occlusion": torch.tensor(anno.get("occlusion", 0)).float(),
                }
            )

        to_tensor = transforms.ToTensor()
        norm = build_imagenet_normalize() if self.normalize else None

        self._to_model_tensor = functools.partial(
            _pil_to_tensor_and_maybe_normalize,
            to_tensor=to_tensor,
            normalize=norm,
        )

    def __len__(self) -> int:
        return len(self._records)

    def _map_bbox_xyxy(
        self,
        bbox_xyxy: torch.Tensor,
        orig_wh: Tuple[int, int],
        meta: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Map axis-aligned bounding boxes through the same preprocessing as images.

        This is a best-effort affine map for resize / letterbox (no independent aug).
        """
        ow, oh = orig_wh
        out = bbox_xyxy.clone().view(4)
        if self.preprocess == PreprocessMode.FIXED_RESIZE:
            if self.output_size_hw is None:
                raise ValueError("output_size_hw is required.")
            out_h, out_w = int(self.output_size_hw[0]), int(self.output_size_hw[1])
            out[0::2] *= out_w / ow
            out[1::2] *= out_h / oh
            return out

        if self.preprocess == PreprocessMode.SCALE_LONGEST_ROUND:
            if "out_hw" not in meta or "orig_wh" not in meta:
                raise KeyError("Expected 'out_hw' and 'orig_wh' in meta for SCALE_LONGEST_ROUND bbox mapping.")
            final_h, final_w = meta["out_hw"]
            orig_w, orig_h = meta["orig_wh"]
            out[0::2] *= float(final_w) / float(orig_w)
            out[1::2] *= float(final_h) / float(orig_h)
            return out

        if self.preprocess == PreprocessMode.LETTERBOX_PATCH_GRID:
            scale = float(meta["scale"])
            pad_left = int(meta["pad_left"])
            pad_top = int(meta["pad_top"])
            out[0::2] = out[0::2] * scale + float(pad_left)
            out[1::2] = out[1::2] * scale + float(pad_top)
            return out

        raise ValueError(f"Unsupported preprocess mode for bbox mapping: {self.preprocess}")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        rec = self._records[index]

        src_pil = Image.open(rec["src_path"]).convert("RGB")
        tgt_pil = Image.open(rec["tgt_path"]).convert("RGB")

        if self.photometric_augment is not None:
            src_pil, tgt_pil = self.photometric_augment(src_pil, tgt_pil)

        src_orig_wh = src_pil.size
        tgt_orig_wh = tgt_pil.size

        src_kps = rec["src_kps_xy"].clone()
        tgt_kps = rec["tgt_kps_xy"].clone()

        src_p2, tgt_p2, src_k2, tgt_k2, meta = preprocess_pair_images_and_keypoints(
            src_pil,
            tgt_pil,
            src_kps,
            tgt_kps,
            mode=self.preprocess,
            patch_size=self.patch_size,
            fixed_size_hw=self.output_size_hw,
            longest_side=self.longest_side,
            letterbox_fill=self.letterbox_fill,
        )

        if self.preprocess == PreprocessMode.FIXED_RESIZE:
            src_bbox = self._map_bbox_xyxy(rec["src_bbox_xyxy"], src_orig_wh, meta)
            tgt_bbox = self._map_bbox_xyxy(rec["tgt_bbox_xyxy"], tgt_orig_wh, meta)
        else:
            # Per-image affine metadata for src/tgt (shapes may differ in patch-rounded scaling).
            src_bbox = self._map_bbox_xyxy(rec["src_bbox_xyxy"], src_orig_wh, meta["src"])
            tgt_bbox = self._map_bbox_xyxy(rec["tgt_bbox_xyxy"], tgt_orig_wh, meta["tgt"])

        src_k_pad, n_src = pad_keypoints_to_max(src_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        tgt_k_pad, n_tgt = pad_keypoints_to_max(tgt_k2, MAX_KEYPOINTS, INVALID_KP_COORD)
        n_valid = int(min(n_src, n_tgt))

        # Bbox-based PCK tolerance (max side length) in **preprocessed target** coordinates.
        tw = tgt_bbox[2] - tgt_bbox[0]
        th = tgt_bbox[3] - tgt_bbox[1]
        pck_threshold_bbox = torch.max(tw, th).clamp_min(1e-6)

        sample: Dict[str, Any] = {
            "pair_id_str": rec["pair_id"],
            "category": rec["category"],
            "category_id": torch.tensor([self._categories.index(rec["category"])], dtype=torch.int64),
            "src_img": self._to_model_tensor(src_p2),
            "tgt_img": self._to_model_tensor(tgt_p2),
            "src_kps": src_k_pad,
            "tgt_kps": tgt_k_pad,
            "n_valid_keypoints": torch.tensor([n_valid], dtype=torch.int64),
            "src_bbox": src_bbox,
            "tgt_bbox": tgt_bbox,
            "pck_threshold_bbox": pck_threshold_bbox,
            "src_imsize": torch.tensor([src_orig_wh[0], src_orig_wh[1]], dtype=torch.float32),
            "tgt_imsize": torch.tensor([tgt_orig_wh[0], tgt_orig_wh[1]], dtype=torch.float32),
            "viewpoint_variation": rec["viewpoint_variation"].view(1),
            "scale_variation": rec["scale_variation"].view(1),
            "truncation": rec["truncation"].view(1),
            "occlusion": rec["occlusion"].view(1),
        }
        return sample


def spair_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate SPair samples into a batched dictionary.

    Tensor fields are stacked along batch dimension ``B``. The string identifier
    ``pair_id_str`` is returned as a ``list[str]`` of length ``B`` (PyTorch default
    collate does not support arbitrary strings in tensor batches).

    Parameters
    ----------
    batch:
        List of samples from :class:`SPair71kPairDataset`.

    Returns
    -------
    dict[str, Any]
        Batched tensors plus ``pair_id_str: list[str]``.
    """
    if len(batch) == 0:
        raise ValueError("Empty batch.")
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        if k in ("pair_id_str", "category"):
            out[k] = [b[k] for b in batch]
            continue
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals, dim=0)
    return out
