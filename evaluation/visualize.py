"""
Keypoint correspondence visualization utilities.

Draw predicted and ground-truth keypoints on source/target image pairs,
color-coded by correctness (PCK threshold). Designed for inline display
in Colab / Jupyter notebooks.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment,misc]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_numpy_hwc(
    img: Union[torch.Tensor, np.ndarray],
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std: Tuple[float, float, float] = _IMAGENET_STD,
) -> np.ndarray:
    """Convert a CHW tensor (ImageNet-normalized) to HWC uint8 numpy array."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().float().numpy()
    else:
        arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    m = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    s = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    arr = arr * s + m
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def visualize_correspondences(
    src_img: Union[torch.Tensor, np.ndarray],
    tgt_img: Union[torch.Tensor, np.ndarray],
    src_kps: Union[torch.Tensor, np.ndarray],
    pred_tgt_kps: Union[torch.Tensor, np.ndarray],
    gt_tgt_kps: Union[torch.Tensor, np.ndarray],
    *,
    pck_threshold: float = 1.0,
    alpha: float = 0.1,
    invalid_value: float = -2.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5),
    marker_size: float = 50.0,
    line_alpha: float = 0.4,
    mean: Tuple[float, float, float] = _IMAGENET_MEAN,
    std: Tuple[float, float, float] = _IMAGENET_STD,
) -> "Figure":
    """
    Visualize predicted correspondences between a source and target image.

    Keypoints are drawn as colored dots; correct predictions (within
    ``alpha * pck_threshold``) are green, incorrect are red. Lines connect
    source keypoints to their predicted target locations.

    Parameters
    ----------
    src_img, tgt_img:
        CHW tensors (ImageNet-normalized) or HWC numpy arrays.
    src_kps:
        Source keypoints, shape ``(N, 2)`` in ``(x, y)`` pixel coords.
    pred_tgt_kps:
        Predicted target keypoints, shape ``(N, 2)``.
    gt_tgt_kps:
        Ground-truth target keypoints, shape ``(N, 2)``.
    pck_threshold:
        Bounding-box scale (max side length) for PCK normalization.
    alpha:
        PCK alpha threshold.
    invalid_value:
        Coordinate value indicating an invalid/padded keypoint.
    title:
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualize_correspondences().")

    def _np(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
        return np.asarray(t, dtype=np.float32)

    src_np = _to_numpy_hwc(src_img, mean, std)
    tgt_np = _to_numpy_hwc(tgt_img, mean, std)
    src_pts = _np(src_kps)
    pred_pts = _np(pred_tgt_kps)
    gt_pts = _np(gt_tgt_kps)

    valid = (gt_pts[:, 0] > invalid_value) & (gt_pts[:, 1] > invalid_value)
    valid &= (src_pts[:, 0] > invalid_value) & (src_pts[:, 1] > invalid_value)

    dist = np.linalg.norm(pred_pts - gt_pts, axis=-1)
    correct = dist <= alpha * pck_threshold

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(src_np)
    axes[0].set_title("Source")
    axes[0].axis("off")
    axes[1].imshow(tgt_np)
    axes[1].set_title("Target")
    axes[1].axis("off")

    for i in range(len(src_pts)):
        if not valid[i]:
            continue
        color = "tab:green" if correct[i] else "tab:red"
        axes[0].scatter(src_pts[i, 0], src_pts[i, 1], s=marker_size, c=color, edgecolors="white", linewidths=0.5, zorder=5)
        axes[1].scatter(pred_pts[i, 0], pred_pts[i, 1], s=marker_size, c=color, edgecolors="white", linewidths=0.5, zorder=5, marker="x")
        axes[1].scatter(gt_pts[i, 0], gt_pts[i, 1], s=marker_size * 0.5, c="blue", edgecolors="white", linewidths=0.3, zorder=4, marker="o", alpha=0.6)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
