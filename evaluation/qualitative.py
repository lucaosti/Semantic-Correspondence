"""
Paper-grade qualitative analysis utilities.

Loads trained checkpoints, runs inference on selected SPair-71k pairs and produces
method-comparison grids, per-keypoint similarity-heatmap overlays, a failure-case
selector ranked by per-image PCK, and a symmetry-ambiguity heuristic. Pure-Python
helpers (:func:`find_failure_cases`, :func:`find_symmetry_ambiguity`) are testable
without torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Method spec & checkpoint resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """Describes a (backbone, method, optional WSA) combination to render qualitatively."""

    backbone: str  # dinov2_vitb14, dinov3_vitb16, sam_vit_b
    method: str    # baseline, ft_lb1, ft_lb2, ft_lb4, lora
    wsa: bool = False

    @property
    def label(self) -> str:
        suffix = " + WSA" if self.wsa else ""
        return f"{self.backbone} / {self.method}{suffix}"


def resolve_checkpoint_path(spec: MethodSpec, ckpt_dir: Path, *, lora_rank: int = 8) -> Optional[Path]:
    """Map a :class:`MethodSpec` to an on-disk checkpoint file (``None`` for baseline)."""
    ckpt_dir = Path(ckpt_dir)
    if spec.method == "baseline":
        return None
    if spec.method == "lora":
        path = ckpt_dir / f"{spec.backbone}_lora_r{lora_rank}_best.pt"
    elif spec.method.startswith("ft_lb"):
        n = spec.method.replace("ft_lb", "")
        path = ckpt_dir / f"{spec.backbone}_lastblocks{n}_best.pt"
    else:
        return None
    return path if path.is_file() else None


# ---------------------------------------------------------------------------
# Pure-Python helpers (testable without torch)
# ---------------------------------------------------------------------------


def find_failure_cases(
    pck_results: Sequence[Dict[str, Any]],
    *,
    run_name: str,
    alpha: float = 0.1,
    k: int = 5,
) -> List[Tuple[int, float]]:
    """Return the indices of the K lowest-PCK pairs for ``run_name``.

    Reads the per-image PCK list from ``r["sd4match_per_image"]["custom_pck<alpha>"]["all"]``.

    Returns a list of ``(pair_index, pck_value)`` sorted ascending. Indices match the
    DataLoader iteration order from :func:`evaluation.experiment_runner.run_spair_pck_eval`.
    """
    target = next((r for r in pck_results if r.get("name") == run_name), None)
    if target is None:
        return []
    per_image = target.get("sd4match_per_image") or {}
    metric_key = f"custom_pck{alpha}"
    bucket = (per_image.get(metric_key) or {}).get("all") or []
    if not bucket:
        return []
    indexed = list(enumerate(float(v) for v in bucket))
    indexed.sort(key=lambda t: t[1])
    return indexed[: max(1, int(k))]


def find_symmetry_ambiguity(
    *,
    src_kps_named: Sequence[Tuple[str, float, float]],
    pred_kps_xy: Sequence[Tuple[float, float]],
    gt_kps_xy: Sequence[Tuple[float, float]],
    pck_threshold: float,
    alpha: float = 0.1,
) -> List[int]:
    """Heuristic: which left/right keypoints failed PCK *and* land closer to the symmetric peer?

    SPair-71k keypoint annotations sometimes carry semantic names (e.g. ``left_eye`` vs
    ``right_eye``); this helper flags the canonical "left/right confusion" failure mode by
    checking whether the prediction is closer to the *other* labelled side than to its own
    GT. Returns a list of ambiguous keypoint indices (subset of failures).
    """
    n = len(gt_kps_xy)
    if n == 0:
        return []
    radius = float(alpha) * float(pck_threshold)
    ambiguous: List[int] = []
    for i, (name_i, _sx, _sy) in enumerate(src_kps_named):
        if i >= n:
            break
        gt_x, gt_y = gt_kps_xy[i]
        pr_x, pr_y = pred_kps_xy[i]
        if math.hypot(pr_x - gt_x, pr_y - gt_y) <= radius:
            continue
        lname = name_i.lower()
        side: Optional[str] = None
        if "left" in lname:
            side = "left"
        elif "right" in lname:
            side = "right"
        if side is None:
            continue
        peer_token = "right" if side == "left" else "left"
        for j, (name_j, _, _) in enumerate(src_kps_named):
            if j == i:
                continue
            if peer_token in name_j.lower():
                gt_jx, gt_jy = gt_kps_xy[j] if j < n else (gt_x, gt_y)
                if math.hypot(pr_x - gt_jx, pr_y - gt_jy) < math.hypot(pr_x - gt_x, pr_y - gt_y):
                    ambiguous.append(i)
                    break
    return ambiguous


# ---------------------------------------------------------------------------
# Inference (torch-dependent)
# ---------------------------------------------------------------------------


def load_method_extractors(
    specs: Sequence[MethodSpec],
    *,
    ckpt_dir: Path,
    pretrained_paths: Dict[str, Optional[str]],
    device: "Any",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_last_blocks: int = 2,
) -> Dict[str, Any]:
    """Build one :class:`DenseFeatureExtractor` per :class:`MethodSpec`.

    ``pretrained_paths`` maps backbone names to the **pretrained** ``.pth`` (used by every
    method, including baseline). For ``lora`` and ``ft_lb*`` the trained weights override
    the relevant subset on top of the pretrained backbone.

    Returns a dict ``{spec.label: extractor}``. Missing checkpoints are skipped with a
    printed warning so analysis can proceed with whatever is available.
    """

    from evaluation.checkpoint_loader import load_encoder_weights_from_pt
    from models.common import (
        BackboneName,
        DenseExtractorConfig,
        DenseFeatureExtractor,
    )
    from models.common.lora import (
        apply_lora_to_last_blocks_mlp,
        apply_lora_to_last_blocks_mlp_sam,
    )

    out: Dict[str, Any] = {}
    for spec in specs:
        try:
            cfg = DenseExtractorConfig(
                name=BackboneName(spec.backbone),
                dinov2_weights_path=pretrained_paths.get("dinov2_vitb14")
                if spec.backbone == "dinov2_vitb14" else None,
                dinov3_weights_path=pretrained_paths.get("dinov3_vitb16")
                if spec.backbone == "dinov3_vitb16" else None,
                sam_checkpoint_path=pretrained_paths.get("sam_vit_b")
                if spec.backbone == "sam_vit_b" else None,
            )
            extractor = DenseFeatureExtractor(cfg, freeze=True)
            if spec.method == "lora":
                # Re-apply LoRA wrappers so the saved adapter weights have a target to load into
                if spec.backbone == "sam_vit_b":
                    apply_lora_to_last_blocks_mlp_sam(
                        extractor.encoder, last_n_blocks=lora_last_blocks,
                        rank=lora_rank, alpha=lora_alpha,
                    )
                else:
                    apply_lora_to_last_blocks_mlp(
                        extractor.encoder, last_n_blocks=lora_last_blocks,
                        rank=lora_rank, alpha=lora_alpha,
                    )
            ckpt_path = resolve_checkpoint_path(spec, ckpt_dir, lora_rank=lora_rank)
            if spec.method != "baseline":
                if ckpt_path is None:
                    print(f"[qualitative] WARN: missing checkpoint for {spec.label}; skipping.")
                    continue
                load_encoder_weights_from_pt(extractor, str(ckpt_path), map_location="cpu")
            extractor.to(device).eval()
            out[spec.label] = extractor
        except Exception as exc:  # noqa: BLE001
            print(f"[qualitative] WARN: failed to build {spec.label}: {exc}")
    return out


def predict_pair(
    extractor: "Any",
    sample: Dict[str, Any],
    *,
    use_wsa: bool,
    img_hw: Tuple[int, int],
    wsa_window: int = 5,
    wsa_temperature: float = 1.0,
) -> Dict[str, Any]:
    """Run cosine-argmax (and optional WSA refinement) on a single SPair pair sample.

    ``sample`` is one item from :class:`SPair71kPairDataset`: keys ``src_img``, ``tgt_img``,
    ``src_kps`` (``[K, 2]`` xy, possibly padded), ``tgt_kps``, ``n_valid_keypoints``,
    ``pck_threshold_bbox``.

    Returns a dict ``{pred_tgt_xy, sim_maps, valid_mask}``.
    """
    import torch

    from models.common.matching import predict_correspondences_cosine_argmax
    from models.common.window_soft_argmax import refine_predictions_window_soft_argmax

    device = next(extractor.parameters()).device
    src = sample["src_img"].unsqueeze(0).to(device)
    tgt = sample["tgt_img"].unsqueeze(0).to(device)
    src_kps = sample["src_kps"].to(device)
    n_valid = int(sample["n_valid_keypoints"].view(-1)[0].item())
    valid_mask = torch.zeros(src_kps.shape[0], dtype=torch.bool, device=device)
    valid_mask[:n_valid] = True

    with torch.no_grad():
        feat_src, meta_src = extractor(src)
        feat_tgt, meta_tgt = extractor(tgt)
        coord_src = meta_src.get("coord_hw", img_hw)
        coord_tgt = meta_tgt.get("coord_hw", img_hw)
        out = predict_correspondences_cosine_argmax(
            feat_src, feat_tgt, src_kps,
            img_hw=coord_src, img_hw_src=coord_src, img_hw_tgt=coord_tgt,
            valid_mask=valid_mask,
        )
        pred_tgt = out["pred_tgt_xy"]
        sim_maps = out["sim_maps"]
        if use_wsa:
            pred_tgt = refine_predictions_window_soft_argmax(
                sim_maps, coord_tgt, window_size=wsa_window, temperature=wsa_temperature,
            )
        # Rescale prediction back to dataset image frame
        if coord_tgt != img_hw:
            from models.common.coord_utils import rescale_keypoints_xy
            pred_tgt = rescale_keypoints_xy(pred_tgt, coord_tgt, img_hw)
    return {
        "pred_tgt_xy": pred_tgt.detach().cpu(),
        "sim_maps": sim_maps.detach().cpu(),
        "valid_mask": valid_mask.detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_method_comparison_grid(
    sample: Dict[str, Any],
    extractors: Dict[str, Any],
    *,
    img_hw: Tuple[int, int],
    use_wsa_per_label: Optional[Dict[str, bool]] = None,
    wsa_window: int = 5,
    wsa_temperature: float = 1.0,
    alpha: float = 0.1,
    figsize_per_axis: Tuple[float, float] = (3.5, 3.5),
):
    """Render a (rows = method, cols = source/target) grid for one SPair pair.

    Each row shows: source with GT source keypoints, target with predicted (colored by
    PCK correctness) and GT target keypoints (small blue dots). Returns the matplotlib
    figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from evaluation.figures import apply_paper_style
    from evaluation.visualize import _to_numpy_hwc

    apply_paper_style()
    use_wsa_per_label = use_wsa_per_label or {}
    pck_thr = float(sample["pck_threshold_bbox"])
    n = max(1, len(extractors))
    fig, axes = plt.subplots(n, 2, figsize=(figsize_per_axis[0] * 2, figsize_per_axis[1] * n),
                             squeeze=False)
    src_np = _to_numpy_hwc(sample["src_img"])
    tgt_np = _to_numpy_hwc(sample["tgt_img"])
    src_kps = sample["src_kps"].cpu().numpy()
    gt_kps = sample["tgt_kps"].cpu().numpy()

    for row, (label, extractor) in enumerate(extractors.items()):
        wsa = use_wsa_per_label.get(label, False)
        result = predict_pair(
            extractor, sample, use_wsa=wsa, img_hw=img_hw,
            wsa_window=wsa_window, wsa_temperature=wsa_temperature,
        )
        pred = result["pred_tgt_xy"].numpy()
        valid = result["valid_mask"].numpy()
        dist = np.linalg.norm(pred - gt_kps, axis=-1)
        correct = (dist <= alpha * pck_thr) & valid

        ax_s, ax_t = axes[row, 0], axes[row, 1]
        ax_s.imshow(src_np)
        ax_s.axis("off")
        ax_t.imshow(tgt_np)
        ax_t.axis("off")
        ax_s.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=80)

        for i in range(len(src_kps)):
            if not valid[i]:
                continue
            color = "tab:green" if correct[i] else "tab:red"
            ax_s.scatter(src_kps[i, 0], src_kps[i, 1], s=40, c=color,
                         edgecolors="white", linewidths=0.5, zorder=5)
            ax_t.scatter(pred[i, 0], pred[i, 1], s=40, c=color, marker="x",
                         edgecolors="white", linewidths=0.5, zorder=5)
            ax_t.scatter(gt_kps[i, 0], gt_kps[i, 1], s=18, c="blue",
                         edgecolors="white", linewidths=0.3, zorder=4, alpha=0.7)
        if row == 0:
            ax_s.set_title("Source", fontsize=10)
            ax_t.set_title("Target (X = pred, o = GT)", fontsize=10)
        # Per-row PCK label
        n_valid = int(valid.sum())
        n_correct = int(correct.sum())
        ax_t.text(0.02, 0.98,
                  f"PCK@{alpha:g}: {n_correct}/{n_valid}",
                  transform=ax_t.transAxes, fontsize=8, va="top", color="white",
                  bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    fig.suptitle(
        f"Pair {sample.get('pair_id_str', '?')} (cat: {sample.get('category', '?')})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def render_similarity_heatmap_overlay(
    sample: Dict[str, Any],
    extractor: "Any",
    *,
    img_hw: Tuple[int, int],
    keypoint_indices: Sequence[int],
    use_wsa: bool = False,
    wsa_window: int = 5,
    wsa_temperature: float = 1.0,
    figsize_per_axis: Tuple[float, float] = (4.0, 4.0),
):
    """Overlay similarity heatmaps on the target image for selected source keypoints.

    Each column shows: source with the picked keypoint highlighted, then target with the
    similarity surface overlaid (perceptually-uniform ``viridis`` colormap, transparency =
    saliency). Useful to diagnose attention spread vs. peakedness.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F

    from evaluation.figures import apply_paper_style
    from evaluation.visualize import _to_numpy_hwc

    apply_paper_style()
    result = predict_pair(extractor, sample, use_wsa=use_wsa, img_hw=img_hw,
                          wsa_window=wsa_window, wsa_temperature=wsa_temperature)
    sim_maps = result["sim_maps"]  # (N, Hf, Wf)
    pred = result["pred_tgt_xy"].numpy()
    valid = result["valid_mask"].numpy()

    src_np = _to_numpy_hwc(sample["src_img"])
    tgt_np = _to_numpy_hwc(sample["tgt_img"])
    src_kps = sample["src_kps"].cpu().numpy()
    gt_kps = sample["tgt_kps"].cpu().numpy()

    keypoint_indices = [i for i in keypoint_indices if 0 <= i < sim_maps.shape[0] and bool(valid[i])]
    if not keypoint_indices:
        fig, ax = plt.subplots(figsize=figsize_per_axis)
        ax.text(0.5, 0.5, "No valid keypoints", ha="center", va="center", transform=ax.transAxes)
        return fig

    n = len(keypoint_indices)
    fig, axes = plt.subplots(n, 2, figsize=(figsize_per_axis[0] * 2, figsize_per_axis[1] * n),
                             squeeze=False)
    h_img, w_img = img_hw
    for row, kp_idx in enumerate(keypoint_indices):
        ax_s = axes[row, 0]
        ax_t = axes[row, 1]
        ax_s.imshow(src_np)
        ax_s.axis("off")
        ax_t.imshow(tgt_np)
        ax_t.axis("off")
        ax_s.scatter(src_kps[kp_idx, 0], src_kps[kp_idx, 1], s=80, c="yellow",
                     edgecolors="black", linewidths=1, zorder=6, marker="o")
        # Resize sim_map to image size
        sm = sim_maps[kp_idx].unsqueeze(0).unsqueeze(0).float()
        sm_resized = F.interpolate(sm, size=(h_img, w_img), mode="bilinear",
                                   align_corners=False)[0, 0].numpy()
        # Normalize to [0, 1] for colormap clarity
        v_min, v_max = float(sm_resized.min()), float(sm_resized.max())
        if v_max > v_min:
            sm_norm = (sm_resized - v_min) / (v_max - v_min)
        else:
            sm_norm = np.zeros_like(sm_resized)
        ax_t.imshow(sm_norm, cmap="viridis", alpha=0.55, vmin=0, vmax=1)
        ax_t.scatter(pred[kp_idx, 0], pred[kp_idx, 1], s=70, c="white", marker="x",
                     linewidths=2, zorder=6)
        ax_t.scatter(gt_kps[kp_idx, 0], gt_kps[kp_idx, 1], s=70, c="blue", marker="o",
                     edgecolors="white", linewidths=1, zorder=5)
        ax_s.set_title(f"keypoint {kp_idx}", fontsize=9)
        ax_t.set_title("similarity heatmap (X = pred, o = GT)", fontsize=9)
    fig.suptitle(
        f"Similarity overlay - pair {sample.get('pair_id_str', '?')} (cat: {sample.get('category', '?')})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


__all__ = [
    "MethodSpec",
    "find_failure_cases",
    "find_symmetry_ambiguity",
    "load_method_extractors",
    "predict_pair",
    "render_method_comparison_grid",
    "render_similarity_heatmap_overlay",
    "resolve_checkpoint_path",
]
