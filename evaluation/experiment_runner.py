"""PCK evaluation runner supporting SPair-71k, PF-Willow, and PF-Pascal.

Use :class:`EvalRunSpec` to describe a single run; :func:`run_spair_pck_eval` returns
the metric dictionaries used for logging, JSON exports and comparison tables.

Runs in ``torch.inference_mode``; only the backbone forward uses bf16 autocast (CUDA
Ampere+), matching and PCK run in fp32. Forward over ``src`` and ``tgt`` is fused into a
single ``extractor(cat([src, tgt]))`` and the matcher is fully batched.

PCK follows the standard protocol: predictions are mapped from the preprocessed frame
back to the **original** target image, compared with the original annotations, and a
keypoint is correct when its error is ``<= alpha * pck_threshold_bbox`` (the dataset's
normalisation length in that frame). ``pck@a`` is the mean of per-image PCK,
``pck_pt@a`` the mean over all keypoints.

Dataset selection is controlled by :attr:`EvalRunSpec.dataset`:

* ``"spair"`` (default) — SPair-71k; uses :func:`data.paths.resolve_spair_root`.
* ``"pf_willow"`` — PF-Willow; uses :func:`data.paths.resolve_pf_willow_root`.
* ``"pf_pascal"`` — PF-Pascal; uses :func:`data.paths.resolve_pf_pascal_root`.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from torch.utils.data import Dataset as _Dataset

from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn
from data.paths import resolve_pf_pascal_root, resolve_pf_willow_root, resolve_spair_root
from evaluation.checkpoint_loader import load_encoder_weights_from_pt
from models.common.coord_utils import rescale_coords
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
from models.common.matching import argmax_to_pixel_xy, sample_descriptors, similarity_maps
from models.common.window_soft_argmax import refine_predictions_window_soft_argmax
from utils.hardware import dataloader_extra_kwargs, pin_memory_for, recommended_device_str

_DIFFICULTY_FLAGS: Tuple[str, ...] = (
    "viewpoint_variation",
    "scale_variation",
    "truncation",
    "occlusion",
)


@dataclass
class EvalRunSpec:
    """One evaluation configuration. Report final numbers on ``test``; use ``val`` for selection."""

    name: str
    backbone: str
    split: str = "val"
    dinov2_weights: Optional[str] = None
    dinov3_weights: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    checkpoint: Optional[str] = None
    use_window_soft_argmax: bool = False
    wsa_window: int = 5
    wsa_temperature: float = 0.05
    preprocess: str = "FIXED_RESIZE"
    height: int = 784
    width: int = 784
    limit: int = 0
    num_workers: int = 1
    batch_size: int = 8
    dino_layer: int = -1
    # Dataset selection: "spair" | "pf_willow" | "pf_pascal"
    dataset: str = "spair"
    # Explicit root for PF-Willow / PF-Pascal (None = auto-resolve via env / repo default)
    dataset_root: Optional[str] = None

    def to_dense_config(self) -> DenseExtractorConfig:
        bname = self.backbone
        if bname.startswith("dinov2"):
            wp = self.dinov2_weights
        elif bname.startswith("dinov3"):
            wp = self.dinov3_weights
        else:
            wp = self.sam_checkpoint
        return DenseExtractorConfig(
            name=BackboneName(bname),
            weights_path=wp,
            dino_layer=self.dino_layer,
        )


def _build_eval_dataset(spec: EvalRunSpec, spair_root: str) -> _Dataset:
    """Instantiate the dataset specified by ``spec.dataset``."""
    mode = PreprocessMode[str(spec.preprocess).strip().upper()]
    size_hw = (int(spec.height), int(spec.width))

    if spec.dataset == "pf_willow":
        from data.pf_dataset import PFWillowPairDataset

        root = resolve_pf_willow_root(spec.dataset_root)
        return PFWillowPairDataset(
            root,
            split=spec.split,
            preprocess=mode,
            output_size_hw=size_hw,
            normalize=True,
            photometric_augment=None,
        )

    if spec.dataset == "pf_pascal":
        from data.pf_dataset import PFPascalPairDataset

        root = resolve_pf_pascal_root(spec.dataset_root)
        return PFPascalPairDataset(
            root,
            split=spec.split,
            preprocess=mode,
            output_size_hw=size_hw,
            normalize=True,
            photometric_augment=None,
        )

    # Default: SPair-71k
    return SPair71kPairDataset(
        spair_root=spair_root,
        split=spec.split,
        preprocess=mode,
        output_size_hw=size_hw,
        normalize=True,
        photometric_augment=None,
    )


def _build_eval_loader(
    dataset: _Dataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    limit: int,
) -> DataLoader:
    if limit and limit > 0:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    extra = dataloader_extra_kwargs(num_workers, for_device="cuda" if pin_memory else "cpu")
    return DataLoader(
        dataset,
        batch_size=int(max(1, batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=spair_collate_fn,
        pin_memory=pin_memory,
        **extra,
    )


def _bucket01(v: Any) -> int:
    if isinstance(v, torch.Tensor):
        return 1 if int(v.reshape(-1)[0].item()) != 0 else 0
    return 1 if int(v) != 0 else 0


def _flag_at(batch: Dict[str, Any], name: str, b: int) -> Optional[int]:
    if name not in batch:
        return None
    val = batch[name]
    if isinstance(val, torch.Tensor):
        return _bucket01(val.view(-1)[b])
    return _bucket01(val[b])


def _autocast_eval(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _predict_batch(
    feats_src: torch.Tensor,
    feats_tgt: torch.Tensor,
    src_kps: torch.Tensor,
    img_hw: Tuple[int, int],
    spec: EvalRunSpec,
) -> torch.Tensor:
    """Return ``(B, N, 2)`` predicted target points in the preprocessed (dataset) frame."""
    desc = sample_descriptors(feats_src, src_kps, img_hw)
    sims = similarity_maps(desc, feats_tgt)  # (B, N, Hf, Wf), fp32
    if not spec.use_window_soft_argmax:
        return argmax_to_pixel_xy(sims, img_hw)
    bsz, n_pts, hf, wf = sims.shape
    return refine_predictions_window_soft_argmax(
        sims.reshape(bsz * n_pts, hf, wf),
        img_hw,
        window_size=spec.wsa_window,
        temperature=spec.wsa_temperature,
    ).reshape(bsz, n_pts, 2)


def _to_original_frame(pred: torch.Tensor, img_hw: Tuple[int, int], orig_wh: torch.Tensor) -> torch.Tensor:
    """Map ``(B, N, 2)`` points from the preprocessed frame to each image's original frame."""
    w0 = orig_wh[:, 0].view(-1, 1)
    h0 = orig_wh[:, 1].view(-1, 1)
    return torch.stack(
        [rescale_coords(pred[..., 0], float(img_hw[1]), w0), rescale_coords(pred[..., 1], float(img_hw[0]), h0)],
        dim=-1,
    )


def run_spair_pck_eval(
    spec: EvalRunSpec,
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run PCK for one configuration.

    Returns a dict with ``metrics`` (``pck@α`` macro per-image and ``pck_pt@α`` micro
    per-point), ``spec``, ``sd4match_per_image`` / ``per_point`` / ``summary`` and a
    summary-only ``sd4match_by_difficulty_flag``.
    """
    if spec.dataset == "pf_willow":
        root = resolve_pf_willow_root(spec.dataset_root or spair_root)
    elif spec.dataset == "pf_pascal":
        root = resolve_pf_pascal_root(spec.dataset_root or spair_root)
    else:
        root = resolve_spair_root(spair_root or spec.dataset_root)
    eval_device = device if device is not None else torch.device(recommended_device_str())
    pin_mem = pin_memory_for(eval_device)
    alphas_t = torch.tensor([float(a) for a in alphas], device=eval_device)

    extractor = DenseFeatureExtractor(spec.to_dense_config(), freeze=True)
    load_info: Optional[Dict[str, int]] = None
    if spec.checkpoint:
        load_info = load_encoder_weights_from_pt(extractor, spec.checkpoint, map_location="cpu")
    extractor.eval().to(eval_device)

    dataset = _build_eval_dataset(spec, root)
    loader = _build_eval_loader(
        dataset,
        batch_size=spec.batch_size,
        num_workers=spec.num_workers,
        pin_memory=pin_mem,
        limit=spec.limit,
    )

    # Per-image PCK rates (0..1) per α / category; per-point booleans per α / category.
    image_acc: Dict[str, Dict[str, List[float]]] = {f"custom_pck{a}": {"all": []} for a in alphas}
    point_acc: Dict[str, Dict[str, List[float]]] = {f"custom_pck{a}": {"all": []} for a in alphas}
    diff_image: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        f: {0: {f"custom_pck{a}": [] for a in alphas},
            1: {f"custom_pck{a}": [] for a in alphas}}
        for f in _DIFFICULTY_FLAGS
    }
    diff_point: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        f: {0: {f"custom_pck{a}": [] for a in alphas},
            1: {f"custom_pck{a}": [] for a in alphas}}
        for f in _DIFFICULTY_FLAGS
    }

    _WARMUP_BATCHES = 3
    _TIMING_BATCHES = 10
    if eval_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(eval_device)
    _batch_idx = 0
    _t0: Optional[float] = None
    _t1: Optional[float] = None
    _timing_images = 0

    with torch.inference_mode():
        for batch in loader:
            if _batch_idx == _WARMUP_BATCHES:
                if eval_device.type == "cuda":
                    torch.cuda.synchronize(eval_device)
                _t0 = time.perf_counter()
            src = batch["src_img"].to(eval_device, non_blocking=pin_mem)
            tgt = batch["tgt_img"].to(eval_device, non_blocking=pin_mem)
            src_kps = batch["src_kps"].to(eval_device)
            tgt_kps_orig = batch["tgt_kps_orig"].to(eval_device)
            tgt_wh = batch["tgt_imsize"].to(eval_device)
            n_valid = batch["n_valid_keypoints"].to(eval_device).reshape(-1)
            pckthres = batch["pck_threshold_bbox"].to(eval_device).reshape(-1)
            categories = batch["category"]
            bsz = int(src.shape[0])
            img_hw = (int(src.shape[2]), int(src.shape[3]))

            with _autocast_eval(eval_device):
                feats, _meta = extractor(torch.cat([src, tgt], dim=0))
            feats = feats.float()
            pred = _predict_batch(feats[:bsz], feats[bsz:], src_kps, img_hw, spec)
            pred_orig = _to_original_frame(pred, img_hw, tgt_wh)
            dists = torch.norm(tgt_kps_orig - pred_orig, dim=-1)  # (B, N), original pixels
            # (n_alpha, B, N) bool
            correct_all = (dists.unsqueeze(0) <= alphas_t.view(-1, 1, 1) * pckthres.view(1, -1, 1)).cpu()
            n_valid_list = n_valid.cpu().tolist()

            if _t0 is not None and _batch_idx < _WARMUP_BATCHES + _TIMING_BATCHES:
                _timing_images += bsz
                if _batch_idx == _WARMUP_BATCHES + _TIMING_BATCHES - 1:
                    if eval_device.type == "cuda":
                        torch.cuda.synchronize(eval_device)
                    _t1 = time.perf_counter()
            _batch_idx += 1

            diff_buckets = [
                [(_flag_at(batch, f, b)) for f in _DIFFICULTY_FLAGS] for b in range(bsz)
            ]
            for b in range(bsz):
                n = int(n_valid_list[b])
                if n == 0:
                    continue
                cat = categories[b]
                for ai, a in enumerate(alphas):
                    cb = correct_all[ai, b, :n].tolist()
                    img_pck = float(sum(cb)) / n
                    key = f"custom_pck{a}"
                    image_acc[key]["all"].append(img_pck)
                    image_acc[key].setdefault(cat, []).append(img_pck)
                    point_acc[key]["all"].extend(float(p) for p in cb)
                    point_acc[key].setdefault(cat, []).extend(float(p) for p in cb)
                    for fi, flag in enumerate(_DIFFICULTY_FLAGS):
                        bucket = diff_buckets[b][fi]
                        if bucket is None:
                            continue
                        diff_image[flag][bucket][key].append(img_pck)
                        diff_point[flag][bucket][key].extend(float(p) for p in cb)

    if eval_device.type == "cuda":
        torch.cuda.synchronize(eval_device)
    _throughput = float("nan")
    if _t0 is not None and _timing_images > 0:
        _end = _t1 if _t1 is not None else time.perf_counter()
        _elapsed = _end - _t0
        if _elapsed > 0:
            _throughput = _timing_images / _elapsed
    _peak_mb = float("nan")
    if eval_device.type == "cuda":
        _peak_mb = torch.cuda.max_memory_allocated(eval_device) / 1e6

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    summary_image = {k: {"all": _mean(image_acc[k]["all"])} for k in image_acc}
    summary_point = {k: {"all": _mean(point_acc[k]["all"])} for k in point_acc}
    metrics: Dict[str, float] = {}
    for a in alphas:
        metrics[f"pck@{a:g}"] = float(summary_image[f"custom_pck{a}"]["all"])
        metrics[f"pck_pt@{a:g}"] = float(summary_point[f"custom_pck{a}"]["all"])

    by_difficulty: Dict[str, Any] = {}
    for flag in _DIFFICULTY_FLAGS:
        by_difficulty[flag] = {}
        for bucket in (0, 1):
            by_difficulty[flag][str(bucket)] = {
                "summary": {
                    "image": {k: {"all": _mean(diff_image[flag][bucket][k])}
                              for k in diff_image[flag][bucket]},
                    "point": {k: {"all": _mean(diff_point[flag][bucket][k])}
                              for k in diff_point[flag][bucket]},
                }
            }

    out: Dict[str, Any] = {
        "name": spec.name,
        "dataset": spec.dataset,
        "dataset_root": root,
        # Keep "spair_root" for backward compatibility with analysis notebooks.
        "spair_root": root if spec.dataset == "spair" else None,
        "metrics": metrics,
        "spec": asdict(spec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "throughput_img_per_sec": _throughput,
        "peak_memory_mb": _peak_mb,
        "sd4match_per_image": image_acc,
        "sd4match_per_point": point_acc,
        "sd4match_summary": {"image": summary_image, "point": summary_point},
        "sd4match_by_difficulty_flag": by_difficulty,
    }
    if load_info is not None:
        out["checkpoint_load"] = load_info
    return out


def run_comparison_batch(
    specs: Sequence[EvalRunSpec],
    *,
    spair_root: Optional[str] = None,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    return [run_spair_pck_eval(s, spair_root=spair_root, alphas=alphas, device=device) for s in specs]


def metrics_rows_for_table(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in results:
        row = {"name": r["name"], "split": r["spec"]["split"], "wsa": r["spec"]["use_window_soft_argmax"]}
        row.update(r["metrics"])
        rows.append(row)
    return rows


__all__ = [
    "EvalRunSpec",
    "metrics_rows_for_table",
    "run_comparison_batch",
    "run_spair_pck_eval",
]
