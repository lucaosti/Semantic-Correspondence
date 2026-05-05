"""SPair-71k PCK evaluation runner.

Use :class:`EvalRunSpec` to describe a single run; :func:`run_spair_pck_eval` returns
the metric dictionaries used for logging, JSON exports and comparison tables.

Runs in ``torch.inference_mode`` with bf16 autocast on CUDA Ampere+. Forward over
``src`` and ``tgt`` is fused into a single ``extractor(cat([src, tgt]))``, the
matcher is fully batched (one ``grid_sample`` + one ``einsum`` per batch), and PCK
aggregation is computed in-house — no SD4Match dependency.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn
from data.paths import resolve_spair_root
from evaluation.checkpoint_loader import load_encoder_weights_from_pt
from models.common.dense_extractor import BackboneName, DenseExtractorConfig, DenseFeatureExtractor
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
    wsa_temperature: float = 1.0
    preprocess: str = "FIXED_RESIZE"
    height: int = 784
    width: int = 784
    limit: int = 0
    num_workers: int = 4
    batch_size: int = 8
    dino_layer_indices: Any = field(default_factory=lambda: 4)

    def to_dense_config(self) -> DenseExtractorConfig:
        return DenseExtractorConfig(
            name=BackboneName(self.backbone),
            dinov2_weights_path=self.dinov2_weights,
            dinov3_weights_path=self.dinov3_weights,
            sam_checkpoint_path=self.sam_checkpoint,
            dino_layer_indices=self.dino_layer_indices,
        )


def _build_eval_dataset(spec: EvalRunSpec, spair_root: str) -> SPair71kPairDataset:
    mode = PreprocessMode[str(spec.preprocess).strip().upper()]
    return SPair71kPairDataset(
        spair_root=spair_root,
        split=spec.split,
        preprocess=mode,
        output_size_hw=(int(spec.height), int(spec.width)),
        normalize=True,
        photometric_augment=None,
    )


def _build_eval_loader(
    dataset: SPair71kPairDataset,
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


def _batched_match(
    feat_src: torch.Tensor,
    feat_tgt: torch.Tensor,
    src_kps_coord: torch.Tensor,
    coord_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pred_xy[B, N, 2], sim_maps[B, N, Hf, Wf])`` in the coord frame."""
    bsz, _, hf, wf = feat_src.shape
    n_pts = src_kps_coord.shape[1]
    ch, cw = float(coord_hw[0]), float(coord_hw[1])

    xf = (src_kps_coord[..., 0] + 0.5) * (float(wf) / cw) - 0.5
    yf = (src_kps_coord[..., 1] + 0.5) * (float(hf) / ch) - 0.5
    gx = (xf / max(wf - 1, 1)) * 2.0 - 1.0
    gy = (yf / max(hf - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2).to(feat_src.dtype)

    padding_mode = "zeros" if feat_src.device.type == "mps" else "border"
    desc = F.grid_sample(
        feat_src, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False
    ).squeeze(-1).transpose(1, 2)  # (B, N, C)

    sims = torch.einsum("bnc,bchw->bnhw", desc, feat_tgt)

    flat = sims.reshape(bsz * n_pts, -1)
    idx = torch.argmax(flat, dim=1)
    xi = (idx % wf).to(sims.dtype)
    yi = (idx // wf).to(sims.dtype)
    x_pix = (xi + 0.5) * (cw / float(wf)) - 0.5
    y_pix = (yi + 0.5) * (ch / float(hf)) - 0.5
    pred = torch.stack([x_pix, y_pix], dim=-1).reshape(bsz, n_pts, 2)
    return pred, sims


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
    root = resolve_spair_root(spair_root)
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

    with torch.inference_mode(), _autocast_eval(eval_device):
        for batch in loader:
            src = batch["src_img"].to(eval_device, non_blocking=pin_mem)
            tgt = batch["tgt_img"].to(eval_device, non_blocking=pin_mem)
            src_kps = batch["src_kps"].to(eval_device)
            tgt_kps = batch["tgt_kps"].to(eval_device)
            n_valid = batch["n_valid_keypoints"].to(eval_device).reshape(-1)
            pckthres = batch["pck_threshold_bbox"].to(eval_device).reshape(-1)
            categories = batch["category"]
            bsz = int(src.shape[0])
            h_in, w_in = int(src.shape[2]), int(src.shape[3])

            feats, meta = extractor(torch.cat([src, tgt], dim=0))
            feats = feats.float()  # autocast bf16 forward + fp32 matching/PCK
            feats_src, feats_tgt = feats[:bsz], feats[bsz:]
            coord_h, coord_w = meta.get("coord_hw", (h_in, w_in))

            if (coord_h, coord_w) != (h_in, w_in):
                scale_to = src_kps.new_tensor([coord_w / float(w_in), coord_h / float(h_in)])
                src_kps_coord = src_kps * scale_to
            else:
                src_kps_coord = src_kps

            pred_coord, sims = _batched_match(
                feats_src, feats_tgt, src_kps_coord, (coord_h, coord_w)
            )

            if spec.use_window_soft_argmax:
                bsz_, n_pts, hf, wf = sims.shape
                refined = refine_predictions_window_soft_argmax(
                    sims.reshape(bsz_ * n_pts, hf, wf),
                    (coord_h, coord_w),
                    window_size=spec.wsa_window,
                    temperature=spec.wsa_temperature,
                ).reshape(bsz_, n_pts, 2)
                pred_coord = refined

            if (coord_h, coord_w) != (h_in, w_in):
                scale_back = pred_coord.new_tensor([w_in / float(coord_w), h_in / float(coord_h)])
                pred_ds = pred_coord * scale_back
            else:
                pred_ds = pred_coord

            dists = torch.norm(tgt_kps - pred_ds, dim=-1)  # (B, N)
            # (n_alpha, B, N) bool
            correct_all = (dists.unsqueeze(0) <= alphas_t.view(-1, 1, 1) * pckthres.view(1, -1, 1)).cpu()
            n_valid_list = n_valid.cpu().tolist()

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
        "spair_root": root,
        "metrics": metrics,
        "spec": asdict(spec),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
