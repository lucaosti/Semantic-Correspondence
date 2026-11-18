"""Vectorized Gaussian-CE training step shared by every backbone.

The forward over ``src`` and ``tgt`` is fused (a single ``extractor(cat([src, tgt]))``)
and the loss is fully batched: descriptor sampling and similarity maps reuse the exact
functions of the evaluation matcher, so training and evaluation share one coordinate
convention.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from data.dataset import INVALID_KP_COORD
from models.common.coord_utils import rescale_coords
from models.common.matching import sample_descriptors, similarity_maps

# Softmax temperature on cosine similarities: logits in [-1/T, 1/T]. Without it the
# softmax over ~1.4k cells cannot put more than ~0.5% mass on any cell.
DEFAULT_LOSS_TEMPERATURE: float = 0.07


def correspondence_gaussian_loss(
    extractor: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    sigma_feat: float = 1.0,
    temperature: float = DEFAULT_LOSS_TEMPERATURE,
) -> torch.Tensor:
    """Mean Gaussian cross-entropy over valid (src, tgt) keypoint pairs.

    For each valid source keypoint: sample its descriptor, build the cosine-similarity map
    over the target grid, and take the cross-entropy between ``softmax(sims / T)`` and a
    normalized Gaussian (``sigma_feat`` cells) centred on the target keypoint. The loss is
    averaged over the valid keypoints of each pair, then over the pairs of the batch.
    Keypoints are in the dataset frame for every backbone (see ``models.common.matching``).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    src = batch["src_img"]
    tgt = batch["tgt_img"]
    bsz = int(src.shape[0])
    if int(tgt.shape[0]) != bsz:
        raise ValueError("src/tgt batch mismatch.")

    feats, _meta = extractor(torch.cat([src, tgt], dim=0))
    feats_src, feats_tgt = feats[:bsz], feats[bsz:]
    img_hw = (int(src.shape[2]), int(src.shape[3]))

    src_kps = batch["src_kps"]
    tgt_kps = batch["tgt_kps"]
    n_valid = batch["n_valid_keypoints"].view(-1)
    n_pts = src_kps.shape[1]
    arange = torch.arange(n_pts, device=src_kps.device).unsqueeze(0)
    valid = (
        (arange < n_valid.view(-1, 1))
        & (src_kps[..., 0] > INVALID_KP_COORD)
        & (src_kps[..., 1] > INVALID_KP_COORD)
        & (tgt_kps[..., 0] > INVALID_KP_COORD)
        & (tgt_kps[..., 1] > INVALID_KP_COORD)
    )

    desc = sample_descriptors(feats_src, src_kps, img_hw)
    sims = similarity_maps(desc, feats_tgt)  # (B, N, Hf, Wf), fp32
    hf, wf = int(sims.shape[-2]), int(sims.shape[-1])

    xf_tgt = rescale_coords(tgt_kps[..., 0].float(), img_hw[1], wf)
    yf_tgt = rescale_coords(tgt_kps[..., 1].float(), img_hw[0], hf)
    yy, xx = torch.meshgrid(
        torch.arange(hf, device=sims.device, dtype=sims.dtype),
        torch.arange(wf, device=sims.device, dtype=sims.dtype),
        indexing="ij",
    )
    dx = xx[None, None] - xf_tgt[..., None, None]
    dy = yy[None, None] - yf_tgt[..., None, None]
    target = torch.softmax(
        (-(dx * dx + dy * dy) / (2.0 * float(sigma_feat) ** 2)).reshape(bsz, n_pts, -1), dim=-1
    )
    log_prob = F.log_softmax(sims.reshape(bsz, n_pts, -1) / temperature, dim=-1)
    per_kp = -(target * log_prob).sum(dim=-1)  # (B, N)

    validf = valid.to(per_kp.dtype)
    n_per_pair = validf.sum(dim=1)
    per_pair = (per_kp * validf).sum(dim=1) / n_per_pair.clamp_min(1.0)
    has_kp = n_per_pair > 0
    if not bool(has_kp.any()):
        raise ValueError("Batch has no valid keypoints.")
    return per_pair[has_kp].mean()


__all__ = ["DEFAULT_LOSS_TEMPERATURE", "correspondence_gaussian_loss"]
