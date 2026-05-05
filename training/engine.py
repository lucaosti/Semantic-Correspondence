"""Vectorized Gaussian-CE training step shared by every backbone.

The forward over ``src`` and ``tgt`` is fused (a single ``extractor(cat([src, tgt]))``)
and the per-pair loop has been replaced by a fully batched expression: per-keypoint
descriptor sampling via one ``grid_sample`` call, similarity maps through ``einsum``,
Gaussian targets built once, mean over the valid mask. No host/device sync inside.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from data.dataset import INVALID_KP_COORD


def correspondence_gaussian_loss(
    extractor: torch.nn.Module,
    batch: Dict[str, Any],
    *,
    sigma_feat: float = 1.0,
) -> torch.Tensor:
    """Mean Gaussian cross-entropy over valid (src, tgt) keypoint pairs.

    Works for any backbone wrapped by :class:`DenseFeatureExtractor`. SAM keypoints are
    rescaled to the extractor's coordinate frame (``meta["coord_hw"]``) once.
    """
    src = batch["src_img"]
    tgt = batch["tgt_img"]
    bsz = int(src.shape[0])
    if int(tgt.shape[0]) != bsz:
        raise ValueError("src/tgt batch mismatch.")

    feats, meta = extractor(torch.cat([src, tgt], dim=0))
    feats_src, feats_tgt = feats[:bsz], feats[bsz:]
    h_in, w_in = int(src.shape[2]), int(src.shape[3])
    coord_h, coord_w = meta.get("coord_hw", (h_in, w_in))

    src_kps = batch["src_kps"]
    tgt_kps = batch["tgt_kps"]
    n_valid = batch["n_valid_keypoints"].view(-1)

    # Validity mask in dataset frame (before any rescale: scaling can shift the
    # padding sentinel, so we read it against INVALID_KP_COORD here).
    n_pts = src_kps.shape[1]
    arange = torch.arange(n_pts, device=src_kps.device).unsqueeze(0)
    valid = (
        (arange < n_valid.view(-1, 1))
        & (src_kps[..., 0] > INVALID_KP_COORD)
        & (src_kps[..., 1] > INVALID_KP_COORD)
        & (tgt_kps[..., 0] > INVALID_KP_COORD)
        & (tgt_kps[..., 1] > INVALID_KP_COORD)
    )

    if (coord_h, coord_w) != (h_in, w_in):
        scale = src_kps.new_tensor([coord_w / float(w_in), coord_h / float(h_in)])
        src_kps = src_kps * scale
        tgt_kps = tgt_kps * scale

    hf, wf = int(feats_src.shape[-2]), int(feats_src.shape[-1])
    fh, fw = float(hf), float(wf)
    ch, cw = float(coord_h), float(coord_w)

    # Pixel -> feature-grid coords (centers aligned), then normalize to [-1, 1].
    xf_src = (src_kps[..., 0] + 0.5) * (fw / cw) - 0.5
    yf_src = (src_kps[..., 1] + 0.5) * (fh / ch) - 0.5
    gx = (xf_src / max(wf - 1, 1)) * 2.0 - 1.0
    gy = (yf_src / max(hf - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(2).to(feats_src.dtype)

    padding_mode = "zeros" if feats_src.device.type == "mps" else "border"
    sampled = F.grid_sample(
        feats_src, grid, mode="bilinear", padding_mode=padding_mode, align_corners=False
    )
    desc = sampled.squeeze(-1).transpose(1, 2)  # (B, N, C)

    # (B, N, Hf, Wf) cosine similarity maps; features are L2-normalized upstream.
    sims = torch.einsum("bnc,bchw->bnhw", desc, feats_tgt)

    # Gaussian target centered on the (rescaled) target keypoints.
    xf_tgt = (tgt_kps[..., 0] + 0.5) * (fw / cw) - 0.5
    yf_tgt = (tgt_kps[..., 1] + 0.5) * (fh / ch) - 0.5
    yy, xx = torch.meshgrid(
        torch.arange(hf, device=feats_src.device, dtype=sims.dtype),
        torch.arange(wf, device=feats_src.device, dtype=sims.dtype),
        indexing="ij",
    )
    dx = xx[None, None] - xf_tgt[..., None, None].to(sims.dtype)
    dy = yy[None, None] - yf_tgt[..., None, None].to(sims.dtype)
    target = torch.softmax(
        (-(dx * dx + dy * dy) / (2.0 * float(sigma_feat) ** 2 + 1e-8)).reshape(bsz, n_pts, -1),
        dim=-1,
    ).reshape_as(sims)

    log_prob = F.log_softmax(sims.reshape(bsz, n_pts, -1), dim=-1).reshape_as(sims)
    per_kp = -(target * log_prob).sum(dim=(2, 3))  # (B, N)

    n_per_pair = valid.sum(dim=1).clamp_min(1).to(per_kp.dtype)
    return ((per_kp * valid).sum(dim=1) / n_per_pair).mean()


__all__ = ["correspondence_gaussian_loss"]
