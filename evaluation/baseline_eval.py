"""
End-to-end evaluation helpers for the training-free baseline (Task 1) and optional WSArgmax (Task 3).

**Split policy:** use ``test`` for reported metrics; ``val`` for model selection; ``train`` for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from data.dataset import INVALID_KP_COORD, SPair71kPairDataset, spair_collate_fn
from utils.hardware import recommended_device_str
from evaluation.pck import pck_distance
from models.common.coord_utils import rescale_keypoints_xy
from models.common.dense_extractor import DenseFeatureExtractor
from models.common.matching import predict_correspondences_cosine_argmax
from models.common.window_soft_argmax import refine_predictions_window_soft_argmax


@dataclass
class EvalAccumulator:
    """Running sums for micro-averaged PCK (valid keypoints only)."""

    total_correct: Dict[float, float]
    total_valid: float

    @staticmethod
    def create(alphas: Sequence[float]) -> "EvalAccumulator":
        return EvalAccumulator(total_correct={float(a): 0.0 for a in alphas}, total_valid=0.0)

    def update(
        self,
        pred_xy: torch.Tensor,
        gt_xy: torch.Tensor,
        pck_threshold: torch.Tensor,
        alphas: Sequence[float],
        *,
        invalid_value: float = INVALID_KP_COORD,
    ) -> None:
        valid = (gt_xy[:, 0] > invalid_value) & (gt_xy[:, 1] > invalid_value)
        n_valid = float(torch.sum(valid.to(dtype=torch.float32)).item())
        self.total_valid += n_valid
        if n_valid <= 0:
            return
        for a in alphas:
            correct, _ = pck_distance(pred_xy, gt_xy, pck_threshold, alpha=float(a), invalid_value=invalid_value)
            self.total_correct[float(a)] += float(correct.sum().item())

    def micro_pck(self, alpha: float) -> float:
        if self.total_valid <= 0:
            return 0.0
        return self.total_correct[float(alpha)] / self.total_valid


def _dataset_hw_from_tensor(img_bchw: torch.Tensor) -> Tuple[int, int]:
    """Return ``(H, W)`` for a ``(B,3,H,W)`` tensor."""
    return int(img_bchw.shape[2]), int(img_bchw.shape[3])


def _rescale_kps_if_needed(
    xy: torch.Tensor,
    hw_from: Tuple[int, int],
    hw_to: Tuple[int, int],
) -> torch.Tensor:
    if hw_from == hw_to:
        return xy
    return rescale_keypoints_xy(xy, hw_from, hw_to)


@torch.no_grad()
def evaluate_spair_loader(
    loader: DataLoader,
    extractor: DenseFeatureExtractor,
    *,
    alphas: Sequence[float] = (0.05, 0.1, 0.15),
    use_window_soft_argmax: bool = False,
    wsa_window: int = 5,
    wsa_temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute micro-averaged PCK@alpha over a dataloader (expects batch size ``1``).

    The extractor must match the dataset preprocessing so that ``meta["coord_hw"]`` aligns
    with keypoint rescaling (especially important for SAM).

    Parameters
    ----------
    loader:
        DataLoader yielding batches from :class:`data.dataset.SPair71kPairDataset`.
    extractor:
        Frozen dense feature extractor.
    alphas:
        PCK relative thresholds.
    use_window_soft_argmax:
        If ``True``, apply window soft-argmax **inference-only** refinement on target similarity maps.
    wsa_window, wsa_temperature:
        Window soft-argmax hyperparameters.
    device:
        If ``None``, uses CUDA if available, else Apple MPS, else CPU.

    Returns
    -------
    dict[str, float]
        Keys like ``pck@0.10`` with micro-averaged scores in ``[0,1]``.
    """
    if device is None:
        device = torch.device(recommended_device_str())
    extractor.eval()
    extractor.to(device)

    acc = EvalAccumulator.create(alphas)

    for batch in loader:
        src = batch["src_img"].to(device)
        tgt = batch["tgt_img"].to(device)
        if src.shape[0] != 1:
            raise ValueError("evaluate_spair_loader currently expects batch size 1.")

        src_kps = batch["src_kps"][0].to(device)
        tgt_kps = batch["tgt_kps"][0].to(device)
        n_valid = int(batch["n_valid_keypoints"][0].item())
        pck_thr = batch["pck_threshold_bbox"].to(device)
        if pck_thr.dim() > 0:
            pck_thr = pck_thr.reshape(())

        feat_src, meta_src = extractor(src)
        feat_tgt, meta_tgt = extractor(tgt)

        ds_src = _dataset_hw_from_tensor(src)
        ds_tgt = _dataset_hw_from_tensor(tgt)
        coord_src = meta_src.get("coord_hw", ds_src)
        coord_tgt = meta_tgt.get("coord_hw", ds_tgt)

        src_match = _rescale_kps_if_needed(src_kps, ds_src, coord_src)

        out = predict_correspondences_cosine_argmax(
            feat_src,
            feat_tgt,
            src_match,
            img_hw=coord_src,
            img_hw_src=coord_src,
            img_hw_tgt=coord_tgt,
            valid_mask=None,
        )
        pred_tgt = out["pred_tgt_xy"]
        sims = out["sim_maps"]

        if use_window_soft_argmax:
            pred_tgt = refine_predictions_window_soft_argmax(
                sims,
                coord_tgt,
                window_size=wsa_window,
                temperature=wsa_temperature,
            )

        pred_in_dataset = _rescale_kps_if_needed(pred_tgt, coord_tgt, ds_tgt)

        acc.update(pred_in_dataset, tgt_kps, pck_thr, alphas)

    return {f"pck@{a:g}": acc.micro_pck(a) for a in alphas}


def build_eval_dataloader(
    spair_root: str,
    split: str,
    *,
    batch_size: int = 1,
    num_workers: int = 4,
    preprocess: str = "fixed_resize",
    output_size_hw: Tuple[int, int] = (784, 784),
    pin_memory: Optional[bool] = None,
) -> DataLoader:
    """
    Convenience constructor for evaluation dataloaders with safe defaults.

    Parameters
    ----------
    spair_root:
        Path to ``SPair-71k``.
    split:
        ``test`` / ``val`` / ``train`` (mapped to ``trn.txt`` for train).
    preprocess:
        Name of :class:`data.dataset.PreprocessMode`.
    pin_memory:
        If ``None``, enables pinned memory when CUDA is available (legacy default).
    """
    from data.dataset import PreprocessMode

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    mode = PreprocessMode[preprocess.strip().upper()]
    ds = SPair71kPairDataset(
        spair_root=spair_root,
        split=split,
        preprocess=mode,
        output_size_hw=output_size_hw,
        normalize=True,
        photometric_augment=None,
    )
    dl_kw = {}
    if num_workers > 0:
        from utils.hardware import dataloader_extra_kwargs

        dl_kw = dataloader_extra_kwargs(num_workers)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=spair_collate_fn,
        pin_memory=pin_memory,
        **dl_kw,
    )


__all__ = ["EvalAccumulator", "build_eval_dataloader", "evaluate_spair_loader"]
