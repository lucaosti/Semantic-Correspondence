"""
Evaluation metrics (e.g., PCK) and benchmark runners.

Final reported numbers must use the ``test`` split only; use ``val`` for model selection.
"""

from .baseline_eval import EvalAccumulator, build_eval_dataloader, evaluate_spair_loader
from .checkpoint_loader import load_encoder_weights_from_pt
from .experiment_runner import EvalRunSpec, metrics_rows_for_table, run_comparison_batch, run_spair_pck_eval
from .pck import mean_pck, pck_distance

__all__ = [
    "EvalAccumulator",
    "EvalRunSpec",
    "build_eval_dataloader",
    "evaluate_spair_loader",
    "load_encoder_weights_from_pt",
    "mean_pck",
    "metrics_rows_for_table",
    "pck_distance",
    "run_comparison_batch",
    "run_spair_pck_eval",
]
