"""Evaluation helpers: PCK runner over SPair-71k.

Final reported numbers must use the ``test`` split; use ``val`` for model selection.
"""

from .checkpoint_loader import load_encoder_weights_from_pt
from .experiment_runner import EvalRunSpec, metrics_rows_for_table, run_comparison_batch, run_spair_pck_eval

__all__ = [
    "EvalRunSpec",
    "load_encoder_weights_from_pt",
    "metrics_rows_for_table",
    "run_comparison_batch",
    "run_spair_pck_eval",
]
