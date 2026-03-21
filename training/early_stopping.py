"""
Early stopping utilities driven by validation PCK or validation loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of consecutive evaluations without improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    mode:
        ``max`` for metrics like PCK, ``min`` for loss.
    """

    patience: int = 5
    min_delta: float = 0.0
    mode: Literal["max", "min"] = "max"
    best_value: Optional[float] = None
    num_bad_epochs: int = 0
    best_epoch: int = field(default=-1)

    def _is_better(self, new: float, old: float) -> bool:
        if self.mode == "max":
            return new > old + self.min_delta
        return new < old - self.min_delta

    def step(self, value: float, epoch: int) -> bool:
        """
        Update internal state.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if self.best_value is None:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.num_bad_epochs = 0
            return False

        assert self.best_value is not None
        if self._is_better(float(value), float(self.best_value)):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


__all__ = ["EarlyStopping"]
