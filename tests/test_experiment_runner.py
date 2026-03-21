"""Tests for evaluation result helpers (no dataset required)."""

from __future__ import annotations

from evaluation.experiment_runner import metrics_rows_for_table


def test_metrics_rows_for_table_flattens():
    results = [
        {
            "name": "a",
            "spec": {"split": "val", "use_window_soft_argmax": False},
            "metrics": {"pck@0.1": 0.5},
        },
        {
            "name": "b",
            "spec": {"split": "test", "use_window_soft_argmax": True},
            "metrics": {"pck@0.1": 0.6},
        },
    ]
    rows = metrics_rows_for_table(results)
    assert len(rows) == 2
    assert rows[0]["name"] == "a"
    assert rows[0]["pck@0.1"] == 0.5
    assert rows[1]["wsa"] is True
