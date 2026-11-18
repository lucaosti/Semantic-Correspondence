from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_pipeline as rp

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("family", ["wsa_sweep", "layer_sweep", "main"])
def test_build_eval_specs_experiment_config(family, tmp_path):
    """Every eval family must build from the real experiment config (regression: duplicate dino_layer)."""
    rp._apply_pipeline_yaml(REPO / "configs" / "experiment.yaml")
    logger = rp.PipelineLogger(tmp_path / "log.txt") if hasattr(rp, "PipelineLogger") else None
    specs = rp._build_eval_specs(REPO, "sam.pth", logger, num_workers=0, family=family)
    if family == "layer_sweep":
        assert {s.dino_layer for s in specs} == {int(li) for li in rp.EVAL_LAYER_LIST} or not specs
