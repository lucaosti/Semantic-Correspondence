"""Tests for ``evaluation.figures`` parsing/aggregation (no plotting side-effects)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import figures as F


# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,backbone,method,last_blocks,wsa", [
    ("dinov2_vitb14_baseline", "dinov2_vitb14", "baseline", None, False),
    ("dinov2_vitb14_baseline_wsa", "dinov2_vitb14", "baseline", None, True),
    ("dinov3_vitb16_ft_lb1", "dinov3_vitb16", "ft_lb1", 1, False),
    ("dinov3_vitb16_ft_lb2_wsa", "dinov3_vitb16", "ft_lb2", 2, True),
    ("sam_vit_b_ft_lb4", "sam_vit_b", "ft_lb4", 4, False),
    ("sam_vit_b_lora", "sam_vit_b", "lora", None, False),
    ("dinov2_vitb14_lora_wsa", "dinov2_vitb14", "lora", None, True),
])
def test_parse_run_name_canonical(name, backbone, method, last_blocks, wsa):
    info = F.parse_run_name(name)
    assert info is not None
    assert info.backbone == backbone
    assert info.method == method
    assert info.last_blocks == last_blocks
    assert info.wsa is wsa


@pytest.mark.parametrize("name,window", [
    ("dinov2_vitb14_baseline_wsa_w3", 3),
    ("dinov3_vitb16_baseline_wsa_w7", 7),
])
def test_parse_run_name_wsa_sweep(name, window):
    info = F.parse_run_name(name)
    assert info is not None
    assert info.method == "baseline"
    assert info.wsa is True
    assert info.wsa_window == window


@pytest.mark.parametrize("name,layer", [
    ("dinov2_vitb14_baseline_layer4", 4),
    ("dinov3_vitb16_baseline_layer11", 11),
])
def test_parse_run_name_layer_sweep(name, layer):
    info = F.parse_run_name(name)
    assert info is not None
    assert info.method == "baseline"
    assert info.wsa is False
    assert info.layer_index == layer


@pytest.mark.parametrize("bad", ["", "garbage", "dinov2_vitb14", "unknown_baseline", "vit_b_baseline"])
def test_parse_run_name_returns_none_for_garbage(bad):
    assert F.parse_run_name(bad) is None


# ---------------------------------------------------------------------------
# Loading exports
# ---------------------------------------------------------------------------


def _write_exports(tmp_path: Path, *, with_per_cat: bool = True, with_diff: bool = True) -> Path:
    exports = tmp_path / "exports"
    exports.mkdir(parents=True)
    pck = [
        {
            "name": "dinov2_vitb14_baseline",
            "metrics": {"pck@0.05": 0.30, "pck@0.1": 0.50, "pck@0.2": 0.72,
                        "pck_pt@0.05": 0.28, "pck_pt@0.1": 0.49, "pck_pt@0.2": 0.71},
            "spec": {"split": "test", "use_window_soft_argmax": False},
        },
        {
            "name": "dinov2_vitb14_baseline_wsa",
            "metrics": {"pck@0.05": 0.33, "pck@0.1": 0.54, "pck@0.2": 0.74,
                        "pck_pt@0.05": 0.31, "pck_pt@0.1": 0.52, "pck_pt@0.2": 0.73},
            "spec": {"split": "test", "use_window_soft_argmax": True},
        },
        {
            "name": "sam_vit_b_ft_lb2",
            "metrics": {"pck@0.05": 0.40, "pck@0.1": 0.62, "pck@0.2": 0.80,
                        "pck_pt@0.05": 0.39, "pck_pt@0.1": 0.61, "pck_pt@0.2": 0.79},
            "spec": {"split": "test", "use_window_soft_argmax": False},
        },
    ]
    (exports / "pck_results.json").write_text(json.dumps(pck))
    if with_per_cat:
        per_cat = [
            {"name": "dinov2_vitb14_baseline",
             "categories": {"cat": {"pck@0.05": 0.30, "pck@0.1": 0.50, "pck@0.2": 0.72},
                            "dog": {"pck@0.05": 0.28, "pck@0.1": 0.45, "pck@0.2": 0.69}}},
            {"name": "sam_vit_b_ft_lb2",
             "categories": {"cat": {"pck@0.05": 0.41, "pck@0.1": 0.63, "pck@0.2": 0.81}}},
        ]
        (exports / "pck_results_per_category.json").write_text(json.dumps(per_cat))
    if with_diff:
        by_diff = [
            {"name": "dinov2_vitb14_baseline",
             "data": {
                 "viewpoint_variation": {
                     "0": {"summary": {"image": {"custom_pck0.1": {"all": 0.55}}}},
                     "1": {"summary": {"image": {"custom_pck0.1": {"all": 0.40}}}},
                 },
                 "occlusion": {
                     "0": {"summary": {"image": {"custom_pck0.1": {"all": 0.52}}}},
                     "1": {"summary": {"image": {"custom_pck0.1": {"all": 0.35}}}},
                 },
             }},
        ]
        (exports / "pck_results_by_difficulty_flag.json").write_text(json.dumps(by_diff))
    return exports


def test_load_pck_exports_with_all_files(tmp_path):
    exports = _write_exports(tmp_path)
    data = F.load_pck_exports(exports)
    assert data["available"]["pck_results"]
    assert data["available"]["per_category"]
    assert data["available"]["by_difficulty_flag"]
    assert len(data["pck_results"]) == 3
    assert len(data["per_category"]) == 2
    assert len(data["by_difficulty_flag"]) == 1


def test_load_pck_exports_missing_files_safe(tmp_path):
    exports = _write_exports(tmp_path, with_per_cat=False, with_diff=False)
    data = F.load_pck_exports(exports)
    assert data["available"]["pck_results"]
    assert not data["available"]["per_category"]
    assert not data["available"]["by_difficulty_flag"]
    assert data["per_category"] is None
    for k in ("wsa_sweep", "layer_sweep"):
        assert data[k] is None
        assert data["available"][k] is False


def test_load_pck_exports_empty_dir(tmp_path):
    data = F.load_pck_exports(tmp_path)
    assert not any(data["available"].values())
    assert data["pck_results"] == []


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------


def test_build_master_table_orders_rows(tmp_path):
    pd = pytest.importorskip("pandas")
    exports = _write_exports(tmp_path)
    data = F.load_pck_exports(exports)
    df = F.build_master_table(data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns)[:4] == ["backbone", "method", "last_blocks", "wsa"]
    # DINOv2 should come before SAM
    assert df["backbone"].iloc[0] == "dinov2_vitb14"
    assert df["backbone"].iloc[-1] == "sam_vit_b"


def test_build_master_table_empty_when_no_results():
    pytest.importorskip("pandas")
    df = F.build_master_table({"pck_results": []})
    assert df.empty


# ---------------------------------------------------------------------------
# LaTeX / Markdown formatting
# ---------------------------------------------------------------------------


def test_dataframe_to_latex_bolds_max(tmp_path):
    pytest.importorskip("pandas")
    exports = _write_exports(tmp_path)
    df = F.build_master_table(F.load_pck_exports(exports))
    tex = F.dataframe_to_latex(df, caption="Test", label="tab:test")
    assert r"\begin{table}" in tex
    assert r"\textbf{0.8000}" in tex  # SAM ft_lb2 has max pck@0.2 = 0.80
    assert r"\caption{Test}" in tex


def test_dataframe_to_latex_handles_empty():
    pd = pytest.importorskip("pandas")
    out = F.dataframe_to_latex(pd.DataFrame())
    assert "empty" in out


def test_dataframe_to_markdown_renders_metric_columns(tmp_path):
    pytest.importorskip("tabulate")
    pytest.importorskip("pandas")
    exports = _write_exports(tmp_path)
    df = F.build_master_table(F.load_pck_exports(exports))
    md = F.dataframe_to_markdown(df)
    assert "pck@0.1" in md
    assert "0.5000" in md or "0.50" in md


# ---------------------------------------------------------------------------
# Per-category / per-difficulty parsing
# ---------------------------------------------------------------------------


def test_per_category_table_shape(tmp_path):
    pd = pytest.importorskip("pandas")
    data = F.load_pck_exports(_write_exports(tmp_path))
    pivot = F.per_category_table(data, alpha=0.1)
    assert isinstance(pivot, pd.DataFrame)
    assert "cat" in pivot.columns
    assert "dinov2_vitb14_baseline" in pivot.index
    assert pivot.loc["dinov2_vitb14_baseline", "cat"] == pytest.approx(0.50)


def test_per_category_table_alpha_missing_returns_empty(tmp_path):
    data = F.load_pck_exports(_write_exports(tmp_path))
    pivot = F.per_category_table(data, alpha=0.99)
    assert pivot.empty


def test_per_difficulty_table_long_format(tmp_path):
    pytest.importorskip("pandas")
    data = F.load_pck_exports(_write_exports(tmp_path))
    df = F.per_difficulty_table(data, alpha=0.1)
    assert {"run", "flag", "bucket", "value"} <= set(df.columns)
    rows = df[(df["flag"] == "viewpoint_variation") & (df["bucket"] == 1)]
    assert len(rows) == 1
    assert rows["value"].iloc[0] == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# Training history loading
# ---------------------------------------------------------------------------


def test_load_training_histories_parses_jsonl(tmp_path):
    h = tmp_path / "ckpts"
    h.mkdir()
    f = h / "dinov2_vitb14_ft_lb2_history.jsonl"
    f.write_text(
        "\n".join([
            json.dumps({"epoch": 0, "train_loss": 1.0, "val_loss": 1.1}),
            "",
            "garbage line that is not json",
            json.dumps({"epoch": 1, "train_loss": 0.8, "val_loss": 0.9}),
        ])
    )
    out = F.load_training_histories(h)
    assert "dinov2_vitb14_ft_lb2" in out
    assert len(out["dinov2_vitb14_ft_lb2"]) == 2


def test_load_training_histories_empty_dir(tmp_path):
    assert F.load_training_histories(tmp_path) == {}


