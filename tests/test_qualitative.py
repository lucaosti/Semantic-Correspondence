"""Tests for ``evaluation.qualitative`` parsing/heuristics (no torch inference)."""

from __future__ import annotations


import pytest

from evaluation import qualitative as Q


# ---------------------------------------------------------------------------
# MethodSpec / checkpoint resolution
# ---------------------------------------------------------------------------


def test_method_spec_label_includes_wsa():
    s = Q.MethodSpec("dinov2_vitb14", "ft_lb2", wsa=True)
    assert "WSA" in s.label
    assert "dinov2_vitb14" in s.label
    assert "ft_lb2" in s.label


def test_resolve_checkpoint_baseline_returns_none(tmp_path):
    spec = Q.MethodSpec("dinov2_vitb14", "baseline")
    assert Q.resolve_checkpoint_path(spec, tmp_path) is None


def test_resolve_checkpoint_ft_returns_path_when_present(tmp_path):
    p = tmp_path / "dinov2_vitb14_lastblocks2_best.pt"
    p.write_bytes(b"")
    spec = Q.MethodSpec("dinov2_vitb14", "ft_lb2")
    out = Q.resolve_checkpoint_path(spec, tmp_path)
    assert out == p


def test_resolve_checkpoint_ft_returns_none_when_missing(tmp_path):
    spec = Q.MethodSpec("dinov2_vitb14", "ft_lb2")
    assert Q.resolve_checkpoint_path(spec, tmp_path) is None


def test_resolve_checkpoint_lora_uses_rank_in_filename(tmp_path):
    p = tmp_path / "sam_vit_b_lora_r8_best.pt"
    p.write_bytes(b"")
    spec = Q.MethodSpec("sam_vit_b", "lora")
    assert Q.resolve_checkpoint_path(spec, tmp_path, lora_rank=8) == p
    assert Q.resolve_checkpoint_path(spec, tmp_path, lora_rank=16) is None


def test_resolve_checkpoint_unknown_method(tmp_path):
    spec = Q.MethodSpec("dinov2_vitb14", "no_such_method")
    assert Q.resolve_checkpoint_path(spec, tmp_path) is None


# ---------------------------------------------------------------------------
# find_failure_cases
# ---------------------------------------------------------------------------


def _result(name: str, pck_list, alpha=0.1):
    return {
        "name": name,
        "sd4match_per_image": {f"custom_pck{alpha}": {"all": list(pck_list)}},
    }


def test_find_failure_cases_returns_lowest_k_sorted_ascending():
    res = [_result("foo", [0.9, 0.1, 0.5, 0.0, 0.7, 0.3])]
    out = Q.find_failure_cases(res, run_name="foo", alpha=0.1, k=3)
    assert out == [(3, 0.0), (1, pytest.approx(0.1)), (5, pytest.approx(0.3))]


def test_find_failure_cases_unknown_run_returns_empty():
    res = [_result("foo", [0.5])]
    assert Q.find_failure_cases(res, run_name="bar", alpha=0.1, k=3) == []


def test_find_failure_cases_no_per_image_returns_empty():
    assert Q.find_failure_cases([{"name": "foo"}], run_name="foo", alpha=0.1, k=3) == []


def test_find_failure_cases_k_capped_at_population():
    res = [_result("foo", [0.5, 0.2])]
    out = Q.find_failure_cases(res, run_name="foo", alpha=0.1, k=10)
    assert len(out) == 2


def test_find_failure_cases_k_minimum_one():
    res = [_result("foo", [0.5, 0.2])]
    out = Q.find_failure_cases(res, run_name="foo", alpha=0.1, k=0)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# find_symmetry_ambiguity
# ---------------------------------------------------------------------------


def test_symmetry_ambiguity_flags_left_right_swap():
    src_named = [("left_eye", 10.0, 10.0), ("right_eye", 30.0, 10.0)]
    # Predictions are swapped: the "left_eye" prediction lands on the right_eye GT.
    gt_xy = [(50.0, 100.0), (150.0, 100.0)]
    pred_xy = [(150.0, 100.0), (50.0, 100.0)]  # swapped
    out = Q.find_symmetry_ambiguity(
        src_kps_named=src_named, pred_kps_xy=pred_xy, gt_kps_xy=gt_xy,
        pck_threshold=200.0, alpha=0.1,
    )
    assert sorted(out) == [0, 1]


def test_symmetry_ambiguity_skips_correct_predictions():
    src_named = [("left_eye", 10.0, 10.0), ("right_eye", 30.0, 10.0)]
    gt_xy = [(50.0, 100.0), (150.0, 100.0)]
    pred_xy = [(50.0, 100.0), (150.0, 100.0)]  # both correct
    out = Q.find_symmetry_ambiguity(
        src_kps_named=src_named, pred_kps_xy=pred_xy, gt_kps_xy=gt_xy,
        pck_threshold=200.0, alpha=0.1,
    )
    assert out == []


def test_symmetry_ambiguity_skips_unlabeled_keypoints():
    src_named = [("nose", 10.0, 10.0), ("ear", 30.0, 10.0)]
    gt_xy = [(50.0, 100.0), (150.0, 100.0)]
    pred_xy = [(150.0, 100.0), (50.0, 100.0)]
    out = Q.find_symmetry_ambiguity(
        src_kps_named=src_named, pred_kps_xy=pred_xy, gt_kps_xy=gt_xy,
        pck_threshold=200.0, alpha=0.1,
    )
    assert out == []


def test_symmetry_ambiguity_empty_inputs():
    assert Q.find_symmetry_ambiguity(
        src_kps_named=[], pred_kps_xy=[], gt_kps_xy=[],
        pck_threshold=100.0, alpha=0.1,
    ) == []


