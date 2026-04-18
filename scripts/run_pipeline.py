#!/usr/bin/env python3
"""Orchestrate dataset verify, train (finetune + LoRA), PCK eval, exports, pytest.

Run: ``python scripts/run_pipeline.py`` or ``python scripts/run_pipeline.py --config config.yaml``.
YAML overrides the in-file defaults (same keys as notebook ``config.yaml``). Triplet toggles
are ``(DINOv2 ViT-B/14, DINOv3 ViT-B/16, SAM ViT-B)``. SAM needs
``checkpoints/sam_vit_b_01ec64.pth`` (from ``scripts/download_pretrained_weights.py``).
Logs: ``runs/logs/pipeline_<utc>.log``, ``runs/logs/current.log``, ``runs/logs/stage_events.jsonl``.
Resume: ``runs/pipeline_state.json`` + ``*_resume.pt`` (saved every
``RESUME_SAVE_INTERVAL`` / ``--resume-save-interval`` batches). Config-fingerprint mismatch
preserves completed steps; full reset via ``SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
_SAM_VIT_B_DEFAULT = _REPO / "checkpoints" / "sam_vit_b_01ec64.pth"

# =============================================================================
# TOGGLES — triplets follow (DINOv2, DINOv3, SAM)
# =============================================================================

PIPELINE_RESUME: bool = True
RUN_VERIFY_DATASET: bool = True
TRAIN_FINETUNE: Tuple[bool, bool, bool] = (True, True, True)
TRAIN_LORA: Tuple[bool, bool, bool] = (True, True, True)
RUN_EVAL_BASELINE: Tuple[bool, bool, bool] = (True, True, True)
RUN_EVAL_BASELINE_WSA: Tuple[bool, bool, bool] = (True, True, True)
RUN_EVAL_FINETUNED_CHECKPOINT: Tuple[bool, bool, bool] = (True, True, True)
RUN_EVAL_LORA_CHECKPOINT: Tuple[bool, bool, bool] = (True, True, True)
RUN_EVAL_FINETUNED_WSA: Tuple[bool, bool, bool] = (True, True, True)
RUN_EVAL_LORA_WSA: Tuple[bool, bool, bool] = (True, True, True)
RUN_EXPORT_METRICS_TABLES: bool = True
RUN_PYTEST: bool = True

# =============================================================================
# SHARED CONFIG
# =============================================================================

SPAIR_ROOT: Optional[str] = None
CHECKPOINT_DIR: str = "checkpoints"
LAST_BLOCKS_LIST: List[int] = [1, 2, 4]
LORA_RANK: int = 8
LORA_LAST_BLOCKS: int = 2

DINOV2_WEIGHTS: Optional[str] = None
DINOV3_WEIGHTS: Optional[str] = None
SAM_CHECKPOINT: Optional[str] = str(_SAM_VIT_B_DEFAULT) if _SAM_VIT_B_DEFAULT.is_file() else None

FT_BATCH_SIZE: int = 20
FT_EPOCHS: int = 50
FT_PATIENCE: int = 7
FT_LR: float = 5e-5
FT_WEIGHT_DECAY: float = 0.01
LORA_BATCH_SIZE: int = 20
LORA_EPOCHS: int = 50
LORA_PATIENCE: int = 7
LORA_LR: float = 1e-3
LORA_ALPHA: float = 16.0
PRECISION: str = "auto"
FT_BATCH_SIZE_BY_BACKBONE: Dict[str, int] = {"sam_vit_b": 4}
LORA_BATCH_SIZE_BY_BACKBONE: Dict[str, int] = {"sam_vit_b": 4}
RESUME_SAVE_INTERVAL: int = 50
LOG_BATCH_INTERVAL: int = 50
DINO_LAYER_INDICES: int = 4
PREPROCESS: str = "FIXED_RESIZE"
IMAGE_SIZE_BY_BACKBONE: Dict[str, Tuple[int, int]] = {
    "dinov2_vitb14": (518, 518),
    "dinov3_vitb16": (512, 512),
    "sam_vit_b": (512, 512),
}
IMAGE_HEIGHT: int = 784
IMAGE_WIDTH: int = 784
NUM_WORKERS: Optional[int] = None
DEVICE: Optional[str] = None

EVAL_SPLIT: str = "test"
EVAL_LIMIT: int = 0
EVAL_ALPHAS: Tuple[float, ...] = (0.05, 0.1, 0.2)
WSA_WINDOW: int = 5
WSA_TEMPERATURE: float = 1.0

BACKBONE_NAMES: Tuple[str, str, str] = ("dinov2_vitb14", "dinov3_vitb16", "sam_vit_b")
_ALLOWED_PRECISION: Tuple[str, ...] = ("auto", "fp32", "bf16", "fp16")


def _normalize_precision_token(value: Any) -> str:
    token = str(value).strip().lower()
    if token not in _ALLOWED_PRECISION:
        raise ValueError(f"PRECISION must be one of: {', '.join(_ALLOWED_PRECISION)}. Got {value!r}.")
    return token


def _parse_backbone_int_map(name: str, value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping {{backbone_name: int}}.")
    out: Dict[str, int] = {}
    for k, v in value.items():
        backbone = str(k).strip()
        if backbone not in BACKBONE_NAMES:
            raise ValueError(f"{name} contains unknown backbone {backbone!r}.")
        iv = int(v)
        if iv < 1:
            raise ValueError(f"{name}[{backbone!r}] must be >= 1, got {iv}.")
        out[backbone] = iv
    return out


def _resolve_image_hw(backbone: str) -> Tuple[int, int]:
    if backbone in IMAGE_SIZE_BY_BACKBONE:
        return IMAGE_SIZE_BY_BACKBONE[backbone]
    return (IMAGE_HEIGHT, IMAGE_WIDTH)


def _resolve_batch_size(stage: str, backbone: str) -> int:
    if stage == "finetune":
        return int(FT_BATCH_SIZE_BY_BACKBONE.get(backbone, FT_BATCH_SIZE))
    if stage == "lora":
        return int(LORA_BATCH_SIZE_BY_BACKBONE.get(backbone, LORA_BATCH_SIZE))
    raise ValueError(f"Unknown stage: {stage!r}")


def _triplet_bool(name: str, value: Any) -> Tuple[bool, bool, bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list/tuple of booleans, got {value!r}")
    return (bool(value[0]), bool(value[1]), bool(value[2]))


def _apply_pipeline_yaml(path: Path) -> None:
    import yaml

    path = path.expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}")

    g = globals()

    paths = raw.get("paths") or {}
    if isinstance(paths, dict):
        if paths.get("spair_root") is not None:
            g["SPAIR_ROOT"] = str(paths["spair_root"])
        if paths.get("checkpoint_dir") is not None:
            g["CHECKPOINT_DIR"] = str(paths["checkpoint_dir"])

    runtime = raw.get("runtime") or {}
    if isinstance(runtime, dict):
        if "device" in runtime:
            g["DEVICE"] = runtime["device"]
        if runtime.get("precision") is not None:
            g["PRECISION"] = _normalize_precision_token(runtime["precision"])
        nw = runtime.get("num_workers")
        if nw is not None:
            g["NUM_WORKERS"] = None if nw in (-1, None) else int(nw)
        if runtime.get("preprocess") is not None:
            g["PREPROCESS"] = str(runtime["preprocess"])
        if runtime.get("image_height") is not None:
            g["IMAGE_HEIGHT"] = int(runtime["image_height"])
        if runtime.get("image_width") is not None:
            g["IMAGE_WIDTH"] = int(runtime["image_width"])
        if runtime.get("image_size_by_backbone") is not None:
            raw_map = runtime["image_size_by_backbone"]
            if isinstance(raw_map, dict):
                current = dict(g.get("IMAGE_SIZE_BY_BACKBONE", {}))
                for k, v in raw_map.items():
                    bk = str(k).strip()
                    if bk not in BACKBONE_NAMES:
                        raise ValueError(f"unknown backbone {bk!r} in image_size_by_backbone.")
                    if not (isinstance(v, (list, tuple)) and len(v) == 2):
                        raise ValueError(f"image_size_by_backbone[{bk!r}] must be [h, w].")
                    current[bk] = (int(v[0]), int(v[1]))
                g["IMAGE_SIZE_BY_BACKBONE"] = current
        if runtime.get("limit_pairs") is not None:
            g["EVAL_LIMIT"] = int(runtime["limit_pairs"])
        if runtime.get("alphas") is not None:
            g["EVAL_ALPHAS"] = tuple(float(a) for a in runtime["alphas"])
        if runtime.get("wsa_window") is not None:
            g["WSA_WINDOW"] = int(runtime["wsa_window"])
        if runtime.get("wsa_temperature") is not None:
            g["WSA_TEMPERATURE"] = float(runtime["wsa_temperature"])
        if runtime.get("log_batch_interval") is not None:
            g["LOG_BATCH_INTERVAL"] = int(runtime["log_batch_interval"])
        if runtime.get("resume_save_interval") is not None:
            g["RESUME_SAVE_INTERVAL"] = int(runtime["resume_save_interval"])
        if runtime.get("dino_layer_indices") is not None:
            g["DINO_LAYER_INDICES"] = int(runtime["dino_layer_indices"])
        if runtime.get("eval_split") is not None:
            g["EVAL_SPLIT"] = str(runtime["eval_split"])

    finetune = raw.get("finetune") or {}
    if isinstance(finetune, dict):
        lb = finetune.get("last_blocks")
        if lb is not None:
            g["LAST_BLOCKS_LIST"] = [int(x) for x in (lb if isinstance(lb, (list, tuple)) else [lb])]
        if finetune.get("epochs") is not None:
            g["FT_EPOCHS"] = int(finetune["epochs"])
        if finetune.get("patience") is not None:
            g["FT_PATIENCE"] = int(finetune["patience"])
        if finetune.get("lr") is not None:
            g["FT_LR"] = float(finetune["lr"])
        if finetune.get("weight_decay") is not None:
            g["FT_WEIGHT_DECAY"] = float(finetune["weight_decay"])
        if finetune.get("dinov2_weights") is not None:
            g["DINOV2_WEIGHTS"] = finetune["dinov2_weights"]
        if finetune.get("dinov3_weights") is not None:
            g["DINOV3_WEIGHTS"] = finetune["dinov3_weights"]
        if finetune.get("sam_checkpoint") is not None:
            g["SAM_CHECKPOINT"] = str(finetune["sam_checkpoint"])
        if finetune.get("batch_size") is not None:
            g["FT_BATCH_SIZE"] = int(finetune["batch_size"])
        if finetune.get("batch_size_by_backbone") is not None:
            g["FT_BATCH_SIZE_BY_BACKBONE"] = _parse_backbone_int_map(
                "finetune.batch_size_by_backbone", finetune["batch_size_by_backbone"]
            )

    lora = raw.get("lora") or {}
    if isinstance(lora, dict):
        if lora.get("epochs") is not None:
            g["LORA_EPOCHS"] = int(lora["epochs"])
        if lora.get("patience") is not None:
            g["LORA_PATIENCE"] = int(lora["patience"])
        if lora.get("rank") is not None:
            g["LORA_RANK"] = int(lora["rank"])
        if lora.get("alpha") is not None:
            g["LORA_ALPHA"] = float(lora["alpha"])
        if lora.get("lr") is not None:
            g["LORA_LR"] = float(lora["lr"])
        if lora.get("batch_size") is not None:
            g["LORA_BATCH_SIZE"] = int(lora["batch_size"])
        if lora.get("last_blocks") is not None:
            g["LORA_LAST_BLOCKS"] = int(lora["last_blocks"])
        if lora.get("batch_size_by_backbone") is not None:
            g["LORA_BATCH_SIZE_BY_BACKBONE"] = _parse_backbone_int_map(
                "lora.batch_size_by_backbone", lora["batch_size_by_backbone"]
            )

    toggles = raw.get("workflow_toggles") or {}
    if isinstance(toggles, dict):
        if "pipeline_resume" in toggles:
            g["PIPELINE_RESUME"] = bool(toggles["pipeline_resume"])
        if "run_verify_dataset" in toggles:
            g["RUN_VERIFY_DATASET"] = bool(toggles["run_verify_dataset"])
        for key, dest in (
            ("train_finetune", "TRAIN_FINETUNE"),
            ("train_lora", "TRAIN_LORA"),
            ("run_eval_baseline", "RUN_EVAL_BASELINE"),
            ("run_eval_baseline_wsa", "RUN_EVAL_BASELINE_WSA"),
            ("run_eval_finetuned_checkpoint", "RUN_EVAL_FINETUNED_CHECKPOINT"),
            ("run_eval_lora_checkpoint", "RUN_EVAL_LORA_CHECKPOINT"),
            ("run_eval_finetuned_wsa", "RUN_EVAL_FINETUNED_WSA"),
            ("run_eval_lora_wsa", "RUN_EVAL_LORA_WSA"),
        ):
            if key in toggles and toggles[key] is not None:
                g[dest] = _triplet_bool(dest, toggles[key])
        if "run_export_metrics_tables" in toggles:
            g["RUN_EXPORT_METRICS_TABLES"] = bool(toggles["run_export_metrics_tables"])
        if "run_pytest" in toggles:
            g["RUN_PYTEST"] = bool(toggles["run_pytest"])
    if raw.get("eval_split") is not None:
        g["EVAL_SPLIT"] = str(raw["eval_split"])


def _fingerprint_payload() -> Dict[str, Any]:
    return {
        "SPAIR_ROOT": SPAIR_ROOT,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "LAST_BLOCKS_LIST": LAST_BLOCKS_LIST,
        "LORA_RANK": LORA_RANK,
        "LORA_LAST_BLOCKS": LORA_LAST_BLOCKS,
        "FT_BATCH_SIZE": FT_BATCH_SIZE,
        "FT_EPOCHS": FT_EPOCHS,
        "FT_PATIENCE": FT_PATIENCE,
        "FT_LR": FT_LR,
        "FT_WEIGHT_DECAY": FT_WEIGHT_DECAY,
        "LORA_BATCH_SIZE": LORA_BATCH_SIZE,
        "FT_BATCH_SIZE_BY_BACKBONE": dict(sorted(FT_BATCH_SIZE_BY_BACKBONE.items())),
        "LORA_BATCH_SIZE_BY_BACKBONE": dict(sorted(LORA_BATCH_SIZE_BY_BACKBONE.items())),
        "LORA_EPOCHS": LORA_EPOCHS,
        "LORA_PATIENCE": LORA_PATIENCE,
        "LORA_LR": LORA_LR,
        "LORA_ALPHA": LORA_ALPHA,
        "PRECISION": PRECISION,
        "RESUME_SAVE_INTERVAL": RESUME_SAVE_INTERVAL,
        "PREPROCESS": PREPROCESS,
        "IMAGE_SIZE_BY_BACKBONE": {k: list(v) for k, v in sorted(IMAGE_SIZE_BY_BACKBONE.items())},
        "IMAGE_HEIGHT": IMAGE_HEIGHT,
        "IMAGE_WIDTH": IMAGE_WIDTH,
        "LOG_BATCH_INTERVAL": LOG_BATCH_INTERVAL,
        "DINO_LAYER_INDICES": DINO_LAYER_INDICES,
        "DINOV2_WEIGHTS": DINOV2_WEIGHTS,
        "DINOV3_WEIGHTS": DINOV3_WEIGHTS,
        "SAM_CHECKPOINT": SAM_CHECKPOINT,
        "EVAL_SPLIT": EVAL_SPLIT,
        "EVAL_LIMIT": EVAL_LIMIT,
        "EVAL_ALPHAS": list(EVAL_ALPHAS),
        "WSA_WINDOW": WSA_WINDOW,
        "WSA_TEMPERATURE": WSA_TEMPERATURE,
        "PIPELINE_RESUME": PIPELINE_RESUME,
        "RUN_VERIFY_DATASET": RUN_VERIFY_DATASET,
        "TRAIN_FINETUNE": list(TRAIN_FINETUNE),
        "TRAIN_LORA": list(TRAIN_LORA),
        "RUN_EVAL_BASELINE": list(RUN_EVAL_BASELINE),
        "RUN_EVAL_BASELINE_WSA": list(RUN_EVAL_BASELINE_WSA),
        "RUN_EVAL_FINETUNED_CHECKPOINT": list(RUN_EVAL_FINETUNED_CHECKPOINT),
        "RUN_EVAL_LORA_CHECKPOINT": list(RUN_EVAL_LORA_CHECKPOINT),
        "RUN_EVAL_FINETUNED_WSA": list(RUN_EVAL_FINETUNED_WSA),
        "RUN_EVAL_LORA_WSA": list(RUN_EVAL_LORA_WSA),
        "RUN_EXPORT_METRICS_TABLES": RUN_EXPORT_METRICS_TABLES,
        "RUN_PYTEST": RUN_PYTEST,
    }


def _trace_stage(cwd: Path, logger: "PipelineLogger", action: str, stage_id: str, **extra: Any) -> None:
    from utils.pipeline_state import append_stage_event

    tail = " ".join(f"{k}={extra[k]!r}" for k in sorted(extra))
    logger.log_line(f"[STAGE] action={action} stage_id={stage_id}" + (f" {tail}" if tail else ""))
    append_stage_event(cwd, {"action": action, "stage_id": stage_id, **extra})


def _pipeline_log_file_only() -> bool:
    v = os.environ.get("SEMANTIC_CORRESPONDENCE_PIPELINE_LOG_FILE_ONLY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class PipelineLogger:
    """Timestamped lines to a per-run log file; optionally mirrors to stdout."""

    def __init__(self, path: Path, *, mirror_to_terminal: bool = True) -> None:
        self.path = path.resolve()
        self._mirror = mirror_to_terminal
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", encoding="utf-8")
        self._fp.write(f"# Pipeline run started at {self._ts()}\n")
        self._fp.write(f"# Log file: {self.path}\n\n")
        self._fp.flush()

    @staticmethod
    def _ts() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def log_line(self, line: str, *, err: bool = False) -> None:
        stream = sys.stderr if err else sys.stdout
        for part in line.split("\n"):
            prefixed = f"[{self._ts()}] {part}\n"
            self._fp.write(prefixed)
            if self._mirror:
                stream.write(prefixed)
        self._fp.flush()
        if self._mirror:
            stream.flush()

    def close(self, exit_code: int) -> None:
        if self._fp.closed:
            return
        self._fp.write(f"\n# Pipeline run finished at {self._ts()} exit_code={exit_code}\n")
        self._fp.close()


def _manifest_append(cwd: Path, row: List[str]) -> None:
    manifest = cwd / "runs" / "logs" / "manifest.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file():
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("timestamp_utc\tevent\tlog_filename\texit_code\n")
    with open(manifest, "a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def _link_current_log_symlink(cwd: Path, log_path: Path) -> None:
    link = cwd / "runs" / "logs" / "current.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(log_path.name)
    except OSError:
        pass


def _validate_triplet(name: str, t: Tuple[bool, ...]) -> None:
    if len(t) != 3:
        raise ValueError(f"{name} must have exactly 3 elements, got {len(t)}")


def _run_cmd(cwd: Path, cmd: List[str], logger: PipelineLogger) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(cwd)
    logger.log_line(f"--- Running: {' '.join(cmd)}")
    p = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    if p.stdout is None:
        return p.wait()
    for raw in p.stdout:
        logger.log_line(raw.rstrip("\n"))
    return p.wait()


def _run_script(cwd: Path, rel_script: str, args: Sequence[str], logger: PipelineLogger) -> int:
    return _run_cmd(cwd, [sys.executable, str(cwd / rel_script), *args], logger)


def _default_finetune_ckpt_for(cwd: Path, backbone: str, n_blocks: int) -> str:
    return str(cwd / CHECKPOINT_DIR / f"{backbone}_lastblocks{n_blocks}_best.pt")


def _default_lora_ckpt(cwd: Path, backbone: str) -> str:
    return str(cwd / CHECKPOINT_DIR / f"{backbone}_lora_r{LORA_RANK}_best.pt")


def _resolve_ckpt_path(cwd: Path, path_str: str) -> Optional[str]:
    p = Path(path_str)
    full = p if p.is_absolute() else cwd / p
    return str(full) if full.is_file() else None


def _build_eval_specs(
    cwd: Path, sam_checkpoint: Optional[str], logger: PipelineLogger, *, num_workers: int,
) -> List[Any]:
    from evaluation.experiment_runner import EvalRunSpec

    specs: List[EvalRunSpec] = []

    def _base_kwargs(backbone: str) -> Dict[str, Any]:
        h, w = _resolve_image_hw(backbone)
        return dict(
            backbone=backbone,
            split=EVAL_SPLIT,
            dinov2_weights=DINOV2_WEIGHTS,
            dinov3_weights=DINOV3_WEIGHTS,
            sam_checkpoint=sam_checkpoint,
            limit=EVAL_LIMIT,
            preprocess=PREPROCESS,
            height=h,
            width=w,
            num_workers=num_workers,
        )

    for idx, backbone in enumerate(BACKBONE_NAMES):
        any_sam_flag = any(
            t[idx] for t in (
                RUN_EVAL_BASELINE, RUN_EVAL_BASELINE_WSA, RUN_EVAL_FINETUNED_CHECKPOINT,
                RUN_EVAL_LORA_CHECKPOINT, RUN_EVAL_FINETUNED_WSA, RUN_EVAL_LORA_WSA,
            )
        )
        if backbone == "sam_vit_b" and not sam_checkpoint:
            if any_sam_flag:
                logger.log_line(f"WARNING: skipping SAM eval for {backbone}: set SAM_CHECKPOINT.", err=True)
            continue

        bk = _base_kwargs(backbone)

        if RUN_EVAL_BASELINE[idx]:
            specs.append(EvalRunSpec(name=f"{backbone}_baseline", **bk))
        if RUN_EVAL_BASELINE_WSA[idx]:
            specs.append(EvalRunSpec(
                name=f"{backbone}_baseline_wsa",
                use_window_soft_argmax=True,
                wsa_window=WSA_WINDOW,
                wsa_temperature=WSA_TEMPERATURE,
                **bk,
            ))

        for nb in LAST_BLOCKS_LIST:
            ck_path = _default_finetune_ckpt_for(cwd, backbone, nb)
            ck = _resolve_ckpt_path(cwd, ck_path)
            if RUN_EVAL_FINETUNED_CHECKPOINT[idx]:
                if not ck:
                    logger.log_line(f"WARNING: fine-tuned checkpoint not found: {ck_path}", err=True)
                else:
                    specs.append(EvalRunSpec(name=f"{backbone}_ft_lb{nb}", checkpoint=ck, **bk))
            if RUN_EVAL_FINETUNED_WSA[idx] and ck:
                specs.append(EvalRunSpec(
                    name=f"{backbone}_ft_lb{nb}_wsa",
                    checkpoint=ck,
                    use_window_soft_argmax=True,
                    wsa_window=WSA_WINDOW,
                    wsa_temperature=WSA_TEMPERATURE,
                    **bk,
                ))

        lora_ck_path = _default_lora_ckpt(cwd, backbone)
        lora_ck = _resolve_ckpt_path(cwd, lora_ck_path)
        if RUN_EVAL_LORA_CHECKPOINT[idx]:
            if not lora_ck:
                logger.log_line(f"WARNING: LoRA checkpoint not found: {lora_ck_path}", err=True)
            else:
                specs.append(EvalRunSpec(name=f"{backbone}_lora", checkpoint=lora_ck, **bk))
        if RUN_EVAL_LORA_WSA[idx] and lora_ck:
            specs.append(EvalRunSpec(
                name=f"{backbone}_lora_wsa",
                checkpoint=lora_ck,
                use_window_soft_argmax=True,
                wsa_window=WSA_WINDOW,
                wsa_temperature=WSA_TEMPERATURE,
                **bk,
            ))

    return specs


def _export_tables(cwd: Path, results: Sequence[Dict[str, Any]], logger: PipelineLogger) -> None:
    from evaluation.experiment_runner import metrics_rows_for_table

    out_dir = cwd / "runs" / "pipeline_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics_rows_for_table(results)
    json_path = out_dir / "pck_results.json"
    csv_path = out_dir / "pck_results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    by_difficulty_flag: List[Dict[str, Any]] = []
    per_category: List[Dict[str, Any]] = []
    for r in results:
        if "sd4match_by_difficulty_flag" in r:
            by_difficulty_flag.append({
                "name": r.get("name"),
                "split": r.get("spec", {}).get("split"),
                "data": r["sd4match_by_difficulty_flag"],
            })

        pi = r.get("sd4match_per_image")
        pp = r.get("sd4match_per_point")
        if not pi:
            continue
        entry: Dict[str, Any] = {"name": r.get("name"), "categories": {}}
        for metric_key, cat_dict in pi.items():
            if not metric_key.startswith("custom_pck"):
                continue
            alpha_str = metric_key.replace("custom_pck", "")
            for cat, vals in cat_dict.items():
                if cat == "all":
                    continue
                entry["categories"].setdefault(cat, {})
                if isinstance(vals, list) and vals:
                    entry["categories"][cat][f"pck@{alpha_str}"] = float(sum(vals) / len(vals))
        if pp:
            for metric_key, cat_dict in pp.items():
                if not metric_key.startswith("custom_pck"):
                    continue
                alpha_str = metric_key.replace("custom_pck", "")
                for cat, vals in cat_dict.items():
                    if cat == "all":
                        continue
                    entry["categories"].setdefault(cat, {})
                    if isinstance(vals, list) and vals:
                        entry["categories"][cat][f"pck_pt@{alpha_str}"] = float(sum(vals) / len(vals))
        per_category.append(entry)

    wrote = [str(json_path), str(csv_path)]
    if by_difficulty_flag:
        by_path = out_dir / "pck_results_by_difficulty_flag.json"
        with open(by_path, "w", encoding="utf-8") as f:
            json.dump(by_difficulty_flag, f, indent=2)
        wrote.append(str(by_path))
    if per_category:
        per_path = out_dir / "pck_results_per_category.json"
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(per_category, f, indent=2)
        wrote.append(str(per_path))
        all_cats = sorted({cat for entry in per_category for cat in entry["categories"]})
        per_csv = out_dir / "pck_results_per_category.csv"
        with open(per_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "metric"] + all_cats)
            w.writeheader()
            for entry in per_category:
                for mk in ["pck@0.05", "pck@0.1", "pck@0.2", "pck_pt@0.05", "pck_pt@0.1", "pck_pt@0.2"]:
                    row: Dict[str, Any] = {"name": entry["name"], "metric": mk}
                    for cat in all_cats:
                        row[cat] = entry["categories"].get(cat, {}).get(mk, "")
                    w.writerow(row)
        wrote.append(str(per_csv))

    logger.log_line("Wrote " + " | ".join(wrote) + ".")


def _run_eval_and_export(
    cwd: Path, sam_checkpoint: Optional[str], logger: PipelineLogger,
    *, num_workers: int, device_str: str,
) -> int:
    import torch
    from evaluation.experiment_runner import run_comparison_batch

    specs = _build_eval_specs(cwd, sam_checkpoint, logger, num_workers=num_workers)
    if not specs:
        logger.log_line("No PCK eval specs selected.")
        return 0

    results = run_comparison_batch(
        specs, spair_root=SPAIR_ROOT, alphas=EVAL_ALPHAS, device=torch.device(device_str),
    )
    for r in results:
        logger.log_line(f"--- {r['name']} --- {r['metrics']}")
    if RUN_EXPORT_METRICS_TABLES:
        _export_tables(cwd, results, logger)
    return 0


def _sam_slot_used() -> bool:
    return any(
        t[2] for t in (
            TRAIN_FINETUNE, TRAIN_LORA,
            RUN_EVAL_BASELINE, RUN_EVAL_BASELINE_WSA,
            RUN_EVAL_FINETUNED_CHECKPOINT, RUN_EVAL_LORA_CHECKPOINT,
            RUN_EVAL_FINETUNED_WSA, RUN_EVAL_LORA_WSA,
        )
    )


def _pipeline_run(cwd: Path, logger: PipelineLogger) -> int:
    from utils.hardware import recommended_dataloader_workers, recommended_device_str
    from utils import pipeline_state as _ps

    for name, t in (
        ("TRAIN_FINETUNE", TRAIN_FINETUNE), ("TRAIN_LORA", TRAIN_LORA),
        ("RUN_EVAL_BASELINE", RUN_EVAL_BASELINE), ("RUN_EVAL_BASELINE_WSA", RUN_EVAL_BASELINE_WSA),
        ("RUN_EVAL_FINETUNED_CHECKPOINT", RUN_EVAL_FINETUNED_CHECKPOINT),
        ("RUN_EVAL_LORA_CHECKPOINT", RUN_EVAL_LORA_CHECKPOINT),
        ("RUN_EVAL_FINETUNED_WSA", RUN_EVAL_FINETUNED_WSA),
        ("RUN_EVAL_LORA_WSA", RUN_EVAL_LORA_WSA),
    ):
        _validate_triplet(name, t)

    eff_device = DEVICE if DEVICE is not None else recommended_device_str()
    eff_precision = _normalize_precision_token(PRECISION)
    eff_workers = NUM_WORKERS if NUM_WORKERS is not None else recommended_dataloader_workers(
        accelerator=eff_device
    )
    logger.log_line(
        f"Adaptive hardware: device={eff_device} precision={eff_precision} "
        f"num_workers={eff_workers} cpu_count={os.cpu_count()} platform={sys.platform}"
    )

    fp = _ps.fingerprint_from_config(_fingerprint_payload())
    if not PIPELINE_RESUME:
        state = {"fingerprint": fp, "completed": []}
        logger.log_line("PIPELINE_RESUME=False: every enabled stage runs.")
    elif _ps.should_reset_from_env():
        state = {"fingerprint": fp, "completed": []}
        _ps.save_state(cwd, state)
        logger.log_line("pipeline_state: cleared (SEMANTIC_CORRESPONDENCE_PIPELINE_RESET set).")
        _trace_stage(cwd, logger, "reset", "pipeline_state", fingerprint_prefix=fp[:16])
    else:
        raw = _ps.load_state(cwd)
        if raw is None:
            state = {"fingerprint": fp, "completed": []}
            logger.log_line(f"pipeline_state: no prior file (will write {_ps.state_path(cwd)}).")
        elif raw.get("fingerprint") != fp:
            state = dict(raw)
            state["completed"] = list(state.get("completed", []))
            state["fingerprint"] = fp
            _ps.save_state(cwd, state)
            logger.log_line("pipeline_state: config fingerprint changed; completed steps preserved.")
        else:
            state = dict(raw)
            state["completed"] = list(state.get("completed", []))
            state["fingerprint"] = fp
            logger.log_line(f"pipeline_state: completed steps = {state['completed']!r}")
    if PIPELINE_RESUME:
        logger.log_line(f"Tracing: {_ps.stage_events_path(cwd)} | state {_ps.state_path(cwd)}")

    effective_sam = SAM_CHECKPOINT or os.environ.get("SAM_CHECKPOINT")
    if _sam_slot_used() and not effective_sam:
        logger.log_line(
            "ERROR: SAM is enabled but SAM_CHECKPOINT is not set.\n"
            "  Set SAM_CHECKPOINT in run_pipeline.py or export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth",
            err=True,
        )
        return 2

    if RUN_VERIFY_DATASET:
        v_sid = "verify_dataset"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), v_sid):
            logger.log_line(f"[SKIP] stage_id={v_sid}")
            _trace_stage(cwd, logger, "skip", v_sid, reason="already_completed")
        else:
            _trace_stage(cwd, logger, "start", v_sid)
            rc = _run_script(
                cwd, "scripts/verify_dataset.py",
                [] if SPAIR_ROOT is None else ["--spair-root", SPAIR_ROOT],
                logger,
            )
            if rc != 0:
                _trace_stage(cwd, logger, "fail", v_sid, exit_code=rc)
                return rc
            if PIPELINE_RESUME:
                _ps.mark_step_done(cwd, state, v_sid)
            _trace_stage(cwd, logger, "done", v_sid, exit_code=0)

    base_train: List[str] = []
    if SPAIR_ROOT:
        base_train.extend(["--spair-root", SPAIR_ROOT])
    base_train.extend([
        "--preprocess", PREPROCESS,
        "--num-workers", str(eff_workers),
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--device", eff_device,
        "--precision", eff_precision,
        "--resume-save-interval", str(RESUME_SAVE_INTERVAL),
        "--log-batch-interval", str(LOG_BATCH_INTERVAL),
    ])
    if DINOV2_WEIGHTS:
        base_train.extend(["--dinov2-weights", DINOV2_WEIGHTS])
    if DINOV3_WEIGHTS:
        base_train.extend(["--dinov3-weights", DINOV3_WEIGHTS])
    if effective_sam:
        base_train.extend(["--sam-checkpoint", effective_sam])

    for i, backbone in enumerate(BACKBONE_NAMES):
        if not TRAIN_FINETUNE[i]:
            continue
        if backbone == "sam_vit_b" and not effective_sam:
            logger.log_line("ERROR: TRAIN_FINETUNE for SAM requires SAM_CHECKPOINT.", err=True)
            return 2
        for nb in LAST_BLOCKS_LIST:
            ft_bs = _resolve_batch_size("finetune", backbone)
            ft_sid = f"finetune:{backbone}:lb{nb}"
            if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), ft_sid):
                logger.log_line(f"[SKIP] stage_id={ft_sid}")
                _trace_stage(cwd, logger, "skip", ft_sid, reason="already_completed")
                continue
            ft_h, ft_w = _resolve_image_hw(backbone)
            args = [
                "--mode", "finetune",
                *base_train,
                "--height", str(ft_h),
                "--width", str(ft_w),
                "--batch-size", str(ft_bs),
                "--lr", str(FT_LR),
                "--weight-decay", str(FT_WEIGHT_DECAY),
                "--backbone", backbone,
                "--epochs", str(FT_EPOCHS),
                "--patience", str(FT_PATIENCE),
                "--last-blocks", str(nb),
                "--layer-indices", str(DINO_LAYER_INDICES),
            ]
            logger.log_line(f"stage_id={ft_sid} batch_size={ft_bs} precision={eff_precision}")
            resume_file = (cwd / CHECKPOINT_DIR / f"{backbone}_lastblocks{nb}_resume.pt").resolve()
            if PIPELINE_RESUME and resume_file.is_file():
                args.extend(["--resume", str(resume_file)])
                logger.log_line(f"[RESUME] stage_id={ft_sid} checkpoint={resume_file}")
                _trace_stage(cwd, logger, "resume_prepare", ft_sid, path=str(resume_file))
            _trace_stage(cwd, logger, "start", ft_sid)
            rc = _run_script(cwd, "scripts/train.py", args, logger)
            if rc != 0:
                _trace_stage(cwd, logger, "fail", ft_sid, exit_code=rc)
                return rc
            if PIPELINE_RESUME:
                _ps.mark_step_done(cwd, state, ft_sid)
            _trace_stage(cwd, logger, "done", ft_sid, exit_code=0)

    for i, backbone in enumerate(BACKBONE_NAMES):
        if not TRAIN_LORA[i]:
            continue
        if backbone == "sam_vit_b" and not effective_sam:
            logger.log_line("ERROR: TRAIN_LORA for SAM requires SAM_CHECKPOINT.", err=True)
            return 2
        lora_bs = _resolve_batch_size("lora", backbone)
        lora_sid = f"lora:{backbone}"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), lora_sid):
            logger.log_line(f"[SKIP] stage_id={lora_sid}")
            _trace_stage(cwd, logger, "skip", lora_sid, reason="already_completed")
            continue
        lora_h, lora_w = _resolve_image_hw(backbone)
        args = [
            "--mode", "lora",
            *base_train,
            "--height", str(lora_h),
            "--width", str(lora_w),
            "--batch-size", str(lora_bs),
            "--lr", str(LORA_LR),
            "--alpha", str(LORA_ALPHA),
            "--backbone", backbone,
            "--epochs", str(LORA_EPOCHS),
            "--patience", str(LORA_PATIENCE),
            "--last-blocks", str(LORA_LAST_BLOCKS),
            "--rank", str(LORA_RANK),
            "--layer-indices", str(DINO_LAYER_INDICES),
        ]
        logger.log_line(f"stage_id={lora_sid} batch_size={lora_bs} precision={eff_precision}")
        resume_lora = (cwd / CHECKPOINT_DIR / f"{backbone}_lora_r{LORA_RANK}_resume.pt").resolve()
        if PIPELINE_RESUME and resume_lora.is_file():
            args.extend(["--resume", str(resume_lora)])
            logger.log_line(f"[RESUME] stage_id={lora_sid} checkpoint={resume_lora}")
            _trace_stage(cwd, logger, "resume_prepare", lora_sid, path=str(resume_lora))
        _trace_stage(cwd, logger, "start", lora_sid)
        rc = _run_script(cwd, "scripts/train.py", args, logger)
        if rc != 0:
            _trace_stage(cwd, logger, "fail", lora_sid, exit_code=rc)
            return rc
        if PIPELINE_RESUME:
            _ps.mark_step_done(cwd, state, lora_sid)
        _trace_stage(cwd, logger, "done", lora_sid, exit_code=0)

    any_eval = any(any(t) for t in (
        RUN_EVAL_BASELINE, RUN_EVAL_BASELINE_WSA,
        RUN_EVAL_FINETUNED_CHECKPOINT, RUN_EVAL_LORA_CHECKPOINT,
        RUN_EVAL_FINETUNED_WSA, RUN_EVAL_LORA_WSA,
    ))
    if any_eval:
        ev_sid = "eval_and_export"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), ev_sid):
            logger.log_line(f"[SKIP] stage_id={ev_sid}")
            _trace_stage(cwd, logger, "skip", ev_sid, reason="already_completed")
        else:
            _trace_stage(cwd, logger, "start", ev_sid)
            rc = _run_eval_and_export(
                cwd, effective_sam, logger, num_workers=eff_workers, device_str=eff_device,
            )
            if rc != 0:
                _trace_stage(cwd, logger, "fail", ev_sid, exit_code=rc)
                return rc
            if PIPELINE_RESUME:
                _ps.mark_step_done(cwd, state, ev_sid)
            _trace_stage(cwd, logger, "done", ev_sid, exit_code=0)

    if RUN_PYTEST:
        py_sid = "pytest"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), py_sid):
            logger.log_line(f"[SKIP] stage_id={py_sid}")
            _trace_stage(cwd, logger, "skip", py_sid, reason="already_completed")
        else:
            _trace_stage(cwd, logger, "start", py_sid)
            rc = _run_cmd(cwd, [sys.executable, "-m", "pytest", "-q", "tests"], logger)
            if rc != 0:
                _trace_stage(cwd, logger, "fail", py_sid, exit_code=rc)
                return rc
            if PIPELINE_RESUME:
                _ps.mark_step_done(cwd, state, py_sid)
            _trace_stage(cwd, logger, "done", py_sid, exit_code=0)

    logger.log_line("Pipeline finished.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run semantic-correspondence pipeline.")
    parser.add_argument("--config", metavar="PATH", default=None, help="Optional YAML file.")
    args = parser.parse_args()

    cwd = _REPO
    os.chdir(cwd)
    if args.config:
        cfg_path = Path(args.config).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (cwd / cfg_path).resolve()
        _apply_pipeline_yaml(cfg_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = cwd / "runs" / "logs" / f"pipeline_{stamp}.log"
    logger = PipelineLogger(log_path, mirror_to_terminal=not _pipeline_log_file_only())
    _link_current_log_symlink(cwd, log_path)
    iso_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _manifest_append(cwd, [iso_start, "start", log_path.name, ""])
    exit_code = 1
    try:
        exit_code = _pipeline_run(cwd, logger)
    finally:
        logger.close(exit_code)
        iso_end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _manifest_append(cwd, [iso_end, "end", log_path.name, str(exit_code)])
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
