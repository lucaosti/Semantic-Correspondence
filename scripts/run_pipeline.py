#!/usr/bin/env python3
"""
Run project steps from one place: verify data, train backbones, evaluate PCK, export tables.

Edit the **boolean flags and 3-element tuples** in the configuration block below, then run from
the repository root with the venv active and ``pip install -e .``:

.. code-block:: bash

   python scripts/run_pipeline.py
   python scripts/run_pipeline.py --config config.yaml

The optional ``--config`` flag loads a YAML file (same structure as the Colab ``AML.ipynb`` notebook)
and overrides the in-script defaults without editing this file.

Order of execution: dataset verification (optional) → fine-tune (per backbone flag) → LoRA (per
flag) → PCK evaluation and optional exports → optional pytest → optional notebook hint.

The three tuple slots always mean **(DINOv2 ViT-B/14, DINOv3 ViT-B/16, SAM ViT-B)**.

Defaults run **all** steps for **all** backbones. SAM needs weights: place
``checkpoints/sam_vit_b_01ec64.pth`` (run ``bash scripts/download_sam_vit_b.sh``), set
``SAM_CHECKPOINT`` below, or export ``SAM_CHECKPOINT``.

Logs: each invocation writes ``runs/logs/pipeline_<UTC_stamp>.log``, updates ``runs/logs/manifest.tsv``,
and sets ``runs/logs/current.log`` (symlink) to that file. Subprocess output is merged into the same file (see
:class:`PipelineLogger`). With ``SEMANTIC_CORRESPONDENCE_PIPELINE_LOG_FILE_ONLY=1`` (set by
``scripts/start_pipeline_detached.sh``), nothing is mirrored to the process stdout/stderr—use
``scripts/reconnect_dashboard.sh`` or ``tail -f runs/logs/current.log`` to watch progress.

Structured stage traces append to ``runs/logs/stage_events.jsonl`` (one JSON object per line: ``action``, ``stage_id``, ``ts_utc``, …).

**Resume / interrupt:** Set ``PIPELINE_RESUME = True`` (default) to skip stages already recorded in ``runs/pipeline_state.json`` and to pass ``--resume`` into training when a ``*_resume.pt`` file exists (optimizer + model state). Training scripts also write that file **during** an epoch every ``--resume-save-interval`` batches (default 2500), not only after validation, so long single epochs are not lost on kill. Changing training/eval hyperparameters changes a **config fingerprint**; a mismatch clears remembered progress so runs are not mixed. For a deliberate full restart without deleting checkpoints, run with env ``SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1``.

**Epochs** are set by ``FT_EPOCHS`` / ``LORA_EPOCHS`` (and ``FT_PATIENCE`` / ``LORA_PATIENCE`` for early stopping) in the SHARED CONFIG block below. Training scripts also print ``epoch k/total``, periodic batch indices, and ``train_loss`` / ``val_loss`` per epoch.

**Hardware:** training defaults to **batch size 100** pairs per step (Gaussian loss averages over the batch). Set ``DEVICE`` / ``NUM_WORKERS`` to ``None`` for adaptive defaults (CUDA → MPS → CPU; worker count scales with logical CPUs and accelerator—see :func:`utils.hardware.recommended_dataloader_workers`). CUDA runs enable cuDNN benchmark and TF32 in training/eval scripts. Pinned memory when ``device=cuda``; persistent DataLoader workers when ``num_workers > 0``. Override with ``TRAIN_BATCH_SIZE`` or ``--batch-size`` in the training CLIs.
"""

from __future__ import annotations

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
# PIPELINE TOGGLES — edit True/False only (or tuples of three bools)
# =============================================================================

# Skip pipeline stages already finished (see ``runs/pipeline_state.json``); resume training epochs
# via ``*_resume.pt`` when present. Set False to always run every stage from the top.
PIPELINE_RESUME: bool = True

# Run ``scripts/verify_dataset.py`` before anything else (recommended).
RUN_VERIFY_DATASET: bool = True

# Fine-tune last transformer blocks (Task 2): one flag per backbone.
TRAIN_FINETUNE: Tuple[bool, bool, bool] = (True, True, True)

# LoRA on last MLP linears (Task 4): one flag per backbone.
TRAIN_LORA: Tuple[bool, bool, bool] = (True, True, True)

# Baseline PCK evaluation (no checkpoint) via ``evaluation.experiment_runner`` (same as CLI eval).
RUN_EVAL_BASELINE: Tuple[bool, bool, bool] = (True, True, True)

# Same, but with window soft-argmax (Task 3, inference-only).
RUN_EVAL_BASELINE_WSA: Tuple[bool, bool, bool] = (True, True, True)

# If True, add an eval row loading a fine-tuned checkpoint for that backbone (path below).
RUN_EVAL_FINETUNED_CHECKPOINT: Tuple[bool, bool, bool] = (True, True, True)

# If True, add an eval row loading a LoRA checkpoint for that backbone (path below).
RUN_EVAL_LORA_CHECKPOINT: Tuple[bool, bool, bool] = (True, True, True)

# Export PCK results to ``runs/pipeline_exports/`` (JSON + CSV) when any eval step runs.
RUN_EXPORT_METRICS_TABLES: bool = True

# Run unit tests (requires ``pip install -e ".[dev]"``).
RUN_PYTEST: bool = True

# Print how to open the comparison notebook (no subprocess).
PRINT_JUPYTER_NOTEBOOK_HINT: bool = True

# =============================================================================
# SHARED CONFIG — paths, training args, evaluation split
# =============================================================================

# ``None`` → ``data/paths.resolve_spair_root()`` (env ``SPAIR_ROOT`` / default under ``data/``).
SPAIR_ROOT: Optional[str] = None

# Dataset/metrics backend selection (PDF-first portability).
# - dataset backend controls how samples are loaded
# - metrics backend controls how PCK is aggregated/reported
DATASET_BACKEND: str = "sd4match"  # "sd4match" | "native"
METRICS_BACKEND: str = "sd4match"  # "sd4match" | "native"

CHECKPOINT_DIR: str = "checkpoints"
LAST_BLOCKS: int = 2
LORA_RANK: int = 8

# Optional official weight files. SAM: defaults to ``checkpoints/sam_vit_b_01ec64.pth`` if that
# file exists; else set this or ``export SAM_CHECKPOINT=...`` when the SAM slot is enabled.
DINOV2_WEIGHTS: Optional[str] = None
DINOV3_WEIGHTS: Optional[str] = None
SAM_CHECKPOINT: Optional[str] = str(_SAM_VIT_B_DEFAULT) if _SAM_VIT_B_DEFAULT.is_file() else None

# Training hyperparameters (passed through to ``train_finetune.py`` / ``train_lora.py``).
# 200 epochs per backbone for both fine-tune and LoRA (early stopping may finish sooner).
TRAIN_BATCH_SIZE: int = 100
FT_EPOCHS: int = 200
FT_PATIENCE: int = 10
LORA_EPOCHS: int = 200
LORA_PATIENCE: int = 10
# Training scripts print batch progress every N steps (0 = epoch summaries only).
# Lower on CPU-only hosts so logs/dashboard move often (each step is slow).
LOG_BATCH_INTERVAL: int = 100
PREPROCESS: str = "FIXED_RESIZE"
# Multiples of ViT patch sizes used here (14 and 16): ``784 = 56*14 = 49*16``.
IMAGE_HEIGHT: int = 784
IMAGE_WIDTH: int = 784
# ``None`` → :func:`utils.hardware.recommended_dataloader_workers`
NUM_WORKERS: Optional[int] = None
# ``None`` → CUDA if available, else Apple MPS, else CPU
DEVICE: Optional[str] = None

# Evaluation split: ``test`` for benchmark numbers (see ``docs/info.md``); use ``val`` for faster runs.
EVAL_SPLIT: str = "test"
EVAL_LIMIT: int = 0  # >0: only first N pairs per run (debug)
# PDF default thresholds.
EVAL_ALPHAS: Tuple[float, ...] = (0.05, 0.1, 0.2)

# Explicit checkpoint paths for eval rows (defaults match training script outputs).
# Set to a string path or leave None to use ``<CHECKPOINT_DIR>/<auto_name>`` when the flag above is True.
FINETUNED_CHECKPOINT_PATHS: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)
LORA_CHECKPOINT_PATHS: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)

# Window soft-argmax knobs (when RUN_EVAL_BASELINE_WSA is used).
WSA_WINDOW: int = 5
WSA_TEMPERATURE: float = 1.0

# =============================================================================
# Backbone order (do not change; matches tuple positions above)
# =============================================================================

BACKBONE_NAMES: Tuple[str, str, str] = ("dinov2_vitb14", "dinov3_vitb16", "sam_vit_b")


def _triplet_bool(name: str, value: Any) -> Tuple[bool, bool, bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a length-3 list/tuple of booleans (DINOv2, DINOv3, SAM), got {value!r}")
    return (bool(value[0]), bool(value[1]), bool(value[2]))


def _apply_pipeline_yaml(path: Path) -> None:
    """Load a Colab/notebook-style YAML file and override module-level pipeline settings."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dependency
        raise ImportError("PyYAML is required for --config (pip install pyyaml).") from exc

    path = path.expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a YAML mapping in {path}, got {type(raw).__name__}")

    g = globals()

    ds = raw.get("dataset") or {}
    if isinstance(ds, dict):
        if ds.get("backend") is not None:
            g["DATASET_BACKEND"] = str(ds["backend"])
        if ds.get("metrics_backend") is not None:
            g["METRICS_BACKEND"] = str(ds["metrics_backend"])

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
        nw = runtime.get("num_workers")
        if nw is not None:
            g["NUM_WORKERS"] = None if nw in (-1, None) else int(nw)
        if runtime.get("preprocess") is not None:
            g["PREPROCESS"] = str(runtime["preprocess"])
        if runtime.get("image_height") is not None:
            g["IMAGE_HEIGHT"] = int(runtime["image_height"])
        if runtime.get("image_width") is not None:
            g["IMAGE_WIDTH"] = int(runtime["image_width"])
        if runtime.get("limit_pairs") is not None:
            g["EVAL_LIMIT"] = int(runtime["limit_pairs"])
        if runtime.get("alphas") is not None:
            alphas = tuple(float(a) for a in runtime["alphas"])
            g["EVAL_ALPHAS"] = alphas
        if runtime.get("wsa_window") is not None:
            g["WSA_WINDOW"] = int(runtime["wsa_window"])
        if runtime.get("wsa_temperature") is not None:
            g["WSA_TEMPERATURE"] = float(runtime["wsa_temperature"])
        if runtime.get("log_batch_interval") is not None:
            g["LOG_BATCH_INTERVAL"] = int(runtime["log_batch_interval"])
        if runtime.get("eval_split") is not None:
            g["EVAL_SPLIT"] = str(runtime["eval_split"])

    finetune = raw.get("finetune") or {}
    if isinstance(finetune, dict):
        if finetune.get("last_blocks") is not None:
            g["LAST_BLOCKS"] = int(finetune["last_blocks"])
        if finetune.get("epochs") is not None:
            g["FT_EPOCHS"] = int(finetune["epochs"])
        if finetune.get("patience") is not None:
            g["FT_PATIENCE"] = int(finetune["patience"])
        if finetune.get("dinov2_weights") is not None:
            g["DINOV2_WEIGHTS"] = finetune["dinov2_weights"]
        if finetune.get("dinov3_weights") is not None:
            g["DINOV3_WEIGHTS"] = finetune["dinov3_weights"]
        if finetune.get("sam_checkpoint") is not None:
            g["SAM_CHECKPOINT"] = str(finetune["sam_checkpoint"])
        if finetune.get("batch_size") is not None:
            g["TRAIN_BATCH_SIZE"] = int(finetune["batch_size"])

    lora = raw.get("lora") or {}
    if isinstance(lora, dict):
        if lora.get("epochs") is not None:
            g["LORA_EPOCHS"] = int(lora["epochs"])
        if lora.get("patience") is not None:
            g["LORA_PATIENCE"] = int(lora["patience"])
        if lora.get("rank") is not None:
            g["LORA_RANK"] = int(lora["rank"])
        if lora.get("batch_size") is not None:
            g["TRAIN_BATCH_SIZE"] = int(lora["batch_size"])

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
        ):
            if key in toggles and toggles[key] is not None:
                g[dest] = _triplet_bool(dest, toggles[key])
        if "run_export_metrics_tables" in toggles:
            g["RUN_EXPORT_METRICS_TABLES"] = bool(toggles["run_export_metrics_tables"])
        if "run_pytest" in toggles:
            g["RUN_PYTEST"] = bool(toggles["run_pytest"])
        if "print_jupyter_notebook_hint" in toggles:
            g["PRINT_JUPYTER_NOTEBOOK_HINT"] = bool(toggles["print_jupyter_notebook_hint"])

    # Optional top-level eval split (some notebook YAMLs use experiments; we only need split)
    if raw.get("eval_split") is not None:
        g["EVAL_SPLIT"] = str(raw["eval_split"])


def _fingerprint_payload() -> Dict[str, Any]:
    """Serializable config for :func:`utils.pipeline_state.fingerprint_from_config`."""
    return {
        "SPAIR_ROOT": SPAIR_ROOT,
        "DATASET_BACKEND": DATASET_BACKEND,
        "METRICS_BACKEND": METRICS_BACKEND,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "LAST_BLOCKS": LAST_BLOCKS,
        "LORA_RANK": LORA_RANK,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "FT_EPOCHS": FT_EPOCHS,
        "FT_PATIENCE": FT_PATIENCE,
        "LORA_EPOCHS": LORA_EPOCHS,
        "LORA_PATIENCE": LORA_PATIENCE,
        "PREPROCESS": PREPROCESS,
        "IMAGE_HEIGHT": IMAGE_HEIGHT,
        "IMAGE_WIDTH": IMAGE_WIDTH,
        "LOG_BATCH_INTERVAL": LOG_BATCH_INTERVAL,
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
        "RUN_EXPORT_METRICS_TABLES": RUN_EXPORT_METRICS_TABLES,
        "RUN_PYTEST": RUN_PYTEST,
        "FINETUNED_CHECKPOINT_PATHS": list(FINETUNED_CHECKPOINT_PATHS),
        "LORA_CHECKPOINT_PATHS": list(LORA_CHECKPOINT_PATHS),
    }


def _trace_stage(cwd: Path, logger: "PipelineLogger", action: str, stage_id: str, **extra: Any) -> None:
    from utils.pipeline_state import append_stage_event

    tail = " ".join(f"{k}={extra[k]!r}" for k in sorted(extra))
    logger.log_line(f"[STAGE] action={action} stage_id={stage_id}" + (f" {tail}" if tail else ""))
    append_stage_event(cwd, {"action": action, "stage_id": stage_id, **extra})


def _pipeline_log_file_only() -> bool:
    """If true, :class:`PipelineLogger` writes only to the log file (for detached + dashboard workflow)."""
    v = os.environ.get("SEMANTIC_CORRESPONDENCE_PIPELINE_LOG_FILE_ONLY", "").strip().lower()
    return v in ("1", "true", "yes", "on")


class PipelineLogger:
    """Timestamped lines to a per-run log file; optionally mirrors to stdout/stderr (interactive runs)."""

    def __init__(self, path: Path, *, mirror_to_terminal: bool = True) -> None:
        self.path = path.resolve()
        self._mirror = mirror_to_terminal
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "w", encoding="utf-8")
        self._fp.write(f"# Pipeline run started at {self._ts()}\n")
        self._fp.write(f"# Log file: {self.path}\n")
        self._fp.write("# Also see: runs/logs/stage_events.jsonl and runs/pipeline_state.json (resume).\n\n")
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
    line = "\t".join(row) + "\n"
    if not manifest.is_file():
        with open(manifest, "w", encoding="utf-8") as f:
            f.write("timestamp_utc\tevent\tlog_filename\texit_code\n")
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(line)


def _repo_root() -> Path:
    return _REPO


def _link_current_log_symlink(cwd: Path, log_path: Path) -> None:
    """Point ``runs/logs/current.log`` at this run so dashboards can reconnect after closing a terminal."""
    link = cwd / "runs" / "logs" / "current.log"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(log_path.name)
    except OSError:
        pass


def _validate_triplet(name: str, t: Tuple[bool, ...]) -> None:
    if len(t) != 3:
        raise ValueError(f"{name} must have exactly 3 elements (DINOv2, DINOv3, SAM), got {len(t)}")


def _python(cwd: Path) -> str:
    return sys.executable


def _run_cmd(cwd: Path, cmd: List[str], logger: PipelineLogger) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    logger.log_line(f"--- Running: {' '.join(cmd)}")
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if p.stdout is None:
        return p.wait()
    for raw in p.stdout:
        logger.log_line(raw.rstrip("\n"))
    return p.wait()


def _run_script(
    cwd: Path,
    rel_script: str,
    args: Sequence[str],
    logger: PipelineLogger,
) -> int:
    cmd = [_python(cwd), str(cwd / rel_script), *args]
    return _run_cmd(cwd, cmd, logger)


def _default_finetune_ckpt(cwd: Path, backbone: str) -> str:
    return str(cwd / CHECKPOINT_DIR / f"{backbone}_lastblocks{LAST_BLOCKS}_best.pt")


def _default_lora_ckpt(cwd: Path, backbone: str) -> str:
    return str(cwd / CHECKPOINT_DIR / f"{backbone}_lora_r{LORA_RANK}_best.pt")


def _resolve_ckpt(
    cwd: Path,
    backbone: str,
    explicit: Optional[str],
    default_fn,
) -> Optional[str]:
    if explicit:
        p = Path(explicit)
        return str(p if p.is_absolute() else cwd / p)
    path = Path(default_fn(cwd, backbone))
    return str(path) if path.is_file() else None


def _build_eval_specs(
    cwd: Path,
    sam_checkpoint: Optional[str],
    logger: PipelineLogger,
    *,
    num_workers: int,
) -> List[Any]:
    from evaluation.experiment_runner import EvalRunSpec

    specs: List[EvalRunSpec] = []
    triples: List[Tuple[str, int]] = list(zip(BACKBONE_NAMES, range(3)))

    for backbone, idx in triples:
        d2 = DINOV2_WEIGHTS
        d3 = DINOV3_WEIGHTS
        sam = sam_checkpoint
        if backbone == "sam_vit_b" and not sam:
            if (
                RUN_EVAL_BASELINE[idx]
                or RUN_EVAL_BASELINE_WSA[idx]
                or RUN_EVAL_FINETUNED_CHECKPOINT[idx]
                or RUN_EVAL_LORA_CHECKPOINT[idx]
            ):
                logger.log_line(
                    f"WARNING: skipping SAM eval for {backbone}: set SAM_CHECKPOINT in run_pipeline.py.",
                    err=True,
                )
            continue

        if RUN_EVAL_BASELINE[idx]:
            specs.append(
                EvalRunSpec(
                    name=f"{backbone}_baseline",
                    backbone=backbone,
                    split=EVAL_SPLIT,
                    dataset_backend=DATASET_BACKEND,
                    metrics_backend=METRICS_BACKEND,
                    dinov2_weights=d2,
                    dinov3_weights=d3,
                    sam_checkpoint=sam,
                    limit=EVAL_LIMIT,
                    preprocess=PREPROCESS,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    num_workers=num_workers,
                )
            )
        if RUN_EVAL_BASELINE_WSA[idx]:
            specs.append(
                EvalRunSpec(
                    name=f"{backbone}_baseline_wsa",
                    backbone=backbone,
                    split=EVAL_SPLIT,
                    dataset_backend=DATASET_BACKEND,
                    metrics_backend=METRICS_BACKEND,
                    dinov2_weights=d2,
                    dinov3_weights=d3,
                    sam_checkpoint=sam,
                    use_window_soft_argmax=True,
                    wsa_window=WSA_WINDOW,
                    wsa_temperature=WSA_TEMPERATURE,
                    limit=EVAL_LIMIT,
                    preprocess=PREPROCESS,
                    height=IMAGE_HEIGHT,
                    width=IMAGE_WIDTH,
                    num_workers=num_workers,
                )
            )
        if RUN_EVAL_FINETUNED_CHECKPOINT[idx]:
            ck = _resolve_ckpt(cwd, backbone, FINETUNED_CHECKPOINT_PATHS[idx], _default_finetune_ckpt)
            if not ck:
                logger.log_line(
                    f"WARNING: fine-tuned checkpoint not found for {backbone} (train or set path).",
                    err=True,
                )
            else:
                specs.append(
                    EvalRunSpec(
                        name=f"{backbone}_finetuned",
                        backbone=backbone,
                        split=EVAL_SPLIT,
                        dataset_backend=DATASET_BACKEND,
                        metrics_backend=METRICS_BACKEND,
                        dinov2_weights=d2,
                        dinov3_weights=d3,
                        sam_checkpoint=sam,
                        checkpoint=ck,
                        limit=EVAL_LIMIT,
                        preprocess=PREPROCESS,
                        height=IMAGE_HEIGHT,
                        width=IMAGE_WIDTH,
                        num_workers=num_workers,
                    )
                )
        if RUN_EVAL_LORA_CHECKPOINT[idx]:
            ck = _resolve_ckpt(cwd, backbone, LORA_CHECKPOINT_PATHS[idx], _default_lora_ckpt)
            if not ck:
                logger.log_line(
                    f"WARNING: LoRA checkpoint not found for {backbone} (train or set path).",
                    err=True,
                )
            else:
                specs.append(
                    EvalRunSpec(
                        name=f"{backbone}_lora",
                        backbone=backbone,
                        split=EVAL_SPLIT,
                        dataset_backend=DATASET_BACKEND,
                        metrics_backend=METRICS_BACKEND,
                        dinov2_weights=d2,
                        dinov3_weights=d3,
                        sam_checkpoint=sam,
                        checkpoint=ck,
                        limit=EVAL_LIMIT,
                        preprocess=PREPROCESS,
                        height=IMAGE_HEIGHT,
                        width=IMAGE_WIDTH,
                        num_workers=num_workers,
                    )
                )
    return specs


def _export_tables(
    cwd: Path,
    results: Sequence[Dict[str, Any]],
    logger: PipelineLogger,
) -> None:
    from evaluation.experiment_runner import metrics_rows_for_table

    out_dir = cwd / "runs" / "pipeline_exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics_rows_for_table(results)
    json_path = out_dir / "pck_results.json"
    csv_path = out_dir / "pck_results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    # PDF requirement: keep per-image and per-point results available by default when using
    # SD4Match metrics backend. These are stored as separate JSON files for convenience.
    per_image = []
    per_point = []
    by_difficulty_flag = []
    for r in results:
        if "sd4match_per_image" in r:
            per_image.append({"name": r.get("name"), "split": r.get("spec", {}).get("split"), "data": r["sd4match_per_image"]})
        if "sd4match_per_point" in r:
            per_point.append({"name": r.get("name"), "split": r.get("spec", {}).get("split"), "data": r["sd4match_per_point"]})
        if "sd4match_by_difficulty_flag" in r:
            by_difficulty_flag.append(
                {
                    "name": r.get("name"),
                    "split": r.get("spec", {}).get("split"),
                    "data": r["sd4match_by_difficulty_flag"],
                }
            )
    per_image_path = out_dir / "pck_results_per_image.json"
    per_point_path = out_dir / "pck_results_per_point.json"
    by_difficulty_flag_path = out_dir / "pck_results_by_difficulty_flag.json"
    if per_image:
        with open(per_image_path, "w", encoding="utf-8") as f:
            json.dump(per_image, f, indent=2)
    if per_point:
        with open(per_point_path, "w", encoding="utf-8") as f:
            json.dump(per_point, f, indent=2)
    if by_difficulty_flag:
        with open(by_difficulty_flag_path, "w", encoding="utf-8") as f:
            json.dump(by_difficulty_flag, f, indent=2)

    wrote = [str(json_path), str(csv_path)]
    if per_image:
        wrote.append(str(per_image_path))
    if per_point:
        wrote.append(str(per_point_path))
    if by_difficulty_flag:
        wrote.append(str(by_difficulty_flag_path))
    logger.log_line("Wrote " + " | ".join(wrote) + ".")


def _run_eval_and_export(
    cwd: Path,
    sam_checkpoint: Optional[str],
    logger: PipelineLogger,
    *,
    num_workers: int,
    device_str: str,
) -> int:
    import torch

    from evaluation.experiment_runner import run_comparison_batch

    specs = _build_eval_specs(cwd, sam_checkpoint, logger, num_workers=num_workers)
    if not specs:
        logger.log_line("No PCK eval specs selected (all RUN_EVAL_* flags False or skipped).")
        return 0

    device = torch.device(device_str)
    results = run_comparison_batch(
        specs,
        spair_root=SPAIR_ROOT,
        alphas=EVAL_ALPHAS,
        device=device,
    )
    for r in results:
        logger.log_line(f"--- {r['name']} --- {r['metrics']}")
    if RUN_EXPORT_METRICS_TABLES:
        _export_tables(cwd, results, logger)
    return 0


def _sam_slot_used() -> bool:
    return (
        TRAIN_FINETUNE[2]
        or TRAIN_LORA[2]
        or RUN_EVAL_BASELINE[2]
        or RUN_EVAL_BASELINE_WSA[2]
        or RUN_EVAL_FINETUNED_CHECKPOINT[2]
        or RUN_EVAL_LORA_CHECKPOINT[2]
    )


def _pipeline_run(cwd: Path, logger: PipelineLogger) -> int:
    from utils.hardware import recommended_dataloader_workers, recommended_device_str

    for name, t in (
        ("TRAIN_FINETUNE", TRAIN_FINETUNE),
        ("TRAIN_LORA", TRAIN_LORA),
        ("RUN_EVAL_BASELINE", RUN_EVAL_BASELINE),
        ("RUN_EVAL_BASELINE_WSA", RUN_EVAL_BASELINE_WSA),
        ("RUN_EVAL_FINETUNED_CHECKPOINT", RUN_EVAL_FINETUNED_CHECKPOINT),
        ("RUN_EVAL_LORA_CHECKPOINT", RUN_EVAL_LORA_CHECKPOINT),
    ):
        _validate_triplet(name, t)

    eff_device = DEVICE if DEVICE is not None else recommended_device_str()
    eff_workers = NUM_WORKERS if NUM_WORKERS is not None else recommended_dataloader_workers(
        accelerator=eff_device
    )
    logger.log_line(
        f"Adaptive hardware: device={eff_device} num_workers={eff_workers} "
        f"cpu_count={os.cpu_count()} platform={sys.platform}"
    )

    from utils import pipeline_state as _ps

    fp = _ps.fingerprint_from_config(_fingerprint_payload())
    if not PIPELINE_RESUME:
        state = {"fingerprint": fp, "completed": []}
        logger.log_line(
            "PIPELINE_RESUME=False: stage skip and pipeline_state.json updates disabled; "
            "each run executes all enabled stages (no automatic --resume)."
        )
    elif _ps.should_reset_from_env():
        state = {"fingerprint": fp, "completed": []}
        _ps.save_state(cwd, state)
        logger.log_line(
            "pipeline_state: cleared (environment SEMANTIC_CORRESPONDENCE_PIPELINE_RESET is set)."
        )
        _trace_stage(cwd, logger, "reset", "pipeline_state", fingerprint_prefix=fp[:16])
    else:
        raw = _ps.load_state(cwd)
        if raw is None:
            state = {"fingerprint": fp, "completed": []}
            logger.log_line(f"pipeline_state: no prior file (will write {_ps.state_path(cwd)}).")
        elif raw.get("fingerprint") != fp:
            state = {"fingerprint": fp, "completed": []}
            logger.log_line(
                "pipeline_state: fingerprint mismatch (config changed); previous completed steps ignored."
            )
            _ps.save_state(cwd, state)
        else:
            state = dict(raw)
            state["completed"] = list(state.get("completed", []))
            state["fingerprint"] = fp
            logger.log_line(f"pipeline_state: completed steps = {state['completed']!r}")
    if PIPELINE_RESUME:
        logger.log_line(
            f"Tracing: structured log {_ps.stage_events_path(cwd)} | state file {_ps.state_path(cwd)}"
        )

    effective_sam = SAM_CHECKPOINT or os.environ.get("SAM_CHECKPOINT")
    if _sam_slot_used() and not effective_sam:
        logger.log_line(
            "ERROR: SAM is enabled in the pipeline but SAM_CHECKPOINT is not set.\n"
            "  Set SAM_CHECKPOINT in run_pipeline.py or export SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth",
            err=True,
        )
        return 2

    if RUN_VERIFY_DATASET:
        v_sid = "verify_dataset"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), v_sid):
            logger.log_line(f"[SKIP] stage_id={v_sid} reason=already_completed")
            _trace_stage(cwd, logger, "skip", v_sid, reason="already_completed")
        else:
            _trace_stage(cwd, logger, "start", v_sid)
            rc = _run_script(
                cwd,
                "scripts/verify_dataset.py",
                ([] if SPAIR_ROOT is None else ["--spair-root", SPAIR_ROOT]),
                logger,
            )
            if rc != 0:
                _trace_stage(cwd, logger, "fail", v_sid, exit_code=rc)
                return rc
            if PIPELINE_RESUME:
                _ps.mark_step_done(cwd, state, v_sid)
            _trace_stage(cwd, logger, "done", v_sid, exit_code=0)

    common_train: List[str] = []
    if SPAIR_ROOT:
        common_train.extend(["--spair-root", SPAIR_ROOT])
    common_train.extend(
        [
            "--preprocess",
            PREPROCESS,
            "--height",
            str(IMAGE_HEIGHT),
            "--width",
            str(IMAGE_WIDTH),
            "--batch-size",
            str(TRAIN_BATCH_SIZE),
            "--num-workers",
            str(eff_workers),
            "--checkpoint-dir",
            CHECKPOINT_DIR,
            "--device",
            eff_device,
        ]
    )
    if DINOV2_WEIGHTS:
        common_train.extend(["--dinov2-weights", DINOV2_WEIGHTS])
    if DINOV3_WEIGHTS:
        common_train.extend(["--dinov3-weights", DINOV3_WEIGHTS])
    if effective_sam:
        common_train.extend(["--sam-checkpoint", effective_sam])
    common_train.extend(["--log-batch-interval", str(LOG_BATCH_INTERVAL)])

    for i, backbone in enumerate(BACKBONE_NAMES):
        if not TRAIN_FINETUNE[i]:
            continue
        if backbone == "sam_vit_b" and not effective_sam:
            logger.log_line("ERROR: TRAIN_FINETUNE for SAM requires SAM_CHECKPOINT.", err=True)
            return 2
        ft_sid = f"finetune:{backbone}"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), ft_sid):
            logger.log_line(f"[SKIP] stage_id={ft_sid} reason=already_completed")
            _trace_stage(cwd, logger, "skip", ft_sid, reason="already_completed")
            continue
        args = [
            *common_train,
            "--backbone",
            backbone,
            "--epochs",
            str(FT_EPOCHS),
            "--patience",
            str(FT_PATIENCE),
            "--last-blocks",
            str(LAST_BLOCKS),
        ]
        resume_file = (cwd / CHECKPOINT_DIR / f"{backbone}_lastblocks{LAST_BLOCKS}_resume.pt").resolve()
        if PIPELINE_RESUME and resume_file.is_file():
            args.extend(["--resume", str(resume_file)])
            logger.log_line(f"[RESUME] stage_id={ft_sid} checkpoint={resume_file}")
            _trace_stage(cwd, logger, "resume_prepare", ft_sid, path=str(resume_file))
        _trace_stage(cwd, logger, "start", ft_sid)
        rc = _run_script(cwd, "scripts/train_finetune.py", args, logger)
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
        lora_sid = f"lora:{backbone}"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), lora_sid):
            logger.log_line(f"[SKIP] stage_id={lora_sid} reason=already_completed")
            _trace_stage(cwd, logger, "skip", lora_sid, reason="already_completed")
            continue
        args = [
            *common_train,
            "--backbone",
            backbone,
            "--epochs",
            str(LORA_EPOCHS),
            "--patience",
            str(LORA_PATIENCE),
            "--last-blocks",
            str(LAST_BLOCKS),
            "--rank",
            str(LORA_RANK),
        ]
        resume_lora = (cwd / CHECKPOINT_DIR / f"{backbone}_lora_r{LORA_RANK}_resume.pt").resolve()
        if PIPELINE_RESUME and resume_lora.is_file():
            args.extend(["--resume", str(resume_lora)])
            logger.log_line(f"[RESUME] stage_id={lora_sid} checkpoint={resume_lora}")
            _trace_stage(cwd, logger, "resume_prepare", lora_sid, path=str(resume_lora))
        _trace_stage(cwd, logger, "start", lora_sid)
        rc = _run_script(cwd, "scripts/train_lora.py", args, logger)
        if rc != 0:
            _trace_stage(cwd, logger, "fail", lora_sid, exit_code=rc)
            return rc
        if PIPELINE_RESUME:
            _ps.mark_step_done(cwd, state, lora_sid)
        _trace_stage(cwd, logger, "done", lora_sid, exit_code=0)

    any_eval = any(
        any(t)
        for t in (
            RUN_EVAL_BASELINE,
            RUN_EVAL_BASELINE_WSA,
            RUN_EVAL_FINETUNED_CHECKPOINT,
            RUN_EVAL_LORA_CHECKPOINT,
        )
    )
    if any_eval:
        ev_sid = "eval_and_export"
        if PIPELINE_RESUME and _ps.is_step_done(state.get("completed", []), ev_sid):
            logger.log_line(f"[SKIP] stage_id={ev_sid} reason=already_completed")
            _trace_stage(cwd, logger, "skip", ev_sid, reason="already_completed")
        else:
            _trace_stage(cwd, logger, "start", ev_sid)
            rc = _run_eval_and_export(
                cwd,
                effective_sam,
                logger,
                num_workers=eff_workers,
                device_str=eff_device,
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
            logger.log_line(f"[SKIP] stage_id={py_sid} reason=already_completed")
            _trace_stage(cwd, logger, "skip", py_sid, reason="already_completed")
        else:
            _trace_stage(cwd, logger, "start", py_sid)
            rc = _run_cmd(cwd, [_python(cwd), "-m", "pytest", "-q", "tests"], logger)
            if rc != 0:
                _trace_stage(cwd, logger, "fail", py_sid, exit_code=rc)
                return rc
            if PIPELINE_RESUME:
                _ps.mark_step_done(cwd, state, py_sid)
            _trace_stage(cwd, logger, "done", py_sid, exit_code=0)

    if PRINT_JUPYTER_NOTEBOOK_HINT:
        nb = cwd / "notebooks" / "verify_and_compare_results.ipynb"
        logger.log_line(
            "--- Notebook (qualitative comparison & plots) ---\n"
            f"  pip install -e \".[notebook]\"\n"
            f"  jupyter notebook {nb}\n"
            "Edit the configuration cell; project rules are in docs/info.md."
        )

    logger.log_line("Pipeline finished (all configured stages for this run).")
    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run semantic-correspondence pipeline.")
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Optional YAML file (Colab / notebook style) overriding in-script defaults.",
    )
    args = parser.parse_args()

    cwd = _repo_root()
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
