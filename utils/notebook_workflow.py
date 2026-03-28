"""
Notebook-friendly orchestration helpers.

This module centralizes the configuration and the small amount of glue that an external
``.py`` / Jupyter notebook needs in order to:

- resolve the project and dataset paths,
- launch fine-tuning or LoRA training with a single config object,
- run the same PCK evaluation path used by the CLI,
- export results to JSON/CSV, and
- visualize dataset samples and metric summaries.

The goal is to keep the notebook outside this repository thin: it should only load a config
file and call these helpers.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from data.dataset import PreprocessMode, SPair71kPairDataset, visualize_pair
from data.paths import resolve_spair_root
from evaluation.experiment_runner import EvalRunSpec, metrics_rows_for_table, run_comparison_batch
from utils.paths import repo_root as _repo_root

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional for headless environments
    plt = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is an optional notebook dependency
    yaml = None  # type: ignore[assignment]


def _infer_repo_root(start: Optional[str] = None) -> Path:
    if start:
        candidate = Path(start).expanduser().resolve()
        for root in [candidate, *candidate.parents]:
            if (root / "pyproject.toml").is_file():
                return root
    return _repo_root()


def _resolve_path(base: Path, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((base / candidate).resolve())


def _normalize_preprocess_name(name: str) -> str:
    return PreprocessMode[name.strip().upper()].name


@dataclass(frozen=True)
class PathsConfig:
    repo_root: Optional[str] = None
    spair_root: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "runs/notebook_exports"


@dataclass(frozen=True)
class RuntimeConfig:
    device: Optional[str] = None
    num_workers: int = -1
    preprocess: str = "FIXED_RESIZE"
    image_height: int = 784
    image_width: int = 784
    limit_pairs: int = 0
    alphas: Tuple[float, float, float] = (0.05, 0.1, 0.2)
    sample_split: str = "val"
    sample_index: int = 0
    wsa_window: int = 5
    wsa_temperature: float = 1.0


@dataclass(frozen=True)
class TrainingJobConfig:
    enabled: bool = True
    backbone: str = "dinov2_vitb14"
    last_blocks: int = 2
    epochs: int = 50
    patience: int = 5
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 100
    num_workers: int = -1
    log_batch_interval: int = 2500
    resume_save_interval: int = 100
    resume: Optional[str] = None
    dinov2_weights: Optional[str] = None
    dinov3_weights: Optional[str] = None
    sam_checkpoint: Optional[str] = None
    rank: int = 8
    alpha: float = 16.0


@dataclass(frozen=True)
class NotebookWorkflowConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    finetune: TrainingJobConfig = field(default_factory=TrainingJobConfig)
    lora: TrainingJobConfig = field(
        default_factory=lambda: TrainingJobConfig(
            enabled=True,
            backbone="dinov2_vitb14",
            last_blocks=2,
            epochs=2,
            patience=3,
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=100,
            num_workers=-1,
            log_batch_interval=2500,
            resume_save_interval=100,
            rank=8,
            alpha=16.0,
        )
    )
    experiments: Tuple[Dict[str, Any], ...] = ()
    device_override: Optional[str] = None

    def repo_root(self) -> Path:
        return _infer_repo_root(self.paths.repo_root)

    def spair_root(self) -> str:
        return resolve_spair_root(self.paths.spair_root)

    def checkpoint_dir(self) -> Path:
        return (self.repo_root() / self.paths.checkpoint_dir).resolve()

    def output_dir(self) -> Path:
        return (self.repo_root() / self.paths.output_dir).resolve()

    def device(self) -> Optional[str]:
        return self.device_override or self.runtime.device

    def preprocess(self) -> str:
        return _normalize_preprocess_name(self.runtime.preprocess)

    def alphas(self) -> Tuple[float, ...]:
        return tuple(float(a) for a in self.runtime.alphas)

    def evaluation_specs(self) -> List[EvalRunSpec]:
        specs: List[EvalRunSpec] = []
        root = self.repo_root()
        for item in self.experiments:
            payload = dict(item)
            for key in ("checkpoint", "dinov2_weights", "dinov3_weights", "sam_checkpoint"):
                if key in payload:
                    payload[key] = _resolve_path(root, payload.get(key))
            if "preprocess" in payload:
                payload["preprocess"] = _normalize_preprocess_name(str(payload["preprocess"]))
            payload.setdefault("limit", self.runtime.limit_pairs)
            payload.setdefault("num_workers", self.runtime.num_workers)
            payload.setdefault("wsa_window", self.runtime.wsa_window)
            payload.setdefault("wsa_temperature", self.runtime.wsa_temperature)
            specs.append(EvalRunSpec(**payload))
        return specs


@dataclass(frozen=True)
class TrainingRunResult:
    mode: str
    command: Tuple[str, ...]
    returncode: int
    log_path: str
    best_checkpoint: Optional[str]
    resume_checkpoint: Optional[str]


def _dataclass_to_plain_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_plain_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_dataclass_to_plain_dict(v) for v in obj]
    if isinstance(obj, list):
        return [_dataclass_to_plain_dict(v) for v in obj]
    return obj


def _kwargs_for_dataclass(cls: type, data: Any) -> dict[str, Any]:
    """Drop keys unknown to ``cls`` so pipeline-style YAML (e.g. Colab ``config.yaml``) can load."""
    if not isinstance(data, dict):
        return {}
    fields = getattr(cls, "__dataclass_fields__", None)
    if not fields:
        return dict(data)
    return {k: v for k, v in data.items() if k in fields}


def load_workflow_config(path: str | os.PathLike[str]) -> NotebookWorkflowConfig:
    """Load a workflow config from YAML or JSON.

    Unknown keys under ``paths`` / ``runtime`` / ``finetune`` / ``lora`` are ignored so YAML
    also used with ``python scripts/run_pipeline.py --config`` (extra ``eval_split``,
    ``resume_save_interval``, ``dataset``, ``workflow_toggles``, …) still loads here.
    """
    p = Path(path).expanduser().resolve()
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs.")
        raw = yaml.safe_load(text) or {}
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping in {p}, got {type(raw).__name__}")

    paths = PathsConfig(**_kwargs_for_dataclass(PathsConfig, raw.get("paths") or {}))
    runtime = RuntimeConfig(**_kwargs_for_dataclass(RuntimeConfig, raw.get("runtime") or {}))
    finetune = TrainingJobConfig(**_kwargs_for_dataclass(TrainingJobConfig, raw.get("finetune") or {}))
    lora = TrainingJobConfig(**_kwargs_for_dataclass(TrainingJobConfig, raw.get("lora") or {}))
    experiments = tuple(dict(item) for item in raw.get("experiments", []))
    device_override = raw.get("device_override")
    return NotebookWorkflowConfig(
        paths=paths,
        runtime=runtime,
        finetune=finetune,
        lora=lora,
        experiments=experiments,
        device_override=device_override,
    )


def save_workflow_config(config: NotebookWorkflowConfig, path: str | os.PathLike[str]) -> Path:
    """Write a workflow config to YAML or JSON."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _dataclass_to_plain_dict(config)
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to save YAML configs.")
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    else:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    return p


def ensure_repo_importable(config: NotebookWorkflowConfig) -> Path:
    """Make the repository importable in ad-hoc notebook sessions."""
    root = config.repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _script_path(config: NotebookWorkflowConfig, script_name: str) -> Path:
    return config.repo_root() / "scripts" / script_name


def _training_result_paths(config: NotebookWorkflowConfig, job: TrainingJobConfig, mode: str) -> Tuple[str, str]:
    ckpt_dir = config.checkpoint_dir()
    if mode == "finetune":
        best = ckpt_dir / f"{job.backbone}_lastblocks{job.last_blocks}_best.pt"
        resume = ckpt_dir / f"{job.backbone}_lastblocks{job.last_blocks}_resume.pt"
    else:
        best = ckpt_dir / f"{job.backbone}_lora_r{job.rank}_best.pt"
        resume = ckpt_dir / f"{job.backbone}_lora_r{job.rank}_resume.pt"
    return str(best), str(resume)


def build_training_command(
    config: NotebookWorkflowConfig,
    job: TrainingJobConfig,
    *,
    mode: str,
) -> List[str]:
    script = "train_finetune.py" if mode == "finetune" else "train_lora.py"
    script_path = _script_path(config, script)
    args: List[str] = [str(script_path)]
    spair_root = config.spair_root()
    if spair_root:
        args.extend(["--spair-root", spair_root])
    args.extend(["--backbone", job.backbone])

    d2 = _resolve_path(config.repo_root(), job.dinov2_weights)
    d3 = _resolve_path(config.repo_root(), job.dinov3_weights)
    sam = _resolve_path(config.repo_root(), job.sam_checkpoint)
    if d2:
        args.extend(["--dinov2-weights", d2])
    if d3:
        args.extend(["--dinov3-weights", d3])
    if sam:
        args.extend(["--sam-checkpoint", sam])

    args.extend(["--last-blocks", str(job.last_blocks)])
    if mode == "finetune":
        args.extend(["--lr", str(job.learning_rate)])
        args.extend(["--weight-decay", str(job.weight_decay)])
    else:
        args.extend(["--rank", str(job.rank)])
        args.extend(["--alpha", str(job.alpha)])
        args.extend(["--lr", str(job.learning_rate)])
    args.extend(["--epochs", str(job.epochs)])
    args.extend(["--batch-size", str(job.batch_size)])
    args.extend(["--num-workers", str(job.num_workers)])
    args.extend(["--preprocess", config.preprocess()])
    args.extend(["--height", str(config.runtime.image_height)])
    args.extend(["--width", str(config.runtime.image_width)])
    args.extend(["--patience", str(job.patience)])
    args.extend(["--checkpoint-dir", str(config.checkpoint_dir())])
    if config.device():
        args.extend(["--device", config.device() or ""])
    args.extend(["--log-batch-interval", str(job.log_batch_interval)])
    args.extend(["--resume-save-interval", str(job.resume_save_interval)])
    resume = _resolve_path(config.repo_root(), job.resume)
    if resume:
        args.extend(["--resume", resume])
    return args


def run_training_job(
    config: NotebookWorkflowConfig,
    job: TrainingJobConfig,
    *,
    mode: str,
    check: bool = True,
    stream_output: bool = True,
) -> TrainingRunResult:
    """Run one CLI training job from a notebook-friendly config object."""
    config.output_dir().mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir() / "config.json"
    save_workflow_config(config, config_path)

    command = [sys.executable, *build_training_command(config, job, mode=mode)]
    log_dir = config.output_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    log_path = log_dir / f"{mode}_{job.backbone}_{timestamp}.log"
    best_ckpt, resume_ckpt = _training_result_paths(config, job, mode)

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(config.repo_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            if stream_output:
                sys.stdout.write(line)
                sys.stdout.flush()
        returncode = process.wait()

    result = TrainingRunResult(
        mode=mode,
        command=tuple(command),
        returncode=int(returncode),
        log_path=str(log_path),
        best_checkpoint=best_ckpt if Path(best_ckpt).is_file() else None,
        resume_checkpoint=resume_ckpt if Path(resume_ckpt).is_file() else None,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"{mode} training failed with exit code {result.returncode}. See {log_path}")
    return result


def run_finetune_job(
    config: NotebookWorkflowConfig,
    *,
    check: bool = True,
    stream_output: bool = True,
) -> TrainingRunResult:
    return run_training_job(config, config.finetune, mode="finetune", check=check, stream_output=stream_output)


def run_lora_job(
    config: NotebookWorkflowConfig,
    *,
    check: bool = True,
    stream_output: bool = True,
) -> TrainingRunResult:
    return run_training_job(config, config.lora, mode="lora", check=check, stream_output=stream_output)


def run_evaluation_suite(
    config: NotebookWorkflowConfig,
    *,
    specs: Optional[Sequence[EvalRunSpec]] = None,
) -> List[Dict[str, Any]]:
    """Run the notebook evaluation suite with the same evaluation code path as the CLI."""
    eval_specs = list(specs) if specs is not None else config.evaluation_specs()
    if not eval_specs:
        raise ValueError("No evaluation specs configured.")
    device = torch.device(config.device()) if config.device() else None
    return run_comparison_batch(eval_specs, spair_root=config.spair_root(), alphas=config.alphas(), device=device)


def save_results(
    results: Sequence[Dict[str, Any]],
    output_dir: str | os.PathLike[str],
    *,
    stem: str = "comparison",
) -> Dict[str, Path]:
    """Persist evaluation results as JSON and CSV."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(results), f, indent=2, default=str)
        f.write("\n")
    rows = metrics_rows_for_table(results)
    if rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {"json": json_path, "csv": csv_path}


def result_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return metrics_rows_for_table(results)


def plot_results(results: Sequence[Dict[str, Any]], *, title: str = "PCK comparison") -> None:
    """Render a compact bar chart for the metric table."""
    if plt is None:
        raise ImportError("matplotlib is required for plot_results().")
    rows = metrics_rows_for_table(results)
    if not rows:
        raise ValueError("No results to plot.")
    metric_cols = [k for k in rows[0].keys() if k.startswith("pck@")]
    if not metric_cols:
        raise ValueError("Results do not contain any pck@ columns.")

    names = [row["name"] for row in rows]
    x = list(range(len(rows)))
    width = 0.8 / max(len(metric_cols), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 4.5))
    for idx, metric in enumerate(metric_cols):
        values = [float(row.get(metric, 0.0)) for row in rows]
        offset = (idx - (len(metric_cols) - 1) / 2.0) * width
        ax.bar([i + offset for i in x], values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("micro PCK")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    plt.show()


def show_dataset_sample(
    config: NotebookWorkflowConfig,
    *,
    split: Optional[str] = None,
    index: int = 0,
) -> None:
    """Display one source/target pair using the same preprocessing as the notebook config."""
    mode = PreprocessMode[config.preprocess()]
    ds = SPair71kPairDataset(
        spair_root=config.spair_root(),
        split=split or config.runtime.sample_split,
        preprocess=mode,
        output_size_hw=(config.runtime.image_height, config.runtime.image_width),
        normalize=True,
        photometric_augment=None,
    )
    sample = ds[index]
    visualize_pair(sample["src_img"], sample["tgt_img"], title=f"{split or config.runtime.sample_split}[{index}]")


__all__ = [
    "NotebookWorkflowConfig",
    "PathsConfig",
    "RuntimeConfig",
    "TrainingJobConfig",
    "TrainingRunResult",
    "build_training_command",
    "ensure_repo_importable",
    "load_workflow_config",
    "plot_results",
    "result_rows",
    "run_evaluation_suite",
    "run_finetune_job",
    "run_lora_job",
    "run_training_job",
    "save_results",
    "save_workflow_config",
    "show_dataset_sample",
]