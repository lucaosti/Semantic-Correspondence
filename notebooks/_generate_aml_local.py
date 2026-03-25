#!/usr/bin/env python3
"""One-off generator for AML.ipynb (local Linux + GPU). Run from repo: python notebooks/_generate_aml_local.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "AML.ipynb"


def cell_md(lines: list[str]) -> dict:
    src = []
    for line in lines:
        src.append(line if line.endswith("\n") else line + "\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def cell_code(lines: list[str]) -> dict:
    src = []
    for line in lines:
        src.append(line if line.endswith("\n") else line + "\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


cells: list[dict] = []

cells.append(
    cell_md(
        [
            "# Semantic Correspondence — Linux + NVIDIA (Jupyter)",
            "",
            "Assumption: the **repository** and **dataset** (`data/SPair-71k/`) are already on the machine. This notebook does **not** clone, **not** download, and **not** extract the dataset.",
            "",
            "Flow: **1** repo+GPU → **2** project progress folders → **3** `pip install` → **4** weights/config → **5** pipeline.",
            "",
            "**Progress** — Logs, `pipeline_state.json`, checkpoints, and weights are stored in **`runs/`** and **`checkpoints/`** at the repo root (see `.gitignore`).",
            "",
            "**GPU** — Cell 1 enables NVIDIA optimizations; `config.yaml` uses `device: cuda` and `num_workers: -1` (CUDA auto mode in `utils.hardware`).",
        ]
    )
)

cells.append(cell_md(["### 1. Repository root and GPU check"]))

cells.append(
    cell_code(
        [
            "from __future__ import annotations",
            "",
            "import os",
            "from pathlib import Path",
            "",
            "",
            "def find_repo_root() -> Path:",
            '    """Directory containing pyproject.toml (walks up from cwd)."""',
            "    cwd = Path.cwd().resolve()",
            "    for p in [cwd, *cwd.parents]:",
            '        if (p / "pyproject.toml").is_file():',
            "            return p",
            "    raise RuntimeError(",
            '        "pyproject.toml not found. Open the notebook from inside the Semantic-Correspondence clone."',
            "    )",
            "",
            "",
            "REPO = find_repo_root()",
            "os.chdir(REPO)",
            'print("REPO =", REPO)',
            "",
            "import torch",
            "",
            "if not torch.cuda.is_available():",
            "    print(",
            '        "WARNING: CUDA is not available. Install a CUDA-enabled PyTorch build and verify with nvidia-smi."',
            "    )",
            "else:",
            "    torch.cuda.set_device(0)",
            "    torch.backends.cudnn.benchmark = True",
            "    torch.backends.cudnn.deterministic = False",
            "    try:",
            '        torch.set_float32_matmul_precision("high")',
            "    except Exception:",
            "        pass",
            '    print("GPU:", torch.cuda.get_device_name(0))',
            '    print("CUDA runtime:", torch.version.cuda, "| PyTorch:", torch.__version__)',
            '    print("NVIDIA: cudnn.benchmark=True, cuda:0 — aligns with device: cuda in config.yaml")',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 2. Project progress (`runs/` and `checkpoints/`)",
            "",
            "Everything written by the pipeline (logs, state files, trained checkpoints, downloaded weights) stays **inside the repository** under `runs/` and `checkpoints/` (no symlink, no Drive, no home folder).",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "if 'find_repo_root' not in globals():",
            "    def find_repo_root() -> Path:",
            "        cwd = Path.cwd().resolve()",
            "        for p in [cwd, *cwd.parents]:",
            '            if (p / "pyproject.toml").is_file():',
            "                return p",
            "        raise RuntimeError('pyproject.toml not found — run cell 1 first.')",
            "",
            "REPO = Path(find_repo_root())",
            "os.chdir(REPO)",
            "",
            'for name in ("runs", "checkpoints"):',
            "    (REPO / name).mkdir(parents=True, exist_ok=True)",
            "",
            'print("Progress and artifacts in repo:")',
            'print(" ", REPO / "runs")',
            'print(" ", REPO / "checkpoints")',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 3. Package installation (`pip install -e`)",
            "",
            "Only project Python dependencies are installed. The dataset is not modified.",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "if 'find_repo_root' not in globals():",
            "    def find_repo_root() -> Path:",
            "        cwd = Path.cwd().resolve()",
            "        for p in [cwd, *cwd.parents]:",
            '            if (p / "pyproject.toml").is_file():',
            "                return p",
            "        raise RuntimeError('pyproject.toml not found — run cell 1 first.')",
            "",
            "REPO = str(find_repo_root())",
            "os.chdir(REPO)",
            "sys.path.insert(0, REPO)",
            "",
            "# Use sys.executable so the current Jupyter kernel environment is used",
            '!{sys.executable} -m pip install -q -e ".[notebook]"',
            "",
            'os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")',
            'print("OK:", REPO)',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 4. Pretrained weights and `config.yaml`",
            "",
            "| Parametro | Valori | Effetto |",
            "|-----------|--------|---------|",
            '| `PIPELINE_RUN_MODE` | `"full"` | verify + fine-tuning + eval + export |',
            '| | `"finetune_only"` | run only `train_finetune` on all three backbones |',
            "| `START_FROM_SCRATCH` | `True` / `False` | start fresh (no resume) vs resume |",
            "| `RESTORE_CONFIG_FROM_ARTIFACTS` | `True` | restore from `runs/notebook_config.yaml` and re-apply the parameters above |",
            "",
            "Automatic config backup to **`runs/notebook_config.yaml`**. Hyperparameters are defined in `_build_fresh_config_dict()`.",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "import shutil",
            "",
            "import yaml",
            "",
            "from pathlib import Path",
            "",
            "if 'find_repo_root' not in globals():",
            "    def find_repo_root() -> Path:",
            "        cwd = Path.cwd().resolve()",
            "        for p in [cwd, *cwd.parents]:",
            '            if (p / "pyproject.toml").is_file():',
            "                return p",
            "        raise RuntimeError('pyproject.toml not found — run cell 1 first.')",
            "",
            "REPO = str(find_repo_root())",
            "os.chdir(REPO)",
            "",
            "# =============================================================================",
            "# PARAMETERS",
            "# =============================================================================",
            'PIPELINE_RUN_MODE = "full"  # "full" | "finetune_only"',
            "START_FROM_SCRATCH = False",
            "RESTORE_CONFIG_FROM_ARTIFACTS = False",
            "# =============================================================================",
            "",
            'cfg_path = os.path.join(REPO, "config.yaml")',
            'BACKUP_CFG = os.path.join(REPO, "runs", "notebook_config.yaml")',
            "os.makedirs(os.path.join(REPO, 'runs'), exist_ok=True)",
            "",
            'os.environ["AML_START_FROM_SCRATCH"] = "1" if START_FROM_SCRATCH else ""',
            'os.environ["AML_PIPELINE_RUN_MODE"] = PIPELINE_RUN_MODE',
            "",
            "",
            "def _workflow_toggles_for_mode(mode: str) -> dict:",
            '    if mode == "finetune_only":',
            "        return {",
            '            "run_verify_dataset": False,',
            '            "train_finetune": [True, True, True],',
            '            "train_lora": [False, False, False],',
            '            "run_eval_baseline": [False, False, False],',
            '            "run_eval_baseline_wsa": [False, False, False],',
            '            "run_eval_finetuned_checkpoint": [False, False, False],',
            '            "run_eval_lora_checkpoint": [False, False, False],',
            '            "run_export_metrics_tables": False,',
            '            "run_pytest": False,',
            '            "print_jupyter_notebook_hint": False,',
            "        }",
            '    if mode != "full":',
            '        raise ValueError("PIPELINE_RUN_MODE must be full or finetune_only")',
            "    return {",
            '        "run_verify_dataset": True,',
            '        "train_finetune": [True, True, True],',
            '        "train_lora": [False, False, False],',
            '        "run_eval_baseline": [True, True, True],',
            '        "run_eval_baseline_wsa": [False, False, False],',
            '        "run_eval_finetuned_checkpoint": [True, True, True],',
            '        "run_eval_lora_checkpoint": [False, False, False],',
            '        "run_export_metrics_tables": True,',
            '        "run_pytest": False,',
            '        "print_jupyter_notebook_hint": True,',
            "    }",
            "",
            "",
            "def _apply_run_mode_to_cfg(cfg: dict) -> None:",
            '    wt = cfg.setdefault("workflow_toggles", {})',
            "    wt.update(_workflow_toggles_for_mode(PIPELINE_RUN_MODE))",
            '    wt["pipeline_resume"] = not START_FROM_SCRATCH',
            "",
            "",
            "def _build_fresh_config_dict() -> dict:",
            '    d2 = f"{REPO}/checkpoints/dinov2_vitb14_pretrain.pth"',
            '    d3 = f"{REPO}/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"',
            '    sam = f"{REPO}/checkpoints/sam_vit_b_01ec64.pth"',
            "    cfg = {",
            '        "dataset": {"backend": "sd4match", "metrics_backend": "sd4match"},',
            '        "paths": {',
            '            "repo_root": REPO,',
            '            "spair_root": f"{REPO}/data/SPair-71k",',
            '            "checkpoint_dir": "checkpoints",',
            '            "output_dir": f"{REPO}/runs/notebook_exports",',
            "        },",
            '        "runtime": {',
            '            "device": "cuda",',
            '            "num_workers": -1,',
            '            "preprocess": "FIXED_RESIZE",',
            '            "image_height": 784,',
            '            "image_width": 784,',
            '            "limit_pairs": 100,',
            '            "eval_split": "val",',
            '            "alphas": [0.05, 0.1, 0.2],',
            '            "wsa_window": 5,',
            '            "wsa_temperature": 1.0,',
            '            "log_batch_interval": 100,',
            "        },",
            '        "finetune": {',
            '            "last_blocks": 2,',
            '            "epochs": 100,',
            '            "patience": 10,',
            '            "batch_size": 10,',
            '            "dinov2_weights": d2,',
            '            "dinov3_weights": d3,',
            '            "sam_checkpoint": sam,',
            "        },",
            '        "lora": {"rank": 16, "epochs": 100, "patience": 3, "batch_size": 10},',
            '        "workflow_toggles": {},',
            "    }",
            "    _apply_run_mode_to_cfg(cfg)",
            "    return cfg",
            "",
            "",
            "if RESTORE_CONFIG_FROM_ARTIFACTS:",
            "    if not os.path.isfile(BACKUP_CFG):",
            "        raise FileNotFoundError(",
            '            "Missing runs/notebook_config.yaml. Run once with RESTORE_CONFIG_FROM_ARTIFACTS=False first."',
            "        )",
            "    shutil.copy(BACKUP_CFG, cfg_path)",
            "    with open(cfg_path, encoding='utf-8') as f:",
            "        cfg = yaml.safe_load(f)",
            "    if not isinstance(cfg, dict):",
            '        raise TypeError("Invalid YAML")',
            "    _apply_run_mode_to_cfg(cfg)",
            "    with open(cfg_path, 'w', encoding='utf-8') as f:",
            "        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)",
            '    print("Config restored + mode applied:", cfg_path)',
            "else:",
            "    import subprocess",
            "    import sys",
            "",
            "    subprocess.run([sys.executable, 'scripts/download_pretrained_weights.py'], check=True)",
            "",
            '    d2 = f"{REPO}/checkpoints/dinov2_vitb14_pretrain.pth"',
            '    d3 = f"{REPO}/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"',
            '    sam = f"{REPO}/checkpoints/sam_vit_b_01ec64.pth"',
            "    missing = []",
            "    for label, p in (('DINOv2', d2), ('DINOv3', d3), ('SAM', sam)):",
            "        ok = os.path.isfile(p)",
            '        print(label, "OK" if ok else "MISSING", p)',
            "        if not ok:",
            "            missing.append((label, p))",
            "    if missing:",
            '        raise RuntimeError(f"Missing weights after download: {missing}")',
            "",
            "    cfg = _build_fresh_config_dict()",
            "    with open(cfg_path, 'w', encoding='utf-8') as f:",
            "        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)",
            '    print("Written:", cfg_path)',
            "",
            "shutil.copy(cfg_path, BACKUP_CFG)",
            'print("Config also saved to:", BACKUP_CFG)',
            "",
            "print(",
            '    f"Mode={PIPELINE_RUN_MODE} | from_scratch={START_FROM_SCRATCH} | pipeline_resume={not START_FROM_SCRATCH}"',
            ")",
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 5. Run pipeline",
            "",
            "Re-run **section 4** after any parameter change. With `START_FROM_SCRATCH = True`, `SEMANTIC_CORRESPONDENCE_PIPELINE_RESET` is set.",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "from pathlib import Path",
            "",
            "if 'find_repo_root' not in globals():",
            "    def find_repo_root() -> Path:",
            "        cwd = Path.cwd().resolve()",
            "        for p in [cwd, *cwd.parents]:",
            '            if (p / "pyproject.toml").is_file():',
            "                return p",
            "        raise RuntimeError('pyproject.toml not found — run cell 1 first.')",
            "",
            "REPO = str(find_repo_root())",
            "os.chdir(REPO)",
            "",
            'if os.environ.get("AML_START_FROM_SCRATCH", "").strip() == "1":',
            '    os.environ["SEMANTIC_CORRESPONDENCE_PIPELINE_RESET"] = "1"',
            "else:",
            '    os.environ.pop("SEMANTIC_CORRESPONDENCE_PIPELINE_RESET", None)',
            "",
            "!{sys.executable} scripts/run_pipeline.py --config config.yaml",
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 6. Dashboard logs (optional)",
            "",
            "Shows pipeline status from log files in `runs/logs/`.",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "from pathlib import Path",
            "",
            "if 'find_repo_root' not in globals():",
            "    def find_repo_root() -> Path:",
            "        cwd = Path.cwd().resolve()",
            "        for p in [cwd, *cwd.parents]:",
            '            if (p / "pyproject.toml").is_file():',
            "                return p",
            "        raise RuntimeError('pyproject.toml not found — run cell 1 first.')",
            "",
            "REPO = str(find_repo_root())",
            "os.chdir(REPO)",
            "",
            "!bash scripts/start_dashboard.sh",
        ]
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Wrote", OUT)
