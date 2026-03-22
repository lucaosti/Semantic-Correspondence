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
            "Presupposto: **repository** e **dataset** (`data/SPair-71k/`) sono già sulla macchina. Il notebook **non** clona, **non** scarica e **non** estrae il dataset.",
            "",
            "Flusso: **1** repo+GPU → **2** cartelle di progresso nel progetto → **3** `pip install` → **4** pesi/config → **5** pipeline.",
            "",
            "**Progresso** — Log, `pipeline_state.json`, checkpoint e pesi stanno in **`runs/`** e **`checkpoints/`** nella root del repo (vedi `.gitignore`).",
            "",
            "**GPU** — Cella 1 abilita ottimizzazioni NVIDIA; il `config.yaml` usa `device: cuda` e `num_workers: -1` (auto per CUDA in `utils.hardware`).",
        ]
    )
)

cells.append(cell_md(["### 1. Root del repository e controllo GPU"]))

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
            '        "WARNING: CUDA non disponibile. Installa PyTorch con CUDA e verifica nvidia-smi."',
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
            '    print("NVIDIA: cudnn.benchmark=True, cuda:0 — allinea a device: cuda nel config.yaml")',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 2. Progresso nel progetto (`runs/` e `checkpoints/`)",
            "",
            "Tutto ciò che la pipeline scrive (log, stato, checkpoint addestrati, pesi scaricati dagli script) resta **dentro il repository** in `runs/` e `checkpoints/` (non symlink, non Drive, non home).",
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
            "REPO = Path(find_repo_root())",
            "os.chdir(REPO)",
            "",
            'for name in ("runs", "checkpoints"):',
            "    (REPO / name).mkdir(parents=True, exist_ok=True)",
            "",
            'print("Progresso e artefatti nel repo:")',
            'print(" ", REPO / "runs")',
            'print(" ", REPO / "checkpoints")',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 3. Installazione pacchetto (`pip install -e`)",
            "",
            "Solo dipendenze Python del progetto. Il dataset non viene toccato.",
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
            '!python -m pip install -q -e ".[notebook]"',
            "",
            'os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")',
            'print("OK:", REPO)',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 4. Pesi pre-addestrati e `config.yaml`",
            "",
            "| Parametro | Valori | Effetto |",
            "|-----------|--------|---------|",
            '| `PIPELINE_RUN_MODE` | `"full"` | verify + fine-tuning + eval + export |',
            '| | `"finetune_only"` | solo `train_finetune` sui tre backbone |',
            "| `START_FROM_SCRATCH` | `True` / `False` | da zero (no resume) vs resume |",
            "| `RESTORE_CONFIG_FROM_ARTIFACTS` | `True` | ripristina da `runs/notebook_config.yaml` nel repo e applica i parametri sopra |",
            "",
            "Backup automatico del config in **`runs/notebook_config.yaml`**. Hyperparametri: `_build_fresh_config_dict()`.",
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
            '            "alphas": [0.05, 0.1, 0.15],',
            '            "wsa_window": 5,',
            '            "wsa_temperature": 1.0,',
            '            "log_batch_interval": 100,',
            "        },",
            '        "finetune": {',
            '            "last_blocks": 2,',
            '            "epochs": 50,',
            '            "patience": 10,',
            '            "dinov2_weights": d2,',
            '            "dinov3_weights": d3,',
            '            "sam_checkpoint": sam,',
            "        },",
            '        "lora": {"rank": 16, "epochs": 5, "patience": 3},',
            '        "workflow_toggles": {},',
            "    }",
            "    _apply_run_mode_to_cfg(cfg)",
            "    return cfg",
            "",
            "",
            "if RESTORE_CONFIG_FROM_ARTIFACTS:",
            "    if not os.path.isfile(BACKUP_CFG):",
            "        raise FileNotFoundError(",
            '            "Manca runs/notebook_config.yaml. Esegui prima con RESTORE_CONFIG_FROM_ARTIFACTS=False."',
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
            "    !python scripts/download_pretrained_weights.py",
            "",
            '    d2 = f"{REPO}/checkpoints/dinov2_vitb14_pretrain.pth"',
            '    d3 = f"{REPO}/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"',
            '    sam = f"{REPO}/checkpoints/sam_vit_b_01ec64.pth"',
            "    for label, p in (('DINOv2', d2), ('DINOv3', d3), ('SAM', sam)):",
            '        print(label, "OK" if os.path.isfile(p) else "MISSING", p)',
            "",
            "    cfg = _build_fresh_config_dict()",
            "    with open(cfg_path, 'w', encoding='utf-8') as f:",
            "        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)",
            '    print("Written:", cfg_path)',
            "",
            "shutil.copy(cfg_path, BACKUP_CFG)",
            'print("Config salvato anche in:", BACKUP_CFG)',
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
            "### 5. Esecuzione pipeline",
            "",
            "Rieseguire la **sezione 4** dopo ogni modifica ai parametri. Con `START_FROM_SCRATCH = True` viene impostato `SEMANTIC_CORRESPONDENCE_PIPELINE_RESET`.",
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
            "!python scripts/run_pipeline.py --config config.yaml",
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 6. Dashboard log (opzionale)",
            "",
            "Mostra lo stato della pipeline sui file di log in `runs/logs/`.",
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
