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
            "# Semantic Correspondence — Linux + NVIDIA GPU (Jupyter)",
            "",
            "Flusso standard: **cella 1** (repo + GPU) → artefatti → `pip install` → pesi/config → pipeline.",
            "",
            "**Repo e dataset già presenti** — Non serve clonare né scaricare nulla: salta le sezioni **2** e **3** (o esegui solo la 3 per una verifica rapida). Serve `data/SPair-71k/` già estratto (o `.tar.gz` da estrarre in sezione 5).",
            "",
            "**Prerequisiti**",
            "- Linux, driver NVIDIA (`nvidia-smi`), Jupyter avviato dal venv dove hai **PyTorch CUDA** ([pytorch.org](https://pytorch.org))",
            "",
            "**Ottimizzazione GPU** — Cella 1: `cudnn.benchmark`, GPU primaria, `config.yaml` con `device: cuda` e `num_workers: -1` (scelta automatica aggressiva dei worker DataLoader in modalità CUDA, come in `utils.hardware`). Training resta a `batch_size=1` per la loss gaussiana.",
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
            "### 2. Repository",
            "",
            "Se hai già questa copia del progetto, **non fare nulla**. Altrimenti, da terminale: `git clone …` e riapri il notebook dalla cartella del clone.",
        ]
    )
)

cells.append(cell_code(['# Nessuna azione richiesta se il repo è già clonato.', 'pass']))

cells.append(
    cell_md(
        [
            "### 3. Dataset SPair-71k (già presente = salta)",
            "",
            "Se `data/SPair-71k/` è già estratto e valido, **non serve** eseguire questa cella.",
            "",
            "Solo se ti manca l'archivio: imposta `FETCH_SPAIR_IF_MISSING = True` per scaricare `SPair-71k.tar.gz` (richiede spazio e rete).",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "from pathlib import Path",
            "",
            "FETCH_SPAIR_IF_MISSING = False",
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
            'data_dir = Path(REPO) / "data"',
            "data_dir.mkdir(parents=True, exist_ok=True)",
            'dataset_url = "https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"',
            'archive = data_dir / "SPair-71k.tar.gz"',
            'sanity = data_dir / "SPair-71k" / "Layout" / "large" / "test.txt"',
            "",
            "if sanity.is_file():",
            '    print("Dataset già presente e valido:", data_dir / "SPair-71k")',
            "elif not FETCH_SPAIR_IF_MISSING:",
            "    raise FileNotFoundError(",
            '        "Manca SPair-71k estratto. Metti data/SPair-71k/ oppure imposta FETCH_SPAIR_IF_MISSING=True per scaricare."',
            "    )",
            "else:",
            '    print("Download archivio…")',
            "    import urllib.request",
            "    urllib.request.urlretrieve(dataset_url, archive)",
            '    print("Salvato:", archive)',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 4. Artefatti persistenti (fuori dal repo)",
            "",
            "Default: `~/semantic_correspondence_artifacts` con `runs/` e `checkpoints/`, collegate al repo tramite symlink. Cambia `ARTIFACTS_ROOT` se preferisci (es. disco dati).",
            "",
            "Variabile d'ambiente: `SEMANTIC_ARTIFACTS_DIR` (e `SEMANTIC_DRIVE_ARTIFACTS` per compatibilità con il vecchio notebook Colab).",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "import shutil",
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
            'ARTIFACTS_ROOT = str(Path.home() / "semantic_correspondence_artifacts")',
            "",
            "os.makedirs(Path(ARTIFACTS_ROOT) / 'runs', exist_ok=True)",
            "os.makedirs(Path(ARTIFACTS_ROOT) / 'checkpoints', exist_ok=True)",
            "",
            "",
            "def _link_repo_dir_to_target(link_name: str, target_dir: str) -> None:",
            '    link_path = os.path.join(REPO, link_name)',
            "    if os.path.lexists(link_path):",
            "        if os.path.islink(link_path):",
            "            os.unlink(link_path)",
            "        elif os.path.isdir(link_path):",
            "            shutil.rmtree(link_path)",
            "        else:",
            "            os.remove(link_path)",
            "    os.symlink(target_dir, link_path)",
            "",
            "",
            "_link_repo_dir_to_target('runs', os.path.join(ARTIFACTS_ROOT, 'runs'))",
            "_link_repo_dir_to_target('checkpoints', os.path.join(ARTIFACTS_ROOT, 'checkpoints'))",
            "",
            'os.environ["SEMANTIC_ARTIFACTS_DIR"] = ARTIFACTS_ROOT',
            'os.environ["SEMANTIC_DRIVE_ARTIFACTS"] = ARTIFACTS_ROOT',
            "",
            'print("Artifacts:", ARTIFACTS_ROOT)',
            'print("  →", os.path.join(REPO, "runs"))',
            'print("  →", os.path.join(REPO, "checkpoints"))',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 5. `pip install` ed estrazione SPair-71k (se serve)",
            "",
            "Se `data/SPair-71k/` è già completo, l'estrazione viene **saltata** automaticamente.",
        ]
    )
)

cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "import tarfile",
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
            "",
            'data_dir = Path(REPO) / "data"',
            'archive = data_dir / "SPair-71k.tar.gz"',
            'target = data_dir / "SPair-71k"',
            'layout = target / "Layout" / "large" / "test.txt"',
            "",
            "if layout.is_file():",
            '    print("Dataset già presente, nessuna estrazione:", target)',
            "elif archive.is_file():",
            '    print("Estrazione da", archive, "...")',
            "    with tarfile.open(archive, 'r:gz') as tar:",
            "        if sys.version_info >= (3, 12):",
            "            tar.extractall(path=data_dir, filter='data')",
            "        else:",
            "            tar.extractall(path=data_dir)",
            '    print("Estrazione completata.")',
            "else:",
            "    raise FileNotFoundError(",
            '        "Servono data/SPair-71k/ oppure data/SPair-71k.tar.gz (sezione 3 se manca l\'archivio)."',
            "    )",
            "",
            "if not layout.is_file():",
            '    raise RuntimeError(f"Layout SPair non valido (manca {layout})")',
            'print("Dataset pronto:", target)',
        ]
    )
)

cells.append(
    cell_md(
        [
            "### 6. Pesi pre-addestrati e `config.yaml`",
            "",
            "| Parametro | Valori | Effetto |",
            "|-----------|--------|---------|",
            '| `PIPELINE_RUN_MODE` | `"full"` | verify + fine-tuning + eval + export |',
            '| | `"finetune_only"` | solo `train_finetune` sui tre backbone |',
            "| `START_FROM_SCRATCH` | `True` / `False` | da zero (no resume) vs resume |",
            "| `RESTORE_CONFIG_FROM_ARTIFACTS` | `True` | copia `config.yaml` da `ARTIFACTS_ROOT` e applica i parametri sopra |",
            "",
            "Hyperparameters: modifica `_build_fresh_config_dict()` (epoche, `limit_pairs`, …). Per la GPU: `device` e `num_workers` in `runtime`.",
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
            'DA = os.environ.get("SEMANTIC_ARTIFACTS_DIR") or os.environ.get("SEMANTIC_DRIVE_ARTIFACTS")',
            "BACKUP_CFG = os.path.join(DA, 'config.yaml') if DA else None",
            'cfg_path = os.path.join(REPO, "config.yaml")',
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
            "    if not BACKUP_CFG or not os.path.isfile(BACKUP_CFG):",
            "        raise FileNotFoundError(",
            '            "Missing config backup. Run once with RESTORE_CONFIG_FROM_ARTIFACTS=False or copy config.yaml into ARTIFACTS_ROOT."',
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
            "if BACKUP_CFG:",
            "    shutil.copy(cfg_path, BACKUP_CFG)",
            '    print("Config backup:", BACKUP_CFG)',
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
            "### 7. Esecuzione pipeline",
            "",
            "Rieseguire la **sezione 6** dopo ogni modifica ai parametri. Con `START_FROM_SCRATCH = True` viene impostato `SEMANTIC_CORRESPONDENCE_PIPELINE_RESET`.",
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
            "### 8. Dashboard log (opzionale)",
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
