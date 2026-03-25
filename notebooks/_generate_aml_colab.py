#!/usr/bin/env python3
"""
Generator for AML_Colab.ipynb (Google Colab, end-to-end).

This notebook is designed to run inside Colab and will:
- clone the repo into /content/Semantic-Correspondence
- download + extract SPair-71k into data/SPair-71k
- install the project (editable) with notebook extras
- download pretrained weights into checkpoints/
- write config.yaml and run the pipeline

Run from repo root:
  python notebooks/_generate_aml_colab.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "AML_Colab.ipynb"


def cell_md(lines: list[str]) -> dict:
    src: list[str] = []
    for line in lines:
        src.append(line if line.endswith("\n") else line + "\n")
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def cell_code(lines: list[str]) -> dict:
    src: list[str] = []
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
            "# Semantic Correspondence — Colab (end-to-end)",
            "",
            "This notebook runs **end-to-end on Google Colab**:",
            "",
            "1. clone the repository",
            "2. download + extract **SPair-71k** to `data/SPair-71k/`",
            "3. install the project (`pip install -e \".[notebook]\"`)",
            "4. download pretrained weights to `checkpoints/`",
            "5. write `config.yaml` and run `scripts/run_pipeline.py --config config.yaml`",
            "",
            "**Artifacts** (logs, state, exports, checkpoints, weights) are stored under `runs/` and `checkpoints/` in the repo root.",
        ]
    )
)

cells.append(cell_md(["### 1. GPU check (Colab runtime)"]))
cells.append(
    cell_code(
        [
            "import torch",
            "",
            "print('torch', torch.__version__)",
            "print('cuda_available=', torch.cuda.is_available())",
            "if torch.cuda.is_available():",
            "    print('gpu=', torch.cuda.get_device_name(0))",
            "    print('cuda_runtime=', torch.version.cuda)",
            "else:",
            "    print('WARNING: CUDA not available. In Colab: Runtime → Change runtime type → GPU.')",
        ]
    )
)

cells.append(cell_md(["### 2. Clone repository"]))
cells.append(
    cell_code(
        [
            "import os",
            "import subprocess",
            "from pathlib import Path",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "REPO_URL = 'https://github.com/lucaosti/Semantic-Correspondence.git'",
            "",
            "if not REPO_DIR.is_dir():",
            "    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL, str(REPO_DIR)], check=True)",
            "os.chdir(REPO_DIR)",
            "print('REPO =', REPO_DIR)",
        ]
    )
)

cells.append(cell_md(["### 3. Download + extract SPair-71k"]))
cells.append(
    cell_code(
        [
            "import os",
            "import tarfile",
            "import urllib.request",
            "from pathlib import Path",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "os.chdir(REPO_DIR)",
            "",
            "SPAIR_URL = 'https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz'",
            "DATA_DIR = REPO_DIR / 'data'",
            "SPAIR_ROOT = DATA_DIR / 'SPair-71k'",
            "TAR_PATH = DATA_DIR / 'SPair-71k.tar.gz'",
            "",
            "DATA_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "if not SPAIR_ROOT.is_dir():",
            "    if not TAR_PATH.is_file():",
            "        print('Downloading:', SPAIR_URL)",
            "        urllib.request.urlretrieve(SPAIR_URL, TAR_PATH)",
            "        print('Saved:', TAR_PATH)",
            "    else:",
            "        print('Already present:', TAR_PATH)",
            "",
            "    print('Extracting to:', DATA_DIR)",
            "    with tarfile.open(TAR_PATH, 'r:gz') as tar:",
            "        tar.extractall(path=DATA_DIR)",
            "",
            "print('SPAIR_ROOT =', SPAIR_ROOT, '| exists =', SPAIR_ROOT.is_dir())",
        ]
    )
)

cells.append(cell_md(["### 4. Install project (editable + notebook extras)"]))
cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "os.chdir(REPO_DIR)",
            "",
            "!{sys.executable} -m pip install -q -e \".[notebook]\"",
            "",
            "os.environ['PYTHONPATH'] = str(REPO_DIR) + os.pathsep + os.environ.get('PYTHONPATH', '')",
            "print('OK:', REPO_DIR)",
        ]
    )
)

cells.append(cell_md(["### 5. Download pretrained weights (DINOv2, DINOv3, SAM)"]))
cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "os.chdir(REPO_DIR)",
            "",
            "!{sys.executable} scripts/download_pretrained_weights.py",
        ]
    )
)

cells.append(cell_md(["### 6. Write config.yaml (Colab defaults)"]))
cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "import yaml",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "os.chdir(REPO_DIR)",
            "",
            "cfg_path = REPO_DIR / 'config.yaml'",
            "",
            "cfg = {",
            "    'dataset': {",
            "        'backend': 'sd4match',",
            "        'metrics_backend': 'sd4match',",
            "    },",
            "    'paths': {",
            "        'repo_root': str(REPO_DIR),",
            "        'spair_root': str(REPO_DIR / 'data' / 'SPair-71k'),",
            "        'checkpoint_dir': 'checkpoints',",
            "        'output_dir': str(REPO_DIR / 'runs' / 'notebook_exports'),",
            "    },",
            "    'runtime': {",
            "        'device': 'cuda',",
            "        'num_workers': 2,",
            "        'preprocess': 'FIXED_RESIZE',",
            "        'image_height': 784,",
            "        'image_width': 784,",
            "        'limit_pairs': 0,",
            "        'eval_split': 'test',",
            "        'alphas': [0.05, 0.1, 0.2],",
            "        'wsa_window': 5,",
            "        'wsa_temperature': 1.0,",
            "        'log_batch_interval': 100,",
            "    },",
            "    'finetune': {",
            "        'last_blocks': 2,",
            "        'epochs': 100,",
            "        'patience': 10,",
            "        'batch_size': 10,",
            "        'dinov2_weights': str(REPO_DIR / 'checkpoints' / 'dinov2_vitb14_pretrain.pth'),",
            "        'dinov3_weights': str(REPO_DIR / 'checkpoints' / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),",
            "        'sam_checkpoint': str(REPO_DIR / 'checkpoints' / 'sam_vit_b_01ec64.pth'),",
            "    },",
            "    'lora': {",
            "        'rank': 16,",
            "        'epochs': 100,",
            "        'patience': 3,",
            "        'batch_size': 10,",
            "    },",
            "    'workflow_toggles': {",
            "        'run_verify_dataset': True,",
            "        'train_finetune': [True, True, True],",
            "        'train_lora': [False, False, False],",
            "        'run_eval_baseline': [True, True, True],",
            "        'run_eval_baseline_wsa': [False, False, False],",
            "        'run_eval_finetuned_checkpoint': [True, True, True],",
            "        'run_eval_lora_checkpoint': [False, False, False],",
            "        'run_export_metrics_tables': True,",
            "        'run_pytest': False,",
            "        'print_jupyter_notebook_hint': True,",
            "        'pipeline_resume': True,",
            "    },",
            "}",
            "",
            "with open(cfg_path, 'w', encoding='utf-8') as f:",
            "    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)",
            "print('Written:', cfg_path)",
        ]
    )
)

cells.append(cell_md(["### 7. Run pipeline"]))
cells.append(
    cell_code(
        [
            "import os",
            "import sys",
            "from pathlib import Path",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "os.chdir(REPO_DIR)",
            "",
            "!{sys.executable} scripts/run_pipeline.py --config config.yaml",
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

