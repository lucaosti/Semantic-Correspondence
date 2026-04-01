#!/usr/bin/env python3
"""
Generator for AML_Colab.ipynb (Google Colab, end-to-end).

This notebook is designed to run inside Colab and will:
- optionally clean old paths under /content
- clone the repo into /content/Semantic-Correspondence
- mount Google Drive and symlink runs/ + checkpoints/ for persistence
- download + extract SPair-71k into data/SPair-71k
- install the project (editable) with notebook extras
- download pretrained weights into checkpoints/
- write config.yaml and run the pipeline
- display results tables, plots, and qualitative examples

Run from repo root:
  python notebooks/_generate_aml_colab.py

Colab-specific defaults (batch_size, epochs) are lower than the pipeline defaults
documented in documentation.md because Colab T4 GPUs have limited VRAM and session
time. The pipeline resume mechanism handles reconnects.
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

# ── Header ──────────────────────────────────────────────────────────────────
cells.append(
    cell_md(
        [
            "# Semantic Correspondence — Colab (end-to-end)",
            "",
            "This notebook runs **end-to-end on Google Colab**:",
            "",
            "1. Clone the repository and mount Google Drive for persistence",
            "2. Download + extract **SPair-71k**",
            "3. Install the project and download pretrained weights",
            "4. Write `config.yaml` and run the full pipeline (training + evaluation)",
            "5. Display results: PCK tables, multi-block comparison, per-category and per-difficulty analysis, qualitative examples",
            "",
            "**Artifacts** (logs, checkpoints, exports) survive Colab disconnects via Google Drive symlinks.",
        ]
    )
)

# ── 1. CUDA checks ─────────────────────────────────────────────────────────
cells.append(
    cell_md(
        [
            "### 1. Runtime + CUDA sanity checks",
            "",
            "Select a **GPU** runtime: Runtime → Change runtime type → Hardware accelerator → GPU.",
        ]
    )
)
cells.append(
    cell_code(
        [
            "import sys",
            "import subprocess",
            "",
            "result = subprocess.run(",
            "    [",
            "        sys.executable,",
            "        '-c',",
            "        \"import torch;\\nprint(torch.__version__);\\nprint('cuda_available=', torch.cuda.is_available());\\nprint('torch_cuda=', torch.version.cuda)\",",
            "    ],",
            "    capture_output=True,",
            "    text=True,",
            ")",
            "print(result.stdout)",
            "",
            "need_reinstall = ('cuda_available= False' in result.stdout) and ('torch_cuda= None' in result.stdout)",
            "",
            "if need_reinstall:",
            "    print('Reinstalling CUDA-enabled torch/torchvision...')",
            "    subprocess.run(",
            "        [",
            "            sys.executable,",
            "            '-m',",
            "            'pip',",
            "            'install',",
            "            '-q',",
            "            '--upgrade',",
            "            '--force-reinstall',",
            "            'torch',",
            "            'torchvision',",
            "            '--index-url',",
            "            'https://download.pytorch.org/whl/cu121',",
            "        ],",
            "        check=True,",
            "    )",
            "",
            "import torch",
            "",
            "print('torch', torch.__version__)",
            "print('cuda_available=', torch.cuda.is_available())",
            "print('torch_cuda=', torch.version.cuda)",
            "if torch.cuda.is_available():",
            "    print('gpu=', torch.cuda.get_device_name(0))",
        ]
    )
)

# ── 2. Cleanup ──────────────────────────────────────────────────────────────
cells.append(
    cell_md(
        [
            "### 2. (Optional) Cleanup old Colab workspace",
            "",
            "Removes old repo from **ephemeral** storage only (nothing on Google Drive is deleted).",
        ]
    )
)
cells.append(
    cell_code(
        [
            "import shutil",
            "from pathlib import Path",
            "",
            "FORCE_CLEAN = True",
            "",
            "paths_to_remove = [",
            "    Path('/content/Semantic-Correspondence'),",
            "    Path('/content/sample_data'),",
            "]",
            "",
            "if FORCE_CLEAN:",
            "    for p in paths_to_remove:",
            "        if p.exists():",
            "            print('Removing:', p)",
            "            shutil.rmtree(p, ignore_errors=True)",
            "else:",
            "    print('FORCE_CLEAN=False: skipping cleanup')",
        ]
    )
)

# ── 3. Clone ────────────────────────────────────────────────────────────────
cells.append(cell_md(["### 3. Clone repository"]))
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

# ── 4. Drive persistence ───────────────────────────────────────────────────
cells.append(
    cell_md(
        [
            "### 4. Persist `runs/` and `checkpoints/` on Google Drive",
            "",
            "Symlinks ensure training checkpoints and pipeline state survive runtime disconnects.",
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
            "from google.colab import drive",
            "",
            "drive.mount('/content/drive', force_remount=False)",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "BASE_DIR = Path('/content/drive/MyDrive/Colab Notebooks/AML_results')",
            "RUNS_DIR = BASE_DIR / 'runs'",
            "CKPT_DIR = BASE_DIR / 'checkpoints'",
            "",
            "RUNS_DIR.mkdir(parents=True, exist_ok=True)",
            "CKPT_DIR.mkdir(parents=True, exist_ok=True)",
            "",
            "os.chdir(REPO_DIR)",
            "for link_name, target in [('runs', RUNS_DIR), ('checkpoints', CKPT_DIR)]:",
            "    p = REPO_DIR / link_name",
            "    if p.is_symlink() or p.is_file():",
            "        p.unlink()",
            "    elif p.is_dir():",
            "        shutil.rmtree(p)",
            "    os.symlink(str(target), str(p))",
            "    print(f'Linked {p} -> {target}')",
        ]
    )
)

# ── 5. SPair-71k ───────────────────────────────────────────────────────────
cells.append(cell_md(["### 5. Download + extract SPair-71k"]))
cells.append(
    cell_code(
        [
            "import os",
            "import sys",
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
            "        if sys.version_info >= (3, 12):",
            "            tar.extractall(path=DATA_DIR, filter='data')",
            "        else:",
            "            tar.extractall(path=DATA_DIR)",
            "",
            "print('SPAIR_ROOT =', SPAIR_ROOT, '| exists =', SPAIR_ROOT.is_dir())",
        ]
    )
)

# ── 6. Install ──────────────────────────────────────────────────────────────
cells.append(cell_md(["### 6. Install project (editable + notebook extras)"]))
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

# ── 7. Weights ──────────────────────────────────────────────────────────────
cells.append(cell_md(["### 7. Download pretrained weights (DINOv2, DINOv3, SAM)"]))
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

# ── 8. Config ───────────────────────────────────────────────────────────────
cells.append(cell_md(["### 8. Write config.yaml (Colab defaults)"]))
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
            "DRIVE_AML_BASE = Path('/content/drive/MyDrive/Colab Notebooks/AML_results')",
            "CHECKPOINT_DIR_ON_DRIVE = str(DRIVE_AML_BASE / 'checkpoints')",
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
            "        'checkpoint_dir': CHECKPOINT_DIR_ON_DRIVE,",
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
            "        'resume_save_interval': 100,",
            "    },",
            "    'finetune': {",
            "        'last_blocks': [1, 2, 4],",
            "        'epochs': 50,",
            "        'patience': 7,",
            "        'batch_size': 10,",
            "        'lr': 5e-5,",
            "        'weight_decay': 0.01,",
            "        'dinov2_weights': str(REPO_DIR / 'checkpoints' / 'dinov2_vitb14_pretrain.pth'),",
            "        'dinov3_weights': str(REPO_DIR / 'checkpoints' / 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'),",
            "        'sam_checkpoint': str(REPO_DIR / 'checkpoints' / 'sam_vit_b_01ec64.pth'),",
            "    },",
            "    'lora': {",
            "        'rank': 8,",
            "        'alpha': 16.0,",
            "        'lr': 1e-3,",
            "        'last_blocks': 2,",
            "        'epochs': 50,",
            "        'patience': 7,",
            "        'batch_size': 10,",
            "    },",
            "    'workflow_toggles': {",
            "        'run_verify_dataset': True,",
            "        'train_finetune': [True, True, True],",
            "        'train_lora': [True, True, True],",
            "        'run_eval_baseline': [True, True, True],",
            "        'run_eval_baseline_wsa': [True, True, True],",
            "        'run_eval_finetuned_checkpoint': [True, True, True],",
            "        'run_eval_lora_checkpoint': [True, True, True],",
            "        'run_eval_finetuned_wsa': [True, True, True],",
            "        'run_eval_lora_wsa': [True, True, True],",
            "        'run_export_metrics_tables': True,",
            "        'run_pytest': False,",
            "        'pipeline_resume': True,",
            "    },",
            "}",
            "",
            "with open(cfg_path, 'w', encoding='utf-8') as f:",
            "    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)",
            "print('Written:', cfg_path)",
            "print('checkpoint_dir:', cfg['paths']['checkpoint_dir'])",
            "print('last_blocks sweep:', cfg['finetune']['last_blocks'])",
        ]
    )
)

# ── 9. Pipeline ─────────────────────────────────────────────────────────────
cells.append(cell_md(["### 9. Run pipeline"]))
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

# ── 10. Load results ────────────────────────────────────────────────────────
cells.append(
    cell_md(
        [
            "---",
            "",
            "## Results Analysis",
            "",
            "### 10. Load pipeline exports",
        ]
    )
)
cells.append(
    cell_code(
        [
            "import json",
            "import os",
            "from pathlib import Path",
            "",
            "import pandas as pd",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "os.chdir(REPO_DIR)",
            "EXPORTS = REPO_DIR / 'runs' / 'pipeline_exports'",
            "",
            "def _load_json(name):",
            "    p = EXPORTS / name",
            "    if p.is_file():",
            "        with open(p, encoding='utf-8') as f:",
            "            return json.load(f)",
            "    print(f'Not found: {p}')",
            "    return None",
            "",
            "pck_results = _load_json('pck_results.json')",
            "per_category = _load_json('pck_results_per_category.json')",
            "by_difficulty = _load_json('pck_results_by_difficulty_flag.json')",
            "per_image = _load_json('pck_results_per_image.json')",
            "",
            "print(f'Loaded {len(pck_results or [])} evaluation runs.')",
        ]
    )
)

# ── 11. Aggregate PCK table ─────────────────────────────────────────────────
cells.append(cell_md(["### 11. Aggregate PCK comparison table"]))
cells.append(
    cell_code(
        [
            "if pck_results:",
            "    rows = []",
            "    for r in pck_results:",
            "        row = {'name': r['name']}",
            "        row.update(r.get('metrics', {}))",
            "        rows.append(row)",
            "    df_pck = pd.DataFrame(rows)",
            "    display(df_pck.style.format({c: '{:.4f}' for c in df_pck.columns if c.startswith('pck@')}).set_caption('PCK Results'))",
            "else:",
            "    print('No results to display.')",
        ]
    )
)

# ── 12. Multi-block comparison plot ─────────────────────────────────────────
cells.append(cell_md(["### 12. Fine-tuning depth comparison (PDF Stage 2)"]))
cells.append(
    cell_code(
        [
            "import re",
            "import matplotlib.pyplot as plt",
            "",
            "if pck_results:",
            "    ft_rows = []",
            "    for r in pck_results:",
            "        m = re.match(r'(.+)_ft_lb(\\d+)$', r['name'])",
            "        if m:",
            "            backbone = m.group(1)",
            "            lb = int(m.group(2))",
            "            row = {'backbone': backbone, 'last_blocks': lb}",
            "            row.update(r.get('metrics', {}))",
            "            ft_rows.append(row)",
            "",
            "    if ft_rows:",
            "        df_ft = pd.DataFrame(ft_rows).sort_values(['backbone', 'last_blocks'])",
            "        display(df_ft)",
            "",
            "        fig, axes = plt.subplots(1, len(df_ft['backbone'].unique()), figsize=(5 * len(df_ft['backbone'].unique()), 4), squeeze=False)",
            "        for col_idx, (bb, grp) in enumerate(df_ft.groupby('backbone')):",
            "            ax = axes[0, col_idx]",
            "            metric_cols = [c for c in grp.columns if c.startswith('pck@')]",
            "            for mc in metric_cols:",
            "                ax.plot(grp['last_blocks'], grp[mc], marker='o', label=mc)",
            "            ax.set_xlabel('Unfrozen blocks')",
            "            ax.set_ylabel('PCK')",
            "            ax.set_title(bb)",
            "            ax.legend(fontsize=8)",
            "            ax.set_xticks(sorted(grp['last_blocks'].unique()))",
            "        fig.suptitle('PCK vs. number of fine-tuned blocks', fontsize=13)",
            "        fig.tight_layout()",
            "        plt.show()",
            "    else:",
            "        print('No fine-tuned checkpoint results found (names must match *_ft_lb<N>).')",
        ]
    )
)

# ── 13. Per-category breakdown ──────────────────────────────────────────────
cells.append(cell_md(["### 13. Per-category PCK breakdown"]))
cells.append(
    cell_code(
        [
            "import numpy as np",
            "",
            "if per_category:",
            "    rows = []",
            "    for entry in per_category:",
            "        for cat, alphas in entry.get('categories', {}).items():",
            "            row = {'run': entry['name'], 'category': cat}",
            "            row.update(alphas)",
            "            rows.append(row)",
            "    if rows:",
            "        df_cat = pd.DataFrame(rows)",
            "        pck_col = [c for c in df_cat.columns if c.startswith('pck@')]",
            "        if pck_col:",
            "            pivot = df_cat.pivot_table(index='category', columns='run', values=pck_col[0], aggfunc='first')",
            "            fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5), max(6, len(pivot) * 0.4)))",
            "            im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)",
            "            ax.set_xticks(range(len(pivot.columns)))",
            "            ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=8)",
            "            ax.set_yticks(range(len(pivot.index)))",
            "            ax.set_yticklabels(pivot.index, fontsize=8)",
            "            for i in range(pivot.shape[0]):",
            "                for j in range(pivot.shape[1]):",
            "                    v = pivot.values[i, j]",
            "                    if not np.isnan(v):",
            "                        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)",
            "            fig.colorbar(im, ax=ax, label=pck_col[0])",
            "            ax.set_title(f'Per-category {pck_col[0]}')",
            "            fig.tight_layout()",
            "            plt.show()",
            "    else:",
            "        print('No per-category data available.')",
            "else:",
            "    print('Per-category export not found.')",
        ]
    )
)

# ── 14. Per-difficulty analysis ─────────────────────────────────────────────
cells.append(cell_md(["### 14. Per-difficulty analysis"]))
cells.append(
    cell_code(
        [
            "if by_difficulty:",
            "    rows = []",
            "    for entry in by_difficulty:",
            "        run_name = entry['name']",
            "        for flag, buckets in entry.get('data', {}).items():",
            "            for bucket, info in buckets.items():",
            "                summary = info.get('summary', {}).get('image', {})",
            "                for metric_key, val_dict in summary.items():",
            "                    row = {",
            "                        'run': run_name,",
            "                        'flag': flag,",
            "                        'value': int(bucket),",
            "                        'metric': metric_key.replace('custom_', ''),",
            "                        'pck': val_dict.get('all', float('nan')),",
            "                    }",
            "                    rows.append(row)",
            "    if rows:",
            "        df_diff = pd.DataFrame(rows)",
            "        display(df_diff.pivot_table(index=['run', 'metric'], columns=['flag', 'value'], values='pck').round(4))",
            "    else:",
            "        print('No difficulty breakdown data.')",
            "else:",
            "    print('Difficulty export not found.')",
        ]
    )
)

# ── 15. Qualitative examples ───────────────────────────────────────────────
cells.append(cell_md(["### 15. Qualitative correspondence examples"]))
cells.append(
    cell_code(
        [
            "import torch",
            "from data.dataset import PreprocessMode, SPair71kPairDataset, spair_collate_fn",
            "from models.common import DenseFeatureExtractor, predict_correspondences_cosine_argmax",
            "from evaluation.visualize import visualize_correspondences",
            "",
            "REPO_DIR = Path('/content/Semantic-Correspondence')",
            "spair_root = str(REPO_DIR / 'data' / 'SPair-71k')",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "",
            "ds = SPair71kPairDataset(",
            "    spair_root=spair_root,",
            "    split='test',",
            "    preprocess=PreprocessMode.FIXED_RESIZE,",
            "    output_size_hw=(784, 784),",
            "    normalize=True,",
            ")",
            "",
            "extractor = DenseFeatureExtractor(backbone='dinov2_vitb14').to(device).eval()",
            "",
            "NUM_EXAMPLES = 4",
            "indices = list(range(0, min(len(ds), NUM_EXAMPLES * 50), max(1, len(ds) // NUM_EXAMPLES)))[:NUM_EXAMPLES]",
            "",
            "for idx in indices:",
            "    sample = ds[idx]",
            "    src = sample['src_img'].unsqueeze(0).to(device)",
            "    tgt = sample['tgt_img'].unsqueeze(0).to(device)",
            "    src_kps = sample['src_kps']",
            "    tgt_kps = sample['tgt_kps']",
            "    pck_thr = float(sample['pck_threshold_bbox'])",
            "",
            "    with torch.no_grad():",
            "        feat_src, meta_src = extractor(src)",
            "        feat_tgt, meta_tgt = extractor(tgt)",
            "        out = predict_correspondences_cosine_argmax(",
            "            feat_src, feat_tgt, src_kps,",
            "            img_hw=(784, 784), img_hw_src=(784, 784), img_hw_tgt=(784, 784),",
            "        )",
            "        pred_tgt = out['pred_tgt_xy']",
            "",
            "    fig = visualize_correspondences(",
            "        sample['src_img'], sample['tgt_img'],",
            "        src_kps, pred_tgt, tgt_kps,",
            "        pck_threshold=pck_thr, alpha=0.1,",
            "        title=f'Pair {idx} (category: {sample.get(\"category\", \"?\")})',",
            "    )",
            "    plt.show()",
            "    plt.close(fig)",
        ]
    )
)

# ── Notebook assembly ──────────────────────────────────────────────────────
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
