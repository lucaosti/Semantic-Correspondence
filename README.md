# Semantic Correspondence on SPair-71k

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/license-academic-green)

Dense keypoint correspondence between semantically similar images, evaluated on **SPair-71k**. This project implements four progressive stages from training-free matching to parameter-efficient adaptation, using three Vision Transformer backbones.

## Table of Contents

- [Project Stages](#project-stages) | [Backbones](#backbones) | [Quick Start](#quick-start) | [Google Colab](#google-colab)
- [Repository Structure](#repository-structure) | [Configuration](#configuration) | [Evaluation](#evaluation) | [Documentation](#documentation)

---

## Project Stages

| Stage | Method | Description |
|-------|--------|-------------|
| **1** | Training-free baseline | Cosine similarity on dense ViT features + argmax |
| **2** | Light fine-tuning | Unfreeze last N transformer blocks, Gaussian CE loss; sweeps over multiple block counts |
| **3** | Window soft-argmax (WSA) | Sub-pixel refinement at inference (applied on top of all methods) |
| **4** | LoRA | Low-Rank Adaptation on late MLP layers; parameter-efficient alternative |

## Backbones

| Backbone | Variant | Patch size | Code path |
|----------|---------|------------|-----------|
| DINOv2 | ViT-B/14 | 14 | `models/dinov2/` |
| DINOv3 | ViT-B/16 | 16 | `models/dinov3/` |
| SAM | ViT-B (image encoder) | — (1024×1024 internal) | `models/sam/` |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/lucaosti/Semantic-Correspondence.git
cd Semantic-Correspondence
pip install -e ".[dev]"

# 2. Verify dataset (SPair-71k under data/SPair-71k/)
python scripts/verify_dataset.py

# 3. Download pretrained weights (DINOv2, DINOv3, SAM)
python scripts/download_pretrained_weights.py

# 4. Run the full pipeline
python scripts/run_pipeline.py
# or with a YAML config:
python scripts/run_pipeline.py --config config.yaml
```

## Notebooks

The project ships **three** Jupyter notebooks at the repository root, with a clear separation of concerns:

| Notebook | Role |
|----------|------|
| **`AML_Local.ipynb`** | Training & orchestration on a local machine (CUDA / MPS / CPU). |
| **`AML_Colab.ipynb`** | Training & orchestration on Google Colab; persists `runs/` and `checkpoints/` on Drive via symlinks so the pipeline resumes across disconnects. |
| **`AML_Analysis.ipynb`** | **Analysis only.** Reads local `runs/pipeline_exports/` and `checkpoints/` and produces paper-ready tables (LaTeX/Markdown/CSV) — including bootstrap 95% CIs, McNemar significance, per-keypoint and literature-baseline tables, an auto-generated hyperparameter table — figures (PDF + PNG, all at dpi=300, colormap **viridis**) covering Δ-PCK, error CDF, calibration, per-category radar, per-difficulty Δ, WSA-window and DINO-layer sensitivity, training curves, **PCA-RGB feature visualizations**, qualitative grids, similarity heatmaps, failure cases, and an efficiency table, under `runs/paper_figures/`. |

### Workflow A — local end-to-end

```text
AML_Local.ipynb  →  runs/pipeline_exports/ + checkpoints/  →  AML_Analysis.ipynb
```

### Workflow B — train on Colab, analyze locally

1. Run **`AML_Colab.ipynb`**. The pipeline writes to `MyDrive/Colab Notebooks/AML_results/{runs,checkpoints}/`.
2. Download `runs/pipeline_exports/`, `runs/logs/`, and `checkpoints/` from that Drive folder into the corresponding locations of your **local** repo clone.
3. Open **`AML_Analysis.ipynb`** locally. No GPU is required: qualitative inference runs on CPU/MPS, all other figures are generated from JSON exports.

The two training notebooks **do not** produce paper figures any more — they only write metric exports, history files, and stage-event logs. All analysis lives in `AML_Analysis.ipynb` and the reusable modules `evaluation/figures.py` + `evaluation/qualitative.py`.

---

## Repository Structure

| Path | Role |
|------|------|
| `models/` | Backbone implementations + shared utilities (matching, LoRA, WSA) |
| `training/` | Losses, training engines, unfreeze, early stopping |
| `evaluation/` | PCK metrics, experiment runner, keypoint visualization |
| `data/` | SPair-71k dataset, preprocessing, data interface |
| `scripts/` | CLI tools and `run_pipeline.py` orchestrator |
| `AML_Local.ipynb` | Local training & orchestration notebook (capability-adaptive) |
| `AML_Colab.ipynb` | Google Colab training & orchestration notebook (Drive symlinks) |
| `AML_Analysis.ipynb` | Analysis-only notebook (paper-ready tables, plots, qualitative figures) |
| `documentation.md` | Full technical reference |

## Configuration

The pipeline is configured via a YAML file or in-script constants. Key parameters:

```yaml
runtime:
  device: cuda
  precision: auto
  num_workers: -1

finetune:
  last_blocks: [1, 2, 4]    # Block count sweep (PDF Stage 2)
  epochs: 50
  batch_size: 20             # Scalar fallback (PDF default)
  batch_size_by_backbone:
    dinov2_vitb14: 20
    dinov3_vitb16: 20
    sam_vit_b: 4
  lr: 5e-5
lora:
  rank: 8
  alpha: 16.0
  lr: 1e-3
  batch_size: 20             # Scalar fallback (PDF default)
  batch_size_by_backbone:
    dinov2_vitb14: 20
    dinov3_vitb16: 20
    sam_vit_b: 4
workflow_toggles:
  run_eval_finetuned_wsa: [true, true, true]   # WSA on fine-tuned models
  run_eval_lora_wsa: [true, true, true]         # WSA on LoRA models
```

See `documentation.md` for the full configuration reference.

## Evaluation

**PCK@alpha** (Percentage of Correct Keypoints) with default thresholds: **0.05, 0.1, 0.2**.

- `test` split for final benchmarks; `val` for model selection
- Per-image, per-point, per-category, per-difficulty, and per-keypoint-name breakdowns via SD4Match metrics on the project's own `SPair71kPairDataset`
- Inference is wrapped in `torch.inference_mode()`, padded keypoints are skipped via `valid_mask`, and the eval DataLoader uses `persistent_workers` + `prefetch_factor`
- Exports: `runs/pipeline_exports/pck_results.json`, `pck_results_per_category.json`, `pck_results_per_keypoint.json`, `pck_errors_per_pair.json` (per-pair normalized L2 errors for CDF / paired-bootstrap), `pck_results_wsa_sweep.json`, `pck_results_layer_sweep.json`, etc.
- Run **`AML_Analysis.ipynb`** to turn those exports into LaTeX tables, paper-ready figures, and qualitative overlays under `runs/paper_figures/`.

## Documentation

- **[`documentation.md`](documentation.md)** — full technical reference (architecture, defaults, maintenance)
