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
# 1. Clone and setup
git clone https://github.com/lucaosti/Semantic-Correspondence.git
cd Semantic-Correspondence
bash scripts/bootstrap_venv.sh
source .venv/bin/activate

# 2. Verify dataset (SPair-71k under data/SPair-71k/)
python scripts/verify_dataset.py

# 3. Download pretrained weights
python scripts/download_pretrained_weights.py
bash scripts/download_sam_vit_b.sh

# 4. Run the full pipeline
python scripts/run_pipeline.py
# or with a YAML config:
python scripts/run_pipeline.py --config config.yaml
```

## Google Colab

Use **`AML_Colab.ipynb`** at the repository root. It is self-contained and handles:
cloning, SPair-71k download, dependency installation, weight download, pipeline execution, and results analysis.

Artifacts persist on Google Drive via symlinks (`runs/`, `checkpoints/`), so training resumes across Colab disconnects.

---

## Repository Structure

| Path | Role |
|------|------|
| `models/` | Backbone implementations + shared utilities (matching, LoRA, WSA) |
| `training/` | Losses, training engines, unfreeze, early stopping |
| `evaluation/` | PCK metrics, experiment runner, keypoint visualization |
| `data/` | SPair-71k dataset, preprocessing, data interface |
| `scripts/` | CLI tools and `run_pipeline.py` orchestrator |
| `docs/` | Project rules (`info.md`), literature review (`references.md`) |
| `notebooks/` | Notebook generators, example configs |
| `AML_Colab.ipynb` | Google Colab end-to-end notebook (generated) |
| `AML.ipynb` | Local Jupyter notebook (generated) |
| `documentation.md` | Full technical reference |

## Configuration

The pipeline is configured via a YAML file or in-script constants. Key parameters:

```yaml
finetune:
  last_blocks: [1, 2, 4]    # Block count sweep (PDF Stage 2)
  epochs: 50
  batch_size: 10             # Reduce for limited VRAM
  lr: 5e-5
lora:
  rank: 8
  alpha: 16.0
  lr: 1e-3
  batch_size: 10
workflow_toggles:
  run_eval_finetuned_wsa: [true, true, true]   # WSA on fine-tuned models
  run_eval_lora_wsa: [true, true, true]         # WSA on LoRA models
```

See `documentation.md` for the full configuration reference.

## Evaluation

**PCK@alpha** (Percentage of Correct Keypoints) with default thresholds: **0.05, 0.1, 0.2**.

- `test` split for final benchmarks; `val` for model selection
- Per-image, per-point, per-category, and per-difficulty breakdowns via SD4Match backend
- Exports: `runs/pipeline_exports/pck_results.json`, `pck_results_per_category.json`, etc.

## Documentation

- **[`documentation.md`](documentation.md)** — full technical reference (architecture, defaults, maintenance)
- **[`docs/info.md`](docs/info.md)** — project rules (splits, weights policy, WSA constraints)
- **[`docs/references.md`](docs/references.md)** — literature review and references
