# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantic correspondence on SPair-71k: predict dense keypoint matches between image pairs in the same object category. Four progressive stages from training-free baseline to LoRA fine-tuning, across three ViT backbones (DINOv2, DINOv3, SAM).

## Setup & Installation

```bash
pip install -e ".[dev]"          # Core + pytest + ruff
pip install -e ".[notebook]"     # Add Jupyter + pandas
```

PyTorch must be installed separately using the [official selector](https://pytorch.org/get-started/locally/) for the target hardware (CUDA/MPS/CPU).

```bash
python scripts/verify_dataset.py                 # Validate SPair-71k under data/SPair-71k/
python scripts/download_pretrained_weights.py    # Fetch DINOv2, DINOv3, SAM weights
```

## Running

```bash
# Full orchestrated pipeline (all stages, all backbones)
python scripts/run_pipeline.py
python scripts/run_pipeline.py --config config.yaml   # YAML override

# Standalone training (unified entry point)
python scripts/train.py --mode finetune --backbone dinov2_vitb14 --last-blocks 2
python scripts/train.py --mode lora --backbone dinov2_vitb14 --rank 8

# Tests
pytest tests/
pytest tests/test_matching.py    # Single test file
```

## Architecture

### Four Stages

| Stage | Method | Entry point |
|-------|--------|-------------|
| 1 | Training-free (cosine similarity + argmax) | `models/common/matching.py` |
| 2 | Fine-tune last N transformer blocks (Gaussian CE loss) | `scripts/train.py --mode finetune` |
| 3 | Window Soft-Argmax (WSA) at inference, no retraining | `models/common/window_soft_argmax.py` |
| 4 | LoRA on late MLP layers | `scripts/train.py --mode lora` + `models/common/lora.py` |

### Core Inference Flow

1. **Feature extraction** — `models/common/dense_extractor.py` (`DenseFeatureExtractor`) wraps all three backbones into a unified interface, producing L2-normalized `(B, C, Hf, Wf)` tensors.
2. **Matching** — `models/common/matching.py`: bilinear-sample source features at keypoint locations → cosine similarity against target feature map → argmax.
3. **Optional refinement** — WSA sub-pixel refinement applied at inference over a local window.

### Training Flow

- **Unfreezing** (`training/unfreeze.py`): freeze all blocks, then unfreeze last N.
- **Loss** (`training/losses.py` + `training/engine.py`): Gaussian CE — build 2D Gaussian target from GT keypoints, compute log-softmax on similarity logits, minimize negative sum.
- **Config dataclasses** in `training/config.py`: `FinetuneConfig`, `LoRAConfig`, `EarlyStoppingConfig`, `TrainPaths`.
- **Resume** via `*_resume.pt` checkpoints (model + optimizer + epoch state) managed in `scripts/_training_common.py`.

### Pipeline Orchestration

`scripts/run_pipeline.py` chains all stages for all backbones × block counts. It tracks completed stages in `runs/pipeline_state.json` (skip on resume), detects config changes via fingerprinting, and logs to `runs/logs/pipeline_<timestamp>.log`.

### Data

`data/dataset.py` (`SPair71kPairDataset`): three splits (`train`/`val`/`test`). Preprocessing modes: `FIXED_RESIZE`, `LETTERBOX_PATCH_GRID`, `SCALE_LONGEST_ROUND`. Keypoints padded to MAX_KEYPOINTS=20; invalid slots use `INVALID_KP_COORD = −2.0`. Dataset root resolved via `SPAIR_ROOT` env var or `data/SPair-71k/` default (`data/paths.py`).

### Evaluation

`evaluation/experiment_runner.py` (`run_spair_pck_eval()`): PCK@alpha at thresholds {0.05, 0.1, 0.2}, per-image/point/category/difficulty, using the SD4Match backend. Exports JSON/CSV to `runs/pipeline_exports/`.

### Hardware

Auto-detects CUDA → MPS → CPU. MPS requires `grid_sample` padding mode `"zeros"` (not `"border"`). Worker count auto-tuned from CPU count (`utils/`).

## Key Files

| File | Role |
|------|------|
| `scripts/run_pipeline.py` | Main orchestrator (start here for full workflow) |
| `models/common/dense_extractor.py` | Unified multi-backbone feature extractor |
| `models/common/matching.py` | Cosine similarity matching + argmax |
| `training/engine.py` | Batched Gaussian CE loss computation |
| `evaluation/experiment_runner.py` | PCK evaluation orchestration |
| `data/dataset.py` | SPair-71k dataset + preprocessing + collate |
| `documentation.md` | Authoritative technical reference (keep in sync with code) |
| `docs/info.md` | Project rules: splits policy, WSA constraints, weights policy |

## Important Constraints (from `docs/info.md`)

- `test` split is for final benchmarks only; use `val` for model selection and hyperparameter tuning.
- WSA is inference-only — never applied during training.
- Pretrained weights must be downloaded via `scripts/download_pretrained_weights.py`; do not commit weight files.
- `documentation.md` must stay in sync with code changes — update it when modifying architecture or defaults.
