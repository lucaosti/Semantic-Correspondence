# Semantic Correspondence — Technical Documentation

This document is the **authoritative technical reference** for the repository: architecture, defaults, theory, and operational contracts. **Keep it synchronized with the code:** whenever you change behavior, CLI defaults, pipeline constants, loss semantics, or evaluation logic, update this file in the same change (and `README.md` / `docs/info.md` when user-facing rules shift).

---

## 1. Purpose and scope

### 1.1 Task

**Category-level semantic correspondence:** given two images of the same object category, predict where annotated keypoints in the **source** image map in the **target** image. The project targets the **SPair-71k** benchmark layout and metrics.

### 1.2 Operating modes

| Mode | Description |
|------|-------------|
| **Training-free matching** | L2-normalized dense features, cosine similarity, per-keypoint argmax on the target grid; optional **window soft-argmax** only at inference. |
| **Supervised fine-tuning** | Load pretrained ViT weights; **freeze** early blocks; **unfreeze** the last **N** transformer blocks; minimize Gaussian CE on similarity maps (§6). |
| **LoRA** | Parameter-efficient adaptation on late **MLP** linears (rank/α configurable); backbone mostly frozen except adapters. |

Training is **transfer learning**, not random initialization, when official checkpoints are provided.

### 1.3 Backbones

| Name | Patch | Code |
|------|-------|------|
| DINOv2 ViT-B/14 | 14 | `dinov2_vitb14` → `models/dinov2/` |
| DINOv3 ViT-B/16 | 16 | `dinov3_vitb16` → `models/dinov3/` |
| SAM ViT-B (image encoder only) | (internal 1024×1024) | `sam_vit_b` → `models/sam/` |

Weights are **not** committed to git. Use `scripts/download_pretrained_weights.py` and/or CLI/path flags as described in `README.md`.

---

## 2. Repository layout

| Path | Responsibility |
|------|----------------|
| `data/` | `SPair71kPairDataset`, `spair_collate_fn`, preprocessing, `data/paths.resolve_spair_root` |
| `models/common/` | Matching, coords, LoRA, window soft-argmax, `vit_intermediate`, input norm |
| `models/dinov2/`, `dinov3/`, `sam/` | Backbone implementations; weight loading in `hub_loader.py` / `backbone.py` |
| `training/` | `losses.py`, `engine.py` (batched Gaussian losses), `unfreeze.py`, `early_stopping.py`, `config.py` |
| `evaluation/` | `pck.py`, `baseline_eval.py`, `experiment_runner.py`, `visualize.py` (keypoint visualization), checkpoint helpers |
| `utils/` | `hardware.py`, `pipeline_state.py`, `paths.py` |
| `scripts/` | `run_pipeline.py` (orchestrator), `train_finetune.py`, `train_lora.py`, `verify_dataset.py`, `download_pretrained_weights.py` |
| `docs/` | `info.md` (rules), `references.md` (references). Only `docs/**/*.pdf` is gitignored under `docs/`; `*.md` files are tracked. |
| `runs/`, `checkpoints/` | Gitignored artifacts: logs, exports, downloaded weights, training checkpoints |

**Root:** `AML_Colab.ipynb` (Colab entry point), `AML_Local.ipynb` (local entry point), `README.md`, `requirements.txt`, `pyproject.toml`, `documentation.md`.

---

## 3. Environment

- **Python:** `>= 3.9` (`pyproject.toml`).
- **Install:** `pip install -e .` (or `pip install -e ".[notebook]"` for Jupyter/pandas).
- **Extras:** `pip install -e ".[dev]"` (pytest, ruff), `".[notebook]"` (jupyter, ipykernel, pandas).
- **PyTorch:** Install a wheel matching your hardware; verify CUDA with `torch.cuda.is_available()` and `torch.cuda.get_device_capability(0)` where relevant (`requirements.txt` notes for legacy NVIDIA GPUs).

**Data:** Place SPair-71k under `data/SPair-71k/` or set **`SPAIR_ROOT`**. Run `python scripts/verify_dataset.py` before training.

---

## 4. Data: SPair-71k

### 4.1 On-disk layout

- Pair lists: `Layout/large/{trn,val,test}.txt`.
- Training list file is **`trn.txt`**; the code exposes this as split **`train`**.
- Images: `JPEGImages/<category>/`.
- Pair JSON: `PairAnnotation/<split>/`.

### 4.2 Split policy (mandatory)

| Split | Role |
|-------|------|
| `train` | Optimization only |
| `val` | Validation and **early stopping**; safe for tuning |
| `test` | **Final benchmarks only** — do not select hyperparameters on `test` |

Normative text: `docs/info.md`. Implementation notes: `data/dataset.py`.

### 4.3 Batching

`spair_collate_fn` stacks tensor fields; `pair_id_str` is a `list[str]` of length **B**. Keypoints are padded to **`MAX_KEYPOINTS`** (20); invalid slots use **`INVALID_KP_COORD`** (`-2.0`). Training steps use **B pairs** per optimizer step when `--batch-size` is B.

---

## 5. Inference stack

### 5.1 Features

- **DINO:** `extract_intermediate_dense_grid` fuses intermediate layer outputs into one **`(B, C, Hf, Wf)`** map (`models/common/vit_intermediate.py`).
- **SAM:** `imagenet_to_sam_input` resizes to **1024×1024**, then `extract_dense_grid_sam` (`models/sam/backbone.py`).

### 5.2 Matching

For each valid source keypoint: bilinear sample on source map, cosine similarity against target map (`models/common/matching.py`), then argmax; optional WSA refinement **only in eval**.

### 5.3 Window soft-argmax

**Inference/evaluation only** (`models/common/window_soft_argmax.py`). **Must not** appear in the training loss (`docs/info.md`).

### 5.4 Weights policy

Prefer **official** sources (Meta / upstream project URLs) or local `.pth` files passed via CLI/YAML (`docs/info.md`). **Hugging Face mirrors are allowed only as a fallback** when official URLs fail; when a known hash is available, enforce **SHA256 verification** (see `scripts/download_pretrained_weights.py`).

---

## 6. Training: loss and optimization

### 6.1 Objective (Gaussian CE on similarity maps)

Aligned with SD4Match-style practice (`training/losses.py`):

1. For each valid correspondence, form similarity **logits** over the target feature grid.
2. Build a **2D Gaussian** target in feature space from ground-truth target keypoints (`sigma_feat` in feature units).
3. Minimize **negative sum of target × log-softmax(logits)** over the grid, averaged over keypoints (`gaussian_ce_loss_from_similarity_maps`).

### 6.2 Batched training (`training/engine.py`)

For batch size **B**:

1. Forward **all source** and **all target** images through the backbone in one batched pass each (where applicable).
2. For each index `b ∈ {0,…,B−1}`, compute the scalar pair loss from `feats_src[b:b+1]`, `feats_tgt[b:b+1]`, and keypoints for that pair.
3. Return the **mean** of the B per-pair losses.

This is **not** multi-keypoint batching inside a single similarity matrix; it is **B independent pairs** amortized over shared forward passes.

### 6.3 Batch size and precision defaults

Fine-tuning and LoRA have **separate** batch size controls to avoid silent overwrites.

| Location | Fine-tune default | LoRA default |
|----------|-------------------|--------------|
| `train_finetune.py` / `train_lora.py` | `--batch-size` **100** | `--batch-size` **100** |
| `run_pipeline.py` | `FT_BATCH_SIZE` **100** | `LORA_BATCH_SIZE` **100** |
| `training/config.py` dataclasses | **100** | **100** |
| Colab config (`AML_Colab.ipynb`, H100-oriented) | map: DINOv2=32, DINOv3=32, SAM=8 | map: DINOv2=48, DINOv3=48, SAM=8 |

`run_pipeline.py` also supports optional per-backbone overrides through
`FT_BATCH_SIZE_BY_BACKBONE` and `LORA_BATCH_SIZE_BY_BACKBONE` (YAML:
`finetune.batch_size_by_backbone`, `lora.batch_size_by_backbone`).
When a backbone is not present in the map, the scalar fallback (`FT_BATCH_SIZE` / `LORA_BATCH_SIZE`) is used.

Training precision is controlled by `PRECISION` (`auto`/`fp32`/`bf16`/`fp16`) and passed to both training scripts.
`auto` resolves to `bf16` (or `fp16`) on CUDA and to `fp32` on non-CUDA devices. **Evaluation** uses **batch size 1** for the PCK loop.

### 6.4 Fine-tuning (last blocks) — multi-block sweep

`unfreeze_last_transformer_blocks(model, n_blocks=N)` (`training/unfreeze.py`): after `freeze_all`, only the last **N** modules in `model.blocks` are trainable. Optimizer sees only `requires_grad=True` parameters.

**PDF Stage 2 requirement:** the pipeline sweeps over `LAST_BLOCKS_LIST` (default `[1, 2, 4]`), training a separate checkpoint for each block count per backbone. This produces checkpoints named `<backbone>_lastblocks<N>_best.pt` and generates evaluation specs for each. The multi-block sweep allows comparing how fine-tuning depth affects correspondence quality.

### 6.5 Data augmentation

Photometric augmentation (`build_photometric_pair_transform` in `data/dataset.py`) is enabled by default for
**training** datasets in both `train_finetune.py` and `train_lora.py`. The same random color jitter is applied
identically to source and target images in each pair, following the recommendation in
[`docs/state-of-art.md`](docs/state-of-art.md). Geometric augmentations are **not** applied because they would
require consistent keypoint remapping. Validation datasets are **not** augmented.

### 6.6 LoRA

Injected on late MLP linears (`models/common/lora.py` + backbone-specific hooks). Only adapter weights are updated.

### 6.7 Early stopping and resume

- **Early stopping:** `training/early_stopping.py` on **validation loss** each epoch (`patience` configurable).
- **Resume files:** `checkpoints/<backbone>_lastblocks*_resume.pt` / LoRA equivalents store model, optimizer, epoch, stopper state, and optional `batch_in_epoch` for mid-epoch resume. Mid-epoch resume is only as granular as the save cadence: training scripts write these files every `--resume-save-interval` batches (`AML_Colab.ipynb` uses **25** for Colab disconnects; `AML_Local.ipynb` uses **50**; the pipeline default is **50**).
- **Two-level resume:** (1) pipeline stage skip via `runs/pipeline_state.json` (`PIPELINE_RESUME=True`), and (2) training resume via `checkpoints/*_resume.pt` (`--resume ...` + periodic saves). If you see training restart from batch 0 after an interruption, it usually means no recent `*_resume.pt` was written (save interval too large) or `checkpoints/` was not persisted.

---

## 7. Evaluation: PCK

**PCK@α** (`evaluation/pck.py`): a keypoint is correct if

`‖pred − gt‖₂ ≤ α · pck_threshold`,

where **`pck_threshold`** is the **bounding-box scale** (max side length) in the **same pixel frame** as predictions and ground truth (bbox-normalized PCK). Invalid padded keypoints are masked with `invalid_value` (default `-2.0`).

Default **α** triple in the pipeline: **`(0.05, 0.1, 0.2)`** (`EVAL_ALPHAS` in `run_pipeline.py`, aligned with the course PDF). **`EVAL_LIMIT > 0`** truncates pairs for debugging.

`experiment_runner.py` schedules multiple **eval specs** (baseline, baseline+WSA, fine-tuned per block count, fine-tuned+WSA, LoRA, LoRA+WSA); each run uses the shared baseline evaluator with **batch size 1** data loading.

**WSA on trained models (PDF Stage 3):** the pipeline generates WSA variants for fine-tuned and LoRA checkpoints when `RUN_EVAL_FINETUNED_WSA` / `RUN_EVAL_LORA_WSA` are enabled. This measures the benefit of sub-pixel refinement on top of adapted features.

**Reporting granularity (PDF requirement):** when the SD4Match metrics backend is enabled, the pipeline exports:
- **Aggregate:** `pck_results.json`, `pck_results.csv`
- **Per-image:** `pck_results_per_image.json`
- **Per-point:** `pck_results_per_point.json`
- **Per-category:** `pck_results_per_category.json` + `pck_results_per_category.csv` — macro (`pck@α`) and micro (`pck_pt@α`) PCK per SPair-71k category per eval run
- **Per-difficulty:** `pck_results_by_difficulty_flag.json` (viewpoint, scale, truncation, occlusion)
- **Averaging note:** `pck@α` = per-image mean (macro); `pck_pt@α` = per-keypoint mean (micro) — the standard metric in most SPair-71k papers (Min et al. 2019, CHM, DHPF, CATS). Both are exported to `pck_results.csv` and the per-category CSV.
- **Training loss curves:** `checkpoints/*_history.jsonl` (per-epoch `train_loss`/`val_loss`, written by `_training_common.py`); loaded by `Analysis_Local.ipynb §11` into `figures/training_loss_history.csv`.

---

## 8. Orchestrated pipeline (`scripts/run_pipeline.py`)

### 8.1 Stage order

1. Optional **`verify_dataset`**
2. **Fine-tune** for each enabled backbone × each block count in `LAST_BLOCKS_LIST`
3. **LoRA** for each enabled backbone
4. **PCK** matrix: baseline, baseline+WSA, fine-tuned (per block count), fine-tuned+WSA, LoRA, LoRA+WSA (per flags) + optional **`runs/pipeline_exports/`** JSON/CSV + per-category export
5. Optional **`pytest`**

**Tuple order is always** `(dinov2_vitb14, dinov3_vitb16, sam_vit_b)`.

### 8.2 Default configuration (in-script; YAML overrides)

| Symbol | Typical default | Description |
|--------|-----------------|-------------|
| `LAST_BLOCKS_LIST` | `[1, 2, 4]` | Block counts for fine-tuning sweep (PDF Stage 2) |
| `FT_BATCH_SIZE` / `LORA_BATCH_SIZE` | 20 / 20 | Pairs per step for DINO backbones; SAM overridden to 4 via per-backbone map |
| `FT_BATCH_SIZE_BY_BACKBONE` / `LORA_BATCH_SIZE_BY_BACKBONE` | `{"sam_vit_b": 4}` | Per-backbone batch overrides (SAM encoder is VRAM-heavy) |
| `FT_LR` / `LORA_LR` | `5e-5` / `1e-3` | Learning rates |
| `FT_WEIGHT_DECAY` | `0.01` | Weight decay for fine-tuning |
| `LORA_ALPHA` | `16.0` | LoRA scaling factor |
| `LORA_LAST_BLOCKS` | `2` | LoRA block count |
| `PRECISION` | `auto` | Training precision policy (`auto`/`fp32`/`bf16`/`fp16`); MPS/CPU always resolve to fp32 |
| `FT_EPOCHS` / `LORA_EPOCHS` | 50 | Epochs per training stage (early stopping typically triggers earlier) |
| `FT_PATIENCE` / `LORA_PATIENCE` | 7 | Early stopping patience |
| `LORA_RANK` | 8 | LoRA rank |
| `PREPROCESS` | `FIXED_RESIZE` | Image preprocessing mode |
| `IMAGE_SIZE_BY_BACKBONE` | `{dinov2_vitb14: (518,518), dinov3_vitb16: (512,512), sam_vit_b: (512,512)}` | Per-backbone input size (exact patch-size multiples: 518=37×14, 512=32×16; SAM features always extracted at 1024×1024 internally) |
| `IMAGE_HEIGHT` / `IMAGE_WIDTH` | 784 / 784 | Global fallback size for backbones not listed in `IMAGE_SIZE_BY_BACKBONE` |
| `EVAL_SPLIT` | `test` | Evaluation split |
| `EVAL_LIMIT` | 0 (full split) | Pair limit for debugging |
| `LOG_BATCH_INTERVAL` | 50 | Batch logging frequency |
| `RESUME_SAVE_INTERVAL` | 50 | Mid-epoch resume checkpoint cadence |
| `DINO_LAYER_INDICES` | `4` | Intermediate ViT layer for DINO feature extraction (passed as `--layer-indices`; ignored for SAM) |

**Workflow toggles:** In addition to baseline and checkpoint eval toggles, `RUN_EVAL_FINETUNED_WSA` and `RUN_EVAL_LORA_WSA` enable WSA evaluation on trained checkpoints (PDF Stage 3 combined with Stages 2 and 4).

**Resume:** `PIPELINE_RESUME` uses `runs/pipeline_state.json`. **`fingerprint_from_config`** hashes `_fingerprint_payload()` (all training hyperparameters, block lists, eval flags, etc.). A mismatch **clears** completed steps so incompatible runs are not mixed.

**YAML `--config`:** `_apply_pipeline_yaml` maps sections into module globals. Fine-tuning and LoRA have **separate** scalar batch size fields (`finetune.batch_size` → `FT_BATCH_SIZE`, `lora.batch_size` → `LORA_BATCH_SIZE`) plus optional per-backbone maps (`finetune.batch_size_by_backbone`, `lora.batch_size_by_backbone`). Precision is configured through `runtime.precision` → `PRECISION`. The `finetune.last_blocks` field accepts either a single integer or a list.

**Colab overrides:** `AML_Colab.ipynb` writes `config.yaml` with `runtime.precision: auto`, `num_workers: 1`, frequent `resume_save_interval` (25), H100-oriented per-backbone batch maps (see §6.3), and `epochs: 50` / `patience: 7`.

### 8.3 Environment variables

| Variable | Effect |
|----------|--------|
| `SEMANTIC_CORRESPONDENCE_PIPELINE_RESET` | Non-empty → reset pipeline state at start |
| `SEMANTIC_CORRESPONDENCE_PIPELINE_LOG_FILE_ONLY` | Detached logging without mirroring to terminal |
| `SEMANTIC_CORRESPONDENCE_PREFETCH_CAP` | Caps DataLoader prefetch (`utils/hardware.py`) |
| `SPAIR_ROOT` | Overrides default dataset root |

### 8.4 Logs

- `runs/logs/pipeline_<UTC>.log`, `runs/logs/current.log` symlink, `runs/logs/manifest.tsv`, `runs/logs/stage_events.jsonl`.

---

## 9. Hardware-aware behavior (`utils/hardware.py`)

- **Device resolution:** CUDA → MPS → CPU when device is `None`.
- **Multi-GPU selection:** if `CUDA_VISIBLE_DEVICES` is set, only those GPUs are visible to the process; the first visible one is used as logical `cuda:0`.
- **`num_workers`:** CLI `-1` means auto via `recommended_dataloader_workers` (CUDA: aggressive prefetch; CPU training: smaller pool).
- **CUDA:** `cudnn.benchmark`, TF32 allow where supported, `pin_memory`, `persistent_workers` + `prefetch_factor` when workers > 0.

**MPS:** Some ops (e.g. `grid_sample` with certain `padding_mode`) may fail; use **CPU** or **CUDA** for parity with Linux training/eval.

---

## 10. Notebook and config file

- **`AML_Colab.ipynb`:** Colab entry point (H100-oriented defaults). Clone repo, Google Drive symlinks for `runs/` and `checkpoints/`, SPair-71k, weights, `config.yaml`, pipeline subprocess with a live refresh over `stage_events.jsonl` + log tail, then the same analysis cells as local (PCK table, fine-tune depth plot, per-category heatmap, per-difficulty, qualitative DINOv2).
- **`AML_Local.ipynb`:** same semantics from the repository root: optional `pip install -e ".[notebook]"`, dataset/weights if missing, hardware-aware batch maps (bf16-capable CUDA vs older GPUs vs CPU/MPS), `config.yaml`, pipeline + analysis.
- **Colab-specific overrides:** `runtime.num_workers: 1`, `resume_save_interval` / `log_batch_interval: 25`, `precision: auto` (bf16 on Hopper), per-backbone batch maps as in §6.3 Colab row, `epochs: 50`, `patience: 7`.
- **`config.yaml`:** notebook-generated pipeline configuration. It contains machine-specific absolute paths, so it is **not tracked in git** (see `.gitignore`). The notebook writes it for `scripts/run_pipeline.py --config config.yaml`.
- **YAML reader:** the pipeline applies keys via `_apply_pipeline_yaml` in `run_pipeline.py` (``dataset``, ``workflow_toggles``, etc.). Key runtime overrides include `runtime.dino_layer_indices` → `DINO_LAYER_INDICES` (int, default 4): intermediate ViT layer for DINO feature extraction.
- **SPair-71k download URL (Colab):** `https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz`
- **Pipeline reset (Colab):** in `AML_Colab.ipynb`, the config cell sets `START_FROM_SCRATCH` (default `False`). The run-pipeline cell launches `scripts/run_pipeline.py` via `subprocess.Popen` and sets **`SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1`** when the flag is true (clears `runs/pipeline_state.json` only; does not delete Drive checkpoints).
- **Qualitative cell:** DINOv2 baseline visualization uses `DenseExtractorConfig(..., dinov2_weights_path=...)` with `checkpoints/dinov2_vitb14_pretrain.pth` (no `torch.hub` fetch).

---

## 11. Coding standards

- **Code language:** English for identifiers, docstrings, and comments (`docs/info.md`, `data/dataset.py`).
- **Docs:** `documentation.md` (this file) = system reference; `README.md` = quick start; `docs/info.md` = normative project rules.
- **Lint:** `ruff` in `pyproject.toml` (line length 100, Python 3.9 syntax target).
- **Comments:** Public APIs and non-obvious logic should have clear docstrings; line-level coverage varies across legacy modules—improve when touching code.

---

## 12. Theoretical consistency

- **Gaussian target + cross-entropy on softmax similarities:** A smooth surrogate for location supervision on a grid; consistent with dense correspondence literature citing SD4Match-style objectives.
- **Transfer learning:** Training only last blocks or LoRA matches standard adaptation of foundation models to paired data with limited compute.
- **WSA outside the loss:** Avoids discontinuities from hard argmax in the training objective; refinement is a **test-time** operator.
- **Split discipline:** Holding out `test` for final metrics limits overfitting to benchmark statistics.
- **Batch averaging:** Mean of per-pair losses is standard ERM; if batch size changes by orders of magnitude, consider **learning-rate retuning** (not automated here).

---

## 13. Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Coverage includes dataset workers, pipeline state, smoke imports, backbone construction, SD4Match interface, window soft-argmax, cosine matching / argmax correspondence, LoRA adapters, PCK helpers, preprocessing/keypoint scaling, ViT unfreezing, and training config dataclasses. Tests do **not** guarantee numerical results on full SPair training. After each run, `tests/conftest.py` generates `docs/test_report.md` (replacing any prior version) with a per-file Markdown table of results, durations, and failure details.

---

## 14. Maintenance contract

Update **`documentation.md`** when any of the following change: default CLI/pipeline constants, loss or engine semantics, eval metrics, resume/fingerprint fields, supported backbones, weight acquisition paths, or documented environment variables. Align **`README.md`** and **`docs/info.md`** when user-visible rules or defaults shift.

See also **[`docs/state-of-art.md`](docs/state-of-art.md)** for the state-of-the-art analysis and design rationale that informed implementation choices (backbone selection, loss design, WSA, LoRA, photometric augmentation, PCK thresholds).

---

## 15. Package version

**`semantic-correspondence`:** version **`0.1.0`** in `pyproject.toml`. This document’s accuracy is tied to the git tree at the time of the last edit to **`documentation.md`**.
