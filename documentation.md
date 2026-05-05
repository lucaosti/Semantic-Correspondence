# Semantic Correspondence — Technical Documentation

This document is the **authoritative technical reference** for the repository: architecture, defaults, theory, and operational contracts.

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
| `models/dinov2/`, `dinov3/`, `sam/` | Backbone implementations; weight loading in `hub_loader.py` / `backbone.py`. `models/dinov2/layers/` and `models/sam/modeling/` are vendored upstream code (Meta); upstream comments/FIXMEs are not first-party concerns. |
| `training/` | `losses.py`, `engine.py` (batched Gaussian losses), `unfreeze.py`, `early_stopping.py`, `config.py` |
| `evaluation/` | `experiment_runner.py` (in-house batched PCK runner, also drives the WSA-window and DINO layer-index inference-only ablations), `figures.py` (paper-grade tables/plots: master / per-category / per-difficulty / efficiency tables, training curves, WSA / depth / LoRA-vs-FT plots, WSA-window and DINO-layer sensitivity sweeps), `qualitative.py` (method grids, similarity heatmaps in `viridis`, find-symmetry-ambiguity helper), `visualize.py`, checkpoint helpers |
| `utils/` | `hardware.py`, `pipeline_state.py`, `paths.py` |
| `scripts/` | `run_pipeline.py` (orchestrator), `train.py` (unified `--mode finetune|lora`), `verify_dataset.py`, `download_pretrained_weights.py` |
| `runs/`, `checkpoints/` | Gitignored artifacts: logs, exports, downloaded weights, training checkpoints |

**Root:** `AML_Local.ipynb` (local training/orchestration), `AML_Colab.ipynb` (Colab training/orchestration), `AML_Analysis.ipynb` (paper-grade analysis only), `README.md`, `requirements.txt`, `pyproject.toml`, `documentation.md`.

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

Implementation notes: `data/dataset.py`.

### 4.3 Batching

`spair_collate_fn` stacks tensor fields; `pair_id_str` is a `list[str]` of length **B**. Keypoints are padded to **`MAX_KEYPOINTS`** (20); invalid slots use **`INVALID_KP_COORD`** (`-2.0`). Training steps use **B pairs** per optimizer step when `--batch-size` is B.

---

## 5. Inference stack

### 5.1 Features

All three backbones are wrapped by **`DenseFeatureExtractor`** (`models/common/dense_extractor.py`), which returns an L2-normalized **`(B, C, Hf, Wf)`** tensor regardless of backbone.

- **DINO:** `extract_intermediate_dense_grid` fuses intermediate layer outputs into one **`(B, C, Hf, Wf)`** map (`models/common/vit_intermediate.py`).
- **SAM:** `imagenet_to_sam_input` resizes to **1024×1024**, then `extract_dense_grid_sam` (`models/sam/backbone.py`). The dataset image frame (typically 512×512) is tracked separately from the SAM internal frame; keypoint coordinates are mapped back to the dataset frame inside `DenseFeatureExtractor`.

### 5.2 Matching

For each valid source keypoint: bilinear sample on source map, cosine similarity against target map (`models/common/matching.py`), then argmax; optional WSA refinement **only in eval**.

### 5.3 Window soft-argmax

**Inference/evaluation only** (`models/common/window_soft_argmax.py`). **Must not** appear in the training loss.

> Note: "Stage 3" in the course PDF refers to this post-processing refinement — not a separate training stage. WSA is evaluated by comparing paired baseline vs baseline+WSA (and finetune vs finetune+WSA, LoRA vs LoRA+WSA) eval specs in `scripts/run_pipeline.py`.

### 5.4 Weights policy

Prefer **official** sources (Meta / upstream project URLs) or local `.pth` files passed via CLI/YAML. **Hugging Face mirrors are allowed only as a fallback** when official URLs fail; when a known hash is available, enforce **SHA256 verification** (see `scripts/download_pretrained_weights.py`).

---

## 6. Training: loss and optimization

### 6.1 Objective (Gaussian CE on similarity maps)

Aligned with SD4Match-style practice (`training/losses.py`):

1. For each valid correspondence, form similarity **logits** over the target feature grid.
2. Build a **2D Gaussian** target in feature space from ground-truth target keypoints (`sigma_feat` in feature units).
3. Minimize **negative sum of target × log-softmax(logits)** over the grid, averaged over keypoints (`gaussian_ce_loss_from_similarity_maps`).

### 6.2 Batched training (`training/engine.py`)

For batch size **B**, `correspondence_gaussian_loss` is fully vectorized:

1. **Fused forward** — source and target images are concatenated and pushed through the `DenseFeatureExtractor` in a **single** call (`extractor(torch.cat([src, tgt], 0))`), then split. This halves backbone kernel launches per step.
2. **Vectorized loss** — per-keypoint descriptors are sampled with one `grid_sample` (`(B, N, C)`); similarity maps are computed with one `einsum` against `feats_tgt` (`(B, N, Hf, Wf)`); the 2D Gaussian target is built once on the feature grid; the masked mean of `−Σ target · log_softmax(sims)` is taken in one shot. No Python loop over batch items, no host/device sync inside the step.
3. The validity mask is read against `INVALID_KP_COORD` in the dataset frame (before any rescale to the extractor coord frame), so SAM-style frame conversions remain numerically safe.

This is **B independent pairs** amortized over shared forward passes; the loss value is identical to the previous per-pair implementation modulo associativity reductions.

### 6.3 Batch size and precision defaults

Fine-tuning and LoRA have **separate** batch size controls to avoid silent overwrites.

| Location | Fine-tune default | LoRA default |
|----------|-------------------|--------------|
| `train.py --mode finetune|lora` | `--batch-size` **20** | `--batch-size` **20** |
| `run_pipeline.py` | `FT_BATCH_SIZE` **20** | `LORA_BATCH_SIZE` **20** |
| `training/config.py` dataclasses | **20** | **20** |
| Per-backbone override (pipeline) | `sam_vit_b` → **4** | `sam_vit_b` → **4** |
| Colab config (`AML_Colab.ipynb`) | PDF defaults: DINOv2=20, DINOv3=20, SAM=4 | PDF defaults: DINOv2=20, DINOv3=20, SAM=4 |
| Local on MPS (`AML_Local.ipynb`, derogation) | DINOv2=8, DINOv3=8, SAM=3 | DINOv2=12, DINOv3=12, SAM=3 |
| Local on CPU (`AML_Local.ipynb`, derogation) | DINOv2=4, DINOv3=4, SAM=2 | DINOv2=4, DINOv3=4, SAM=2 |

`run_pipeline.py` also supports optional per-backbone overrides through
`FT_BATCH_SIZE_BY_BACKBONE` and `LORA_BATCH_SIZE_BY_BACKBONE` (YAML:
`finetune.batch_size_by_backbone`, `lora.batch_size_by_backbone`).
When a backbone is not present in the map, the scalar fallback (`FT_BATCH_SIZE` / `LORA_BATCH_SIZE`) is used.
See §6.3 above for the normative batch-size policy.

Training precision is controlled by `PRECISION` (`auto`/`fp32`/`bf16`/`fp16`) and passed to both training scripts.
`auto` resolves to `bf16` (or `fp16`) on CUDA and to `fp32` on non-CUDA devices. **Evaluation** runs batched (default `batch_size=8`) under `torch.inference_mode()` with bf16 autocast on CUDA Ampere+; matching is fully tensorized so PCK aggregation is the only Python-level loop.

The CUDA optimizer step uses `AdamW(..., fused=True)` (both fine-tune and LoRA paths in `scripts/train.py`); on non-CUDA devices the flag is silently dropped.

Throughput tweaks in `scripts/_training_common.run_gaussian_training_loop`: the train DataLoader is built with `drop_last=True` (no trailing partial micro-batch — keeps `torch.compile` shapes stable and gradient-accumulation groups complete) and host-to-device transfers use `non_blocking=True` so they overlap with the previous step's compute (pinned memory is already on by default via `utils.hardware.pin_memory_for`).

**Gradient accumulation.** `scripts/train.py --accumulation-steps N` (or `finetune.accumulation_steps` / `lora.accumulation_steps` in `config.yaml`) divides the per-batch loss by `N` and steps the optimizer every `N` micro-batches, yielding an **effective batch of `batch_size × N`** without extra VRAM. Default `1` (off). Useful on memory-tight devices (e.g. SAM with `B=4` on consumer GPUs) to recover the PDF effective batch.

#### Effect of changing batch size or precision

The Gaussian correspondence loss averages over pairs, so the loss scale is batch-independent; however, smaller batches produce noisier gradient estimates and an effectively higher learning rate. The linear scaling rule would suggest reducing LR proportionally, but AdamW is generally robust enough that moderate changes (e.g. 20 → 8) converge comparably. Expect more oscillation in validation loss and possibly earlier early-stopping triggers with very small batches.

Switching from `bf16` to `fp16` (or vice versa) has negligible impact on final PCK: `bf16` has wider dynamic range but lower mantissa precision, while `fp16` is the reverse. When `fp16` is selected, `GradScaler` is activated automatically to prevent gradient underflow. Occasional skipped optimizer steps (overflow detected) are benign and handled silently. PCK evaluation loads the saved `*_best.pt` checkpoint and is independent of training precision.

### 6.4 Fine-tuning (last blocks) — multi-block sweep

`unfreeze_last_transformer_blocks(model, n_blocks=N)` (`training/unfreeze.py`): after `freeze_all`, only the last **N** modules in `model.blocks` are trainable. Optimizer sees only `requires_grad=True` parameters.

**PDF Stage 2 requirement:** the pipeline sweeps over `LAST_BLOCKS_LIST` (default `[1, 2, 4]`), training a separate checkpoint for each block count per backbone. This produces checkpoints named `<backbone>_lastblocks<N>_best.pt` and generates evaluation specs for each. The multi-block sweep allows comparing how fine-tuning depth affects correspondence quality.

### 6.5 Data augmentation

Photometric augmentation (`build_photometric_pair_transform` in `data/dataset.py`) is enabled by default for
**training** datasets in both training modes (`scripts/train.py --mode {finetune,lora}`). The same random color jitter is applied
identically to source and target images in each pair, following standard practice in dense correspondence literature. Geometric augmentations are **not** applied because they would
require consistent keypoint remapping. Validation datasets are **not** augmented.

### 6.6 LoRA

Injected on late MLP linears (`models/common/lora.py` + backbone-specific hooks). Only adapter weights are updated.

### 6.7 Early stopping and resume

- **Early stopping:** `training/early_stopping.py` on **validation loss** each epoch. Configurable via `--patience` (default **7**) and `--min-delta` (default **0.0** = strict). An epoch counts as improvement only if `val_loss < best − min_delta`; otherwise the bad-epoch counter increments and training stops once it reaches `patience`. Active in both `--mode finetune` and `--mode lora`.
- **Resume preserves stopper state:** `best_value`, `num_bad_epochs`, `best_epoch`, `patience`, `min_delta`, `mode` are serialized into `*_resume.pt` and restored on `--resume` (`scripts/_training_common.py`). So resuming never resets the patience counter — if training was interrupted after 4 bad epochs, it continues from 4 and stops after 3 more stagnant ones (assuming `patience=7`). No wasted epochs.
- **Resume files:** `checkpoints/<backbone>_lastblocks*_resume.pt` / LoRA equivalents store model, optimizer, epoch, stopper state, and optional `batch_in_epoch` for mid-epoch resume. Mid-epoch resume is only as granular as the save cadence: training scripts write these files every `--resume-save-interval` batches (`AML_Local.ipynb` and `AML_Colab.ipynb` both set **500**; the pipeline / `config.yaml` default is **50**).
- **Two-level resume:** (1) pipeline stage skip via `runs/pipeline_state.json` (`PIPELINE_RESUME=True`), and (2) training resume via `checkpoints/*_resume.pt` (`--resume ...` + periodic saves). If you see training restart from batch 0 after an interruption, it usually means no recent `*_resume.pt` was written (save interval too large) or `checkpoints/` was not persisted.
- **Notebook auto-detect (fresh vs resume):** Both notebooks set `START_FROM_SCRATCH` automatically based on persistence. `AML_Local.ipynb` checks if `runs/` and `checkpoints/` exist under the repo root. `AML_Colab.ipynb` checks Drive: resumes if `.../AML_results/runs/pipeline_state.json` exists **or** `.../AML_results/checkpoints/` contains any `*.pt`. Cold start = delete those on Drive (or the local folders) and re-run the notebook. No manual toggle needed in either case.

---

## 7. Evaluation: PCK

**PCK@α** (in-house batched evaluator in `evaluation/experiment_runner.py`): a keypoint is correct if

`‖pred − gt‖₂ ≤ α · pck_threshold`,

where **`pck_threshold`** is the **bounding-box scale** (max side length) in the **same pixel frame** as predictions and ground truth (bbox-normalized PCK). Invalid padded keypoints are masked with `invalid_value` (default `-2.0`).

Default **α** triple in the pipeline: **`(0.05, 0.1, 0.2)`** (`EVAL_ALPHAS` in `run_pipeline.py`, aligned with the course PDF). **`EVAL_LIMIT > 0`** truncates pairs for debugging.

`experiment_runner.py` schedules multiple **eval specs** (baseline, baseline+WSA, fine-tuned per block count, fine-tuned+WSA, LoRA, LoRA+WSA, plus the WSA-window and DINO layer-index inference-only ablations described in §8.2). Each run uses the project's own `SPair71kPairDataset` (the legacy SD4Match vendor was removed), wraps inference in `torch.inference_mode()` with bf16 autocast on CUDA Ampere+, fuses the source/target forward (`extractor(torch.cat([src, tgt], 0))`), and runs a fully batched matcher (one `grid_sample` plus one `einsum` per batch). Data loading uses `batch_size=8` (configurable per spec), `persistent_workers=True`, and `prefetch_factor` (via `utils.hardware.dataloader_extra_kwargs`); padded keypoints are filtered by per-pair `n_valid` after the matcher returns.

**WSA on trained models (PDF Stage 3):** the pipeline generates WSA variants for fine-tuned and LoRA checkpoints when `RUN_EVAL_FINETUNED_WSA` / `RUN_EVAL_LORA_WSA` are enabled. This measures the benefit of sub-pixel refinement on top of adapted features.

**Reporting granularity (PDF requirement):** the pipeline exports:
- **Aggregate:** `pck_results.json`, `pck_results.csv`
- **Per-category:** `pck_results_per_category.json` + `pck_results_per_category.csv` — macro (`pck@α`) and micro (`pck_pt@α`) PCK per SPair-71k category per eval run.
- **Per-difficulty:** `pck_results_by_difficulty_flag.json` (viewpoint, scale, truncation, occlusion).
- **WSA / layer ablations:** `pck_results_wsa_sweep.json` (WSA window {3,5,7,9} on the baseline) and `pck_results_layer_sweep.json` (DINO intermediate layer index in `EVAL_LAYER_INDICES_LIST`) — inference-only re-runs of the baseline, no extra training.
- **Averaging note:** `pck@α` = per-image mean (macro); `pck_pt@α` = per-keypoint mean (micro) — the standard metric in most SPair-71k papers (Min et al. 2019, CHM, DHPF, CATs). Both land in `pck_results.csv` and the per-category CSV.
- **Training loss curves + LR:** `checkpoints/*_history.jsonl` (per-epoch `train_loss`/`val_loss`/`lr`, written by `_training_common.py`); plotted by `AML_Analysis.ipynb` (§F) via `evaluation.figures.plot_training_curves`.
- **Paper deliverables:** `runs/paper_figures/` (created by `AML_Analysis.ipynb`) contains `tables/master_pck.{tex,md,csv}`, `tables/per_category_a*.csv`, `tables/per_difficulty_a*.csv`, `tables/efficiency.{csv,tex}`, and `figures/{wsa_gain,ft_depth,lora_vs_ft,per_category,per_difficulty,training_curves,wsa_window_sensitivity_a0.1,dino_layer_sensitivity_a0.1}.{pdf,png}` plus qualitative overlays under `qualitative/`. Every figure ships in both PDF (vector, dpi=300) and PNG (raster, dpi=300); matching colormap policy is **viridis** everywhere, never `jet`. Every table has a `.tex` form with column-wise best entries bolded.

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
| `FT_PATIENCE` / `LORA_PATIENCE` | 7 | Early stopping patience (epochs of stagnant val loss before stop) |
| `FT_MIN_DELTA` / `LORA_MIN_DELTA` | `0.0` | Early-stopping tolerance on val loss. Set > 0 to ignore tiny improvements (e.g. `1e-3`) |
| `LORA_RANK` | 8 | LoRA rank |
| `PREPROCESS` | `FIXED_RESIZE` | Image preprocessing mode |
| `IMAGE_SIZE_BY_BACKBONE` | `{dinov2_vitb14: (518,518), dinov3_vitb16: (512,512), sam_vit_b: (512,512)}` | Per-backbone input size (exact patch-size multiples: 518=37×14, 512=32×16; SAM features always extracted at 1024×1024 internally) |
| `IMAGE_HEIGHT` / `IMAGE_WIDTH` | 784 / 784 | Global fallback size for backbones not listed in `IMAGE_SIZE_BY_BACKBONE` |
| `EVAL_SPLIT` | `test` | Evaluation split |
| `EVAL_LIMIT` | 0 (full split) | Pair limit for debugging |
| `LOG_BATCH_INTERVAL` | 50 | Batch logging frequency (pipeline default; `scripts/train.py --log-batch-interval` standalone default is **100**) |
| `RESUME_SAVE_INTERVAL` | 50 | Mid-epoch resume checkpoint cadence |
| `DINO_LAYER_INDICES` | `4` | Intermediate ViT layer for DINO feature extraction (passed as `--layer-indices`; ignored for SAM) |
| `WSA_WINDOW_LIST` | `[3, 5, 7, 9]` | Window sizes scanned by the inference-only WSA-window ablation (baseline only); produces `pck_results_wsa_sweep.json` |
| `EVAL_LAYER_INDICES_LIST` | `[4, 9, 11]` | DINO intermediate layer indices scanned by the inference-only layer ablation (baseline only); produces `pck_results_layer_sweep.json` |

**Workflow toggles:** In addition to baseline and checkpoint eval toggles, `RUN_EVAL_FINETUNED_WSA` and `RUN_EVAL_LORA_WSA` enable WSA evaluation on trained checkpoints (PDF Stage 3 combined with Stages 2 and 4). `RUN_EVAL_WSA_SWEEP` and `RUN_EVAL_LAYER_SWEEP` enable the two inference-only ablations on the baseline (no extra training, costs only one extra eval pass per backbone × hyperparameter setting).

**Resume:** `PIPELINE_RESUME` uses `runs/pipeline_state.json`. **`fingerprint_from_config`** hashes `_fingerprint_payload()` (all training hyperparameters, block lists, eval flags, etc.). A mismatch **clears** completed steps so incompatible runs are not mixed.

**YAML `--config`:** `_apply_pipeline_yaml` maps sections into module globals. Fine-tuning and LoRA have **separate** scalar batch size fields (`finetune.batch_size` → `FT_BATCH_SIZE`, `lora.batch_size` → `LORA_BATCH_SIZE`) plus optional per-backbone maps (`finetune.batch_size_by_backbone`, `lora.batch_size_by_backbone`). Precision is configured through `runtime.precision` → `PRECISION`. The `finetune.last_blocks` field accepts either a single integer or a list.

**Compile:** `COMPILE=True` is the in-script default in `run_pipeline.py`; `scripts/_training_common.maybe_compile_model` activates `torch.compile(mode="reduce-overhead")` only on CUDA Ampere+ and silently no-ops everywhere else (MPS, CPU, older CUDA). The wrapper forwards to the original module so state-dict / save / resume keys stay clean.

**Notebook overrides:** both `AML_Colab.ipynb` and `AML_Local.ipynb` write `config.yaml` with PDF batch defaults (DINO=20, SAM=4 on CUDA), `precision` selected from compute capability (Ampere+ → bf16, Volta/Turing → fp16, MPS/CPU → fp32), and `resume_save_interval` adapted to the speed tier (`slow=50`, `medium=200`, `fast=500`). Hyperparameters (`epochs=50`, `patience=7`, LR/weight-decay/LoRA rank/alpha) are project-fixed and not modified by the notebooks.

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

## 10. Notebooks and config file

The repository ships **three** notebooks at the root, each with a single, well-defined responsibility:

- **`AML_Local.ipynb` (training & orchestration, local):** `pip install -e ".[notebook]"`, dataset/weights if missing, capability-adaptive runtime knobs (precision, compile, workers, resume cadence), `config.yaml`, pipeline subprocess with a live `stage_events.jsonl` dashboard and tail of `runs/logs/current.log`, plus a small post-run sanity table. **No analysis cells** — point users at `AML_Analysis.ipynb`.
- **`AML_Colab.ipynb` (training & orchestration, Colab):** clones the repo, mounts Drive, sets `runs/` and `checkpoints/` symlinks under `MyDrive/Colab Notebooks/AML_results/`, downloads SPair-71k + weights, writes `config.yaml`, runs the pipeline with the same dashboard. **No analysis cells.** End-of-run message instructs the user to download the artifacts and run `AML_Analysis.ipynb` locally.
- **`AML_Analysis.ipynb` (analysis only, runs locally on CPU/MPS):** sections **A–I** drive the deliverable pipeline. The notebook is **read-only with respect to `runs/`**: it consumes the pipeline exports listed in §7 and never re-runs training or eval.
  - **A** Setup & inventory of `runs/pipeline_exports/`, `checkpoints/`, `runs/logs/`, `data/SPair-71k/`.
  - **B** Master PCK table → `runs/paper_figures/tables/master_pck.{tex,md,csv}` (column-wise best bolded in LaTeX).
  - **C** Δ-PCK figures: `wsa_gain_a*.pdf/png` (C.1), `ft_depth.pdf/png` (C.2, PDF Stage 2 question), `lora_vs_ft.pdf/png` (C.3, param-efficiency scatter, x = log-trainable params), and the inference-only **C.4 ablations**: WSA window vs PCK from `pck_results_wsa_sweep.json` → `figures/wsa_window_sensitivity_a0.1.{pdf,png}`; DINO intermediate layer index vs PCK from `pck_results_layer_sweep.json` → `figures/dino_layer_sensitivity_a0.1.{pdf,png}`.
  - **D** Per-category heatmap (run × SPair category) plus top-5 / bottom-5 mean-PCK bar chart.
  - **E** Per-difficulty (viewpoint / scale / truncation / occlusion) low-vs-high bars and CSV.
  - **F** Training curves with annotated best-epoch from each `*_history.jsonl` (also plots LR if present).
  - **G** Qualitative paper-grade overlays: method-comparison grids (rows = method, cols = source/target with predictions vs GT), similarity-heatmap overlays per keypoint (colormap **viridis**), and a failure-case browser ranked by per-image PCK from `pck_results.json`.
  - **H** Efficiency table from saved checkpoints: trainable parameter count per (backbone, method) plus best-epoch / wall-clock estimate.
  - **I** Tree print of `runs/paper_figures/` so the user can copy-paste the artifacts into a LaTeX paper.
- **Adaptation policy (training notebooks):** classify the device by compute capability and VRAM only (no GPU model name is hard-coded). Set `precision`, `compile`, `num_workers`, and `resume_save_interval` from the resulting tier. Hyperparameters (batch sizes, LR, weight decay, epochs, patience, LoRA rank/alpha) are project-fixed; the only deviation is the MPS/CPU batch derogation in `AML_Local.ipynb`.
- **`config.yaml`:** notebook-generated pipeline configuration. It contains machine-specific absolute paths, so it is **not tracked in git** (see `.gitignore`). The training notebooks write it for `scripts/run_pipeline.py --config config.yaml`.
- **YAML reader:** the pipeline applies keys via `_apply_pipeline_yaml` in `run_pipeline.py` (``dataset``, ``workflow_toggles``, etc.). Key runtime overrides include `runtime.dino_layer_indices` → `DINO_LAYER_INDICES` (int, default 4): intermediate ViT layer for DINO feature extraction.
- **SPair-71k download URL (Colab):** `https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz`
- **Pipeline reset:** the config cell of each training notebook sets `START_FROM_SCRATCH` automatically (Colab inspects Drive, Local inspects the repo) and the run-pipeline cell launches `scripts/run_pipeline.py` via `subprocess.Popen`, exporting **`SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1`** when the flag is true (clears `runs/pipeline_state.json` only; does not delete checkpoints).
- **Analysis modules:** `AML_Analysis.ipynb` is a thin orchestrator on top of two reusable Python modules — `evaluation/figures.py` (load/aggregate/format pipeline exports, plotting primitives) and `evaluation/qualitative.py` (load checkpoints, render method grids and similarity heatmaps, identify failure cases / symmetry-ambiguity). Both modules have unit tests in `tests/test_figures.py` and `tests/test_qualitative.py` that cover parsing/aggregation logic without invoking matplotlib or torch inference.
- **Workflow Colab → local analysis:** after `AML_Colab.ipynb` finishes, copy the contents of `MyDrive/Colab Notebooks/AML_results/{runs,checkpoints}/` into your local repo (preserving the layout) and open `AML_Analysis.ipynb`. No GPU is required: qualitative inference runs on CPU/MPS.

---

## 11. Coding standards

- **Code language:** English for identifiers, docstrings, and comments (see `data/dataset.py` for examples).
- **Docs:** `documentation.md` (this file) = system reference; `README.md` = quick start.
- **Lint:** `ruff` in `pyproject.toml` (line length 100, Python 3.9 syntax target).
- **Comment / docstring policy:**
  - *First-party code* (`data/`, `training/`, `evaluation/`, `models/common/`, `scripts/`, `utils/`, `tests/`): English only; short module summaries; NumPy-style blocks on public APIs where helpful; sparse `#` comments only for non-obvious logic. Prefer ASCII (hyphens, `x` for dimensions, `->` for flow) in comments and docstrings.
  - *Vendored backbones* (`models/dinov2/`, `models/dinov3/`, `models/sam/`) and *third-party code* (`third_party/sd4match/`): keep upstream style; do not mass-edit for house consistency.
- **Check:** `python scripts/audit_first_party_comments.py` (exit 0 if no non-ASCII lines in first-party trees).

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

Coverage includes dataset workers, pipeline state, smoke imports, backbone construction, SD4Match interface, window soft-argmax, cosine matching / argmax correspondence, LoRA adapters, PCK helpers, preprocessing/keypoint scaling, ViT unfreezing, and training config dataclasses. Tests do **not** guarantee numerical results on full SPair training. After each run, `tests/conftest.py` generates a per-file Markdown table of results, durations, and failure details (local only, not tracked in git).

---

## 14. Design rationale

Implementation choices (backbone selection, loss design, WSA, LoRA, photometric augmentation, PCK thresholds) follow the dense correspondence literature cited in the project report.

---

## 15. Package version

**`semantic-correspondence`:** version **`0.1.0`** in `pyproject.toml`. This document’s accuracy is tied to the git tree at the time of the last edit to **`documentation.md`**.
