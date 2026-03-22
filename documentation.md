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

Weights are **not** committed to git. Use `scripts/download_pretrained_weights.py`, `scripts/download_sam_vit_b.sh`, and/or CLI/path flags as described in `README.md`.

---

## 2. Repository layout

| Path | Responsibility |
|------|----------------|
| `data/` | `SPair71kPairDataset`, `spair_collate_fn`, preprocessing, `data/paths.resolve_spair_root` |
| `models/common/` | Matching, coords, LoRA, window soft-argmax, `vit_intermediate`, input norm |
| `models/dinov2/`, `dinov3/`, `sam/` | Backbone implementations; weight loading in `hub_loader.py` / `backbone.py` |
| `training/` | `losses.py`, `engine.py` (batched Gaussian losses), `unfreeze.py`, `early_stopping.py`, `config.py` |
| `evaluation/` | `pck.py`, `baseline_eval.py`, `experiment_runner.py`, checkpoint helpers |
| `utils/` | `hardware.py`, `pipeline_state.py`, `notebook_workflow.py`, `paths.py` |
| `scripts/` | CLIs and `run_pipeline.py` (see §8) |
| `docs/` | `info.md` (rules), `claude.md` (Italian notes), `stato-arte.md` (references). Only `docs/**/*.pdf` is gitignored under `docs/`; `*.md` files are tracked. |
| `notebooks/` | `verify_and_compare_results.ipynb`, `notebook_config.example.yaml`, `_generate_aml_local.py`, root `AML.ipynb` (generated) |
| `runs/`, `checkpoints/` | Gitignored artifacts: logs, exports, downloaded weights, training checkpoints |

**Root:** `README.md`, `requirements.txt`, `pyproject.toml`, `documentation.md`.

---

## 3. Environment

- **Python:** `>= 3.9` (`pyproject.toml`).
- **Install:** `bash scripts/bootstrap_venv.sh` then `source .venv/bin/activate`.
- **Extras:** `pip install -e ".[dev]"` (pytest, ruff), `".[notebook]"`, `".[dashboard]"`.
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

Do **not** use Hugging Face Hub checkpoints for these three backbones in this project. Use official Meta URLs or local `.pth` files passed via CLI/YAML (`docs/info.md`).

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

### 6.3 Batch size defaults

| Location | Default |
|----------|---------|
| `train_finetune.py` / `train_lora.py` | `--batch-size` **100** |
| `run_pipeline.py` | `TRAIN_BATCH_SIZE` **100** |
| `training/config.py` dataclasses | **100** |
| `utils/notebook_workflow.py` / `notebooks/notebook_config.example.yaml` | **100** |

Reduce **batch size** if you hit **OOM** (especially **SAM** at 1024×1024). **Evaluation** (`baseline_eval.py`, `eval_baseline.py`, `experiment_runner`) uses **batch size 1** for the PCK loop (`evaluate_spair_loader` contract).

### 6.4 Fine-tuning (last blocks)

`unfreeze_last_transformer_blocks(model, n_blocks=N)` (`training/unfreeze.py`): after `freeze_all`, only the last **N** modules in `model.blocks` are trainable. Optimizer sees only `requires_grad=True` parameters.

### 6.5 LoRA

Injected on late MLP linears (`models/common/lora.py` + backbone-specific hooks). Only adapter weights are updated.

### 6.6 Early stopping and resume

- **Early stopping:** `training/early_stopping.py` on **validation loss** each epoch (`patience` configurable).
- **Resume files:** `checkpoints/<backbone>_lastblocks*_resume.pt` / LoRA equivalents store model, optimizer, epoch, stopper state, and optional `batch_in_epoch` for mid-epoch resume.

---

## 7. Evaluation: PCK

**PCK@α** (`evaluation/pck.py`): a keypoint is correct if

`‖pred − gt‖₂ ≤ α · pck_threshold`,

where **`pck_threshold`** is the **bounding-box scale** (max side length) in the **same pixel frame** as predictions and ground truth (bbox-normalized PCK). Invalid padded keypoints are masked with `invalid_value` (default `-2.0`).

Default **α** triple in the pipeline: **`(0.05, 0.1, 0.15)`** (`EVAL_ALPHAS` in `run_pipeline.py`). **`EVAL_LIMIT > 0`** truncates pairs for debugging.

`experiment_runner.py` schedules multiple **eval specs** (baseline, WSA, checkpoints); each run uses the shared baseline evaluator with **batch size 1** data loading unless changed in code.

---

## 8. Orchestrated pipeline (`scripts/run_pipeline.py`)

### 8.1 Stage order

1. Optional **`verify_dataset`**
2. **Fine-tune** for each enabled backbone (tuple flags)
3. **LoRA** for each enabled backbone
4. **PCK** matrix: baseline, WSA, fine-tuned checkpoint, LoRA checkpoint (per flags) + optional **`runs/pipeline_exports/`** JSON/CSV
5. Optional **`pytest`**
6. Optional Jupyter hint

**Tuple order is always** `(dinov2_vitb14, dinov3_vitb16, sam_vit_b)`.

### 8.2 Default configuration (in-script; YAML overrides)

| Symbol | Typical default |
|--------|-------------------|
| `TRAIN_BATCH_SIZE` | 100 |
| `FT_EPOCHS` / `LORA_EPOCHS` | 200 |
| `FT_PATIENCE` / `LORA_PATIENCE` | 10 |
| `LAST_BLOCKS` | 2 |
| `LORA_RANK` | 8 |
| `PREPROCESS` | `FIXED_RESIZE` |
| `IMAGE_HEIGHT` / `IMAGE_WIDTH` | 784 / 784 |
| `EVAL_SPLIT` | `test` |
| `EVAL_LIMIT` | 0 (full split) |
| `LOG_BATCH_INTERVAL` | 100 |

**Resume:** `PIPELINE_RESUME` uses `runs/pipeline_state.json`. **`fingerprint_from_config`** hashes `_fingerprint_payload()` (includes `TRAIN_BATCH_SIZE`, epochs, weights paths, eval flags, etc.). A mismatch **clears** completed steps so incompatible runs are not mixed.

**YAML `--config`:** `_apply_pipeline_yaml` maps sections into module globals. If both **`finetune.batch_size`** and **`lora.batch_size`** are present, the **lora** block is applied **after** finetune, so **`lora.batch_size` wins** for `TRAIN_BATCH_SIZE`.

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
- **`num_workers`:** CLI `-1` means auto via `recommended_dataloader_workers` (CUDA: aggressive prefetch; CPU training: smaller pool).
- **CUDA:** `cudnn.benchmark`, TF32 allow where supported, `pin_memory`, `persistent_workers` + `prefetch_factor` when workers > 0.

**MPS:** Some ops (e.g. `grid_sample` with certain `padding_mode`) may fail; use **CPU** or **CUDA** for parity with Linux training/eval.

---

## 10. Notebooks and config files

- **`utils/notebook_workflow.py`:** builds subprocess commands to `train_finetune.py` / `train_lora.py` / eval scripts from YAML.
- **`notebooks/notebook_config.example.yaml`:** full template.
- **`AML.ipynb`:** generated by `notebooks/_generate_aml_local.py`; writes **`config.yaml`** at repo root for `run_pipeline.py --config`.
- **Notebook env:** when `START_FROM_SCRATCH` is true, the generated notebook sets `AML_START_FROM_SCRATCH=1`; the pipeline cell maps that to **`SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1`** for that subprocess (pipeline resume bookkeeping, not deletion of weight checkpoints).

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

Coverage includes dataset workers, pipeline state, smoke imports, backbone construction, notebook command wiring. Tests do **not** guarantee numerical results on full SPair training.

---

## 14. Maintenance contract

Update **`documentation.md`** when any of the following change: default CLI/pipeline constants, loss or engine semantics, eval metrics, resume/fingerprint fields, supported backbones, weight acquisition paths, or documented environment variables. Align **`README.md`** and **`docs/info.md`** when user-visible rules or defaults shift.

---

## 15. Package version

**`semantic-correspondence`:** version **`0.1.0`** in `pyproject.toml`. This document’s accuracy is tied to the git tree at the time of the last edit to **`documentation.md`**.
