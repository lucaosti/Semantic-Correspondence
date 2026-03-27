# Semantic correspondence on SPair-71k

This repository is a PyTorch codebase for **semantic correspondence**: given two images of the same object category, the goal is to predict where keypoints in one image land in the other. The project follows the SPair-71k benchmark and explores **training-free** dense matching (cosine similarity + argmax), optional **window soft-argmax** at inference only, **full fine-tuning** of the last transformer blocks, and **LoRA** on late MLP layers. Three **ViT-B** backbones are supported end to end: **DINOv2**, **DINOv3**, and **SAM** (image encoder only). Their architectures live under `models/dinov2/`, `models/dinov3/`, and `models/sam/`; you still obtain official **weights** separately and pass paths on the command line where documented.

If you are setting up from scratch, start from the repository root. The bootstrap script creates a `.venv`, installs dependencies from `requirements.txt`, and runs an editable install (`pip install -e .`) so packages such as `data`, `models`, `training`, and `evaluation` import cleanly without tweaking `PYTHONPATH`. Activate the venv, then point the code at SPair-71k—typically by unpacking it under `data/SPair-71k/` or by setting `SPAIR_ROOT`. Running `python scripts/verify_dataset.py` once is the quickest way to confirm paths, split lists, and that the PyTorch dataset can load a sample from each split.

**Image size:** training and evaluation scripts default to **784×784** `FIXED_RESIZE` so spatial sizes are divisible by both patch **14** (DINOv2 ViT-B/14) and **16** (DINOv3 ViT-B/16). SAM still applies its own 1024×1024 path inside the dense extractor when selected.

**Training length & progress:** the orchestrated driver (`scripts/run_pipeline.py`) defaults to **200** epochs each for fine-tuning and LoRA (all backbones), with early stopping via `FT_PATIENCE` / `LORA_PATIENCE` — edit those constants to change schedule. CLI examples below may use shorter runs. Logs: `epoch k/total`, batch indices when `--log-batch-interval` > 0, and `tail -f runs/logs/current.log`.

**Hardware:** training defaults to **batch size 100** pairs per step (Gaussian loss averaged over the batch). Reduce `--batch-size` if you hit OOM. Training/eval pick **CUDA → Apple MPS → CPU** automatically when `--device` is omitted (or set `DEVICE = None` in `run_pipeline.py`). DataLoader workers default to **auto**: **CUDA** prefetches with up to ~¾ of logical CPUs (cap 64); **CPU** training uses about **n/4** workers (cap 16) so the main process keeps cores for ViT forward/backward, with **OMP_NUM_THREADS=1** inside workers to reduce oversubscription. With CUDA, **cuDNN benchmark** and **TF32** are enabled. Pinned memory on CUDA only.

**NVIDIA/CUDA checklist (recommended before long runs):**

1. Confirm the GPU is visible to the system:
   ```bash
   nvidia-smi
   ```
2. Confirm your installed PyTorch build can use CUDA:
   ```bash
   python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('cuda_version=', torch.version.cuda); print('device0=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
   ```
3. If CUDA is unavailable, reinstall `torch`/`torchvision` from the [official PyTorch installer](https://pytorch.org/get-started/locally/) for your driver + CUDA stack.

**Multi-GPU note:** this project uses CUDA's default visible device index. To force one GPU, set `CUDA_VISIBLE_DEVICES` before running scripts (for example, `CUDA_VISIBLE_DEVICES=1 ...`). After masking, the selected GPU becomes logical `cuda:0` for the process.

**Forcing CUDA in the orchestrated pipeline:** `scripts/run_pipeline.py` does not expose a `--device` CLI flag; set it in YAML with `runtime.device: "cuda"` and run with `--config`.

**Pascal GPUs (e.g. GTX 1080 Ti, ~11 GB VRAM, compute capability 6.1):** default batch 100 may OOM on SAM (1024×1024 internally); lower `--batch-size` or use DINO backbones first. Close other GPU processes if you hit OOM. Install a PyTorch build whose CUDA binaries still include **sm_61** (check with `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"`). TF32 and Tensor Core–style speedups target newer architectures; on Pascal, expect limited benefit from those toggles versus raw FP32.

**Many-core host CPUs (e.g. AMD Ryzen 7 1800X, 8 cores / 16 threads):** auto mode may choose a **large** `num_workers` (prefetch). If the CPU is saturated or the system feels sluggish, set **`NUM_WORKERS`** explicitly in `run_pipeline.py`, your YAML `--config`, or `num_workers` in the notebook config—**6–8** is a reasonable starting point. You can raise **`SEMANTIC_CORRESPONDENCE_PREFETCH_CAP`** (see `utils/hardware.py`) if you increase workers and want a deeper prefetch queue.

**Terminal dashboard:** with `pip install -e ".[dashboard]"`, run `bash scripts/start_dashboard.sh` (or `python scripts/pipeline_dashboard.py`) while training runs; it tails `runs/logs/current.log` when present (symlink to the active `pipeline_*.log`), else the newest timestamped log, parses epoch/loss/batch lines, and shows `manifest.tsv`.

**Detached pipeline (close terminal safely):** `bash scripts/start_pipeline_detached.sh` runs the driver under `nohup` (PID in `runs/pipeline.pid`; verbose output goes to `runs/logs/current.log`, not the terminal). Stop with `bash scripts/kill_pipeline.sh`. **Reconnect later:** `bash scripts/reconnect_dashboard.sh` or `tail -f runs/logs/current.log`.

From there you can work in two styles. **Ad hoc**: call `scripts/eval_baseline.py` for PCK, `scripts/train_finetune.py` for Task-style fine-tuning, or `scripts/train_lora.py` for parameter-efficient training, each with `--backbone` and the right weight flags. **Orchestrated**: `scripts/run_pipeline.py` is the single entry point. By default it runs the **full stack** (dataset check, fine-tune and LoRA for all three backbones, all PCK evaluation modes including window soft-argmax and trained checkpoints, metric export, `pytest`, and a Jupyter hint). Turn steps off in the configuration block if you need a shorter run. **SAM** weights are not in git: run `bash scripts/download_sam_vit_b.sh` once to save `checkpoints/sam_vit_b_01ec64.pth` (Meta’s official URL); the pipeline picks up that path automatically, or you can set `SAM_CHECKPOINT` / `export SAM_CHECKPOINT=...`. For tables and plots in Jupyter, install the notebook extra (`pip install -e ".[notebook]"`) and use `notebooks/verify_and_compare_results.ipynb`; it shares the same evaluation path as the CLI.

Project rules—splits (`train` / `val` / `test`), **prefer official weights (HF mirrors allowed only as fallback)**, window soft-argmax only at inference—are spelled out in **`docs/info.md`**. Italian working notes and layout expectations are in **`docs/claude.md`**. A literature-oriented overview lives in **`docs/stato-arte.md`**. **Full technical reference:** **`documentation.md`** (keep it updated when changing behavior or defaults). Those files are the canonical place for constraints; the root README stays a practical tour of the repo.

### External notebook workflow

If you want a notebook outside this repository to control training, evaluation, and plots, use the notebook helpers in **`utils/notebook_workflow.py`** and keep the tunables in a single YAML file. A complete template lives at **`notebooks/notebook_config.example.yaml`**.

Recommended setup:

1. Install the project and notebook extras: `pip install -e ".[notebook]"`.
2. Copy `notebooks/notebook_config.example.yaml` next to your external notebook and edit the paths, training jobs, and experiment list there.
3. In the notebook, load the config with `from utils.notebook_workflow import load_workflow_config`.
4. Run training with `run_finetune_job(cfg)` and `run_lora_job(cfg)`; both functions call the existing CLI scripts and write logs under `runs/notebook_exports/logs/`.
5. Run evaluation with `run_evaluation_suite(cfg)`, then export and plot with `save_results(...)` and `plot_results(...)`.
6. Use `show_dataset_sample(cfg, split="val", index=0)` for a quick visual check of preprocessing and data loading.

The YAML is split into:

- `paths`: repo root, SPair-71k root, checkpoint directory, output directory.
- `runtime`: device, worker count, preprocessing mode, image size, PCK thresholds, WSA settings.
- `finetune`: last blocks, epochs, patience, learning rate, weight decay, resume path, backbone weights.
- `lora`: LoRA rank/alpha plus the same backbone and runtime controls.
- `experiments`: the evaluation runs you want to compare, including baseline, WSA, fine-tuned checkpoints, and LoRA checkpoints.

That keeps the notebook thin: the repository code handles path resolution, command construction, result export, and plotting.

---

### Google Colab (end-to-end)

If you want to run this project inside **Google Colab** (instead of a local machine), use **`AML_Colab.ipynb`** at the repository root. It is self-contained and will:

- clone the repository to `/content/Semantic-Correspondence`
- download + extract **SPair-71k** to `data/SPair-71k/`
- install the project (`pip install -e ".[notebook]"`)
- download pretrained weights into `checkpoints/`
- write `config.yaml` and run `scripts/run_pipeline.py --config config.yaml`

For a local Linux + NVIDIA Jupyter workflow (repo + dataset already present), keep using **`AML.ipynb`**.

---

### Commands (quick reference)

Environment and checks:

```bash
bash scripts/bootstrap_venv.sh
source .venv/bin/activate
python scripts/verify_dataset.py
```

Baseline PCK (use `--split test` for reported numbers; `val` for quick checks):

```bash
python scripts/eval_baseline.py --backbone dinov2_vitb14 --split test --device cuda
python scripts/eval_baseline.py --backbone dinov3_vitb16 --dinov3-weights /path/to/weights.pth --split test --device cuda
python scripts/eval_baseline.py --backbone sam_vit_b --sam-checkpoint /path/to/sam_vit_b_01ec64.pth --split test --device cuda
python scripts/eval_baseline.py --backbone dinov2_vitb14 --split test --window-soft-argmax --wsa-window 5 --device cuda
```

Fine-tuned or LoRA checkpoints:

```bash
python scripts/eval_baseline.py --backbone dinov2_vitb14 --checkpoint checkpoints/dinov2_vitb14_lastblocks2_best.pt --split test --device cuda
```

Training:

```bash
python scripts/train_finetune.py --backbone dinov2_vitb14 --epochs 50 --patience 5 --device cuda
python scripts/train_lora.py --backbone dinov2_vitb14 --epochs 2 --device cuda
```

Optional driver and tests:

```bash
# example config (config.yaml):
# runtime:
#   device: "cuda"
#   num_workers: -1
python scripts/run_pipeline.py --config config.yaml
pip install -e ".[dev]"
pytest -q
```

Detached pipeline and dashboard (close the terminal anytime; reconnect with `bash scripts/reconnect_dashboard.sh` or `bash scripts/start_dashboard.sh`):

```bash
bash scripts/kill_pipeline.sh
bash scripts/start_pipeline_detached.sh
# later, in a new terminal:
bash scripts/reconnect_dashboard.sh
# or: bash scripts/start_pipeline_and_dashboard.sh   # kill → detached pipeline → dashboard (foreground)
# Full restart (clear resume bookkeeping; keeps weight checkpoints): SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1 bash scripts/start_pipeline_detached.sh
```

**Resume:** `runs/pipeline_state.json` records finished stages; `runs/logs/stage_events.jsonl` logs `start` / `done` / `skip` / `fail` per stage. Training uses `checkpoints/*_resume.pt` for mid-training resume when the pipeline is interrupted; to avoid losing too much progress on preemptible runtimes (e.g. Colab), use a smaller `--resume-save-interval` (recommend **100**) and ensure `checkpoints/` is persisted (Drive symlink in `AML_Colab.ipynb`).

If you need a specific CUDA build of PyTorch, reinstall `torch` / `torchvision` after bootstrap using the [official installer](https://pytorch.org/get-started/locally/); see also comments in `requirements.txt`.

---

### Where things live

| Path | Role |
|------|------|
| `documentation.md` | Full technical reference (defaults, theory, maintenance; update when code changes) |
| `data/dataset.py` | SPair-71k loading, preprocessing, splits |
| `models/common/` | Matching, dense extractors, LoRA, window soft-argmax |
| `models/dinov2`, `dinov3`, `sam` | Backbone code and adapters (see each `hub_loader.py` / `backbone.py`; **`documentation.md`** for full map) |
| `training/` | Losses, training loop helpers, unfreeze utilities |
| `evaluation/` | PCK, baseline evaluation, `experiment_runner` (shared with scripts and notebook) |
| `scripts/` | CLI tools and `run_pipeline.py` |
| `docs/` | Guidelines (`info.md`), Italian notes (`claude.md`), literature (`stato-arte.md`) |
| `notebooks/` | PCK comparison notebook and notebook generators |
| `AML.ipynb` | Local Linux + NVIDIA Jupyter notebook (generated) |
| `AML_Colab.ipynb` | Google Colab end-to-end notebook (generated) |

Large weights and run artifacts are not meant for git: use ignored paths such as `checkpoints/` and `runs/` (see `.gitignore`). Each `python scripts/run_pipeline.py` run writes a **timestamped log** under `runs/logs/pipeline_<UTC_datetime>.log` (every line prefixed with UTC date/time) and appends a row to `runs/logs/manifest.tsv` (start/end events and exit code).

Pipeline metric exports are written under `runs/pipeline_exports/`: aggregate tables (`pck_results.json`, `pck_results.csv`) plus SD4Match-style granular outputs (`pck_results_per_image.json`, `pck_results_per_point.json`) and a flag-wise difficulty breakdown (`pck_results_by_difficulty_flag.json`) when the SD4Match metrics backend is enabled.
