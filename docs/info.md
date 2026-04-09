# Project rules (canonical)

This file is the canonical checklist for implementation and experiments. The root [`README.md`](../README.md) is the practical setup guide; constraints below must match the codebase.

## Data splits (SPair-71k)

- **`train`** (on-disk list `trn.txt`): training only.
- **`val`**: validation, hyperparameter tuning, and early stopping.
- **`test`**: **final evaluation only** — do not use for model selection or tuning.

Scripts that report benchmark numbers should use `--split test` unless you intentionally want a quick check on `val`.

## Backbone weights

- Prefer **official** weight sources (Meta URLs / upstream project URLs) or local paths.
- **Hugging Face mirrors are allowed only as a fallback** when official URLs fail (some upstream hosts can intermittently 401/403).
- When using any mirror, downloads must be protected by **SHA256 verification** when a known hash is available (see [`scripts/download_pretrained_weights.py`](../scripts/download_pretrained_weights.py)).

## Window soft-argmax (WSA)

- Use **only at inference / evaluation** when refining dense matches (optional post-processing).
- **Do not** use WSA inside the training loss or fine-tuning objective. See [`models/common/window_soft_argmax.py`](../models/common/window_soft_argmax.py).

## Training

- The Gaussian correspondence loss **averages over pairs in a batch**. Training scripts default to **`--batch-size 20`** for DINO backbones; SAM uses 4 (set via `FT_BATCH_SIZE_BY_BACKBONE`/`LORA_BATCH_SIZE_BY_BACKBONE` in the pipeline). Reduce if OOM; larger batches improve GPU utilization when VRAM allows.

## Language

- Code symbols, docstrings, and comments are in **English** by project convention. This file is in English so it stays aligned with the codebase.
