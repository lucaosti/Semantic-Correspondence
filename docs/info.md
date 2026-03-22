# Project rules (canonical)

This file is the canonical checklist for implementation and experiments. The root [`README.md`](../README.md) is the practical setup guide; constraints below must match the codebase.

## Data splits (SPair-71k)

- **`train`** (on-disk list `trn.txt`): training only.
- **`val`**: validation, hyperparameter tuning, and early stopping.
- **`test`**: **final evaluation only** — do not use for model selection or tuning.

Scripts that report benchmark numbers should use `--split test` unless you intentionally want a quick check on `val`.

## Backbone weights

- **Do not** use Hugging Face Hub checkpoints for **DINOv2**, **DINOv3**, or **SAM** image encoders in this project.
- Load official weights via Meta URLs or local paths, as documented in each backbone module and in [`README.md`](../README.md).

## Window soft-argmax (WSA)

- Use **only at inference / evaluation** when refining dense matches (optional post-processing).
- **Do not** use WSA inside the training loss or fine-tuning objective. See [`models/common/window_soft_argmax.py`](../models/common/window_soft_argmax.py).

## Training

- The Gaussian correspondence loss is written for **batch size 1** (one image pair per step). Training scripts default to `--batch-size 1`; keep it unless the loss and collate are extended for larger batches.

## Language

- Code symbols, docstrings, and comments are in **English** by project convention. This file is in English so it stays aligned with the codebase.
