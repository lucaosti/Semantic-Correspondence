This directory vendors a minimal subset of the `ActiveVisionLab/SD4Match` codebase
needed for dataset loading and evaluation metrics.

Upstream project:
- Repository: `https://github.com/ActiveVisionLab/SD4Match`

License note:
- At the time this subset was vendored, the upstream repository root did **not** expose a `LICENSE`
  file via the GitHub contents API. If you redistribute this project, you should double-check the
  upstream licensing status and comply accordingly.

Vendored modules (kept as close as possible to upstream behavior):
- `dataset/` (SPair-71k pair dataset)
- `utils/` (geometry, matching, evaluator)

If you update these files, prefer copying from upstream verbatim and keeping changes
minimal and explicit, so metric behavior remains identical.

