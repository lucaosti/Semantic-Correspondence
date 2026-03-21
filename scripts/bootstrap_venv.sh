#!/usr/bin/env bash
# Create `.venv`, upgrade pip, install project dependencies, and editable install.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT}/requirements.txt"
python -m pip install -e "${ROOT}"

echo ""
echo "Done. Activate this environment with:"
echo "  source ${ROOT}/.venv/bin/activate"
echo ""
echo "Run Python from the repo root so imports resolve (editable install handles packages)."
echo "Next steps:"
echo "  - Verify SPair-71k: python scripts/verify_dataset.py"
echo "  - Backbones: models/dinov2, models/dinov3, models/sam (see README.md)."
echo "  - Optional: pip install -e '.[dev]' (pytest) or pip install -e '.[notebook]' (Jupyter)."
