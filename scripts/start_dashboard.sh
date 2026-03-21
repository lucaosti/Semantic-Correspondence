#!/usr/bin/env bash
# Terminal dashboard; follows runs/logs/current.log when present (same run after reopening a terminal).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

exec "${PYTHON}" "${ROOT}/scripts/pipeline_dashboard.py" "$@"
