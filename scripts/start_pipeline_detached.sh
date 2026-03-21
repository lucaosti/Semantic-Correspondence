#!/usr/bin/env bash
# Run the full pipeline under nohup so it survives terminal closure. PID → runs/pipeline.pid.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
mkdir -p runs/logs

PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

OUT="${ROOT}/runs/logs/nohup_console.txt"
: >"${OUT}"
nohup "${PYTHON}" "${ROOT}/scripts/run_pipeline.py" >>"${OUT}" 2>&1 &
echo $! >"${ROOT}/runs/pipeline.pid"

echo "Pipeline started detached (PID $(cat "${ROOT}/runs/pipeline.pid"))."
echo "  Console copy: ${OUT}"
echo "  Structured log: runs/logs/current.log → latest pipeline_*.log (for dashboard / tail -f)"
echo "Reconnect dashboard anytime: bash scripts/start_dashboard.sh"
