#!/usr/bin/env bash
# Run the full pipeline under nohup. All verbose output goes to runs/logs/pipeline_*.log
# (see current.log). Stdout of the driver process is only rare messages → driver_nohup.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
mkdir -p runs/logs

PYTHON="${PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
fi

OUT="${ROOT}/runs/logs/driver_nohup.txt"
: >"${OUT}"
nohup env SEMANTIC_CORRESPONDENCE_PIPELINE_LOG_FILE_ONLY=1 \
  "${PYTHON}" "${ROOT}/scripts/run_pipeline.py" >>"${OUT}" 2>&1 &
echo $! >"${ROOT}/runs/pipeline.pid"

PID="$(cat "${ROOT}/runs/pipeline.pid")"
echo ""
echo "Pipeline PID ${PID}  |  log → runs/logs/current.log  |  stop: bash scripts/kill_pipeline.sh"
echo "Watch: bash scripts/reconnect_dashboard.sh   or   tail -f runs/logs/current.log"
echo "Full reset (stages): SEMANTIC_CORRESPONDENCE_PIPELINE_RESET=1 bash scripts/start_pipeline_detached.sh"
echo "Wipe logs+state:     bash scripts/kill_pipeline.sh --clean"
echo ""
