#!/usr/bin/env bash
# Stop detached pipeline (PID file), child training processes, and optional dashboard.
# Usage:  bash scripts/kill_pipeline.sh
#         bash scripts/kill_pipeline.sh --clean   # same + wipe runs/logs, state, exports (not checkpoints)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="${ROOT}/runs/pipeline.pid"
CLEAN=""
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN=1
fi

if [[ -f "${PIDFILE}" ]]; then
  pid="$(cat "${PIDFILE}" 2>/dev/null || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -0 "${pid}" 2>/dev/null && kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PIDFILE}"
fi

pkill -f "scripts/run_pipeline.py" 2>/dev/null || true
pkill -f "scripts/train_finetune.py" 2>/dev/null || true
pkill -f "scripts/train_lora.py" 2>/dev/null || true
pkill -f "scripts/pipeline_dashboard.py" 2>/dev/null || true

echo "Stopped pipeline-related processes (if any were running)."

if [[ -n "${CLEAN}" ]]; then
  rm -rf "${ROOT}/runs/logs"
  mkdir -p "${ROOT}/runs/logs"
  rm -f "${ROOT}/runs/pipeline_state.json"
  rm -rf "${ROOT}/runs/pipeline_exports" 2>/dev/null || true
  echo "Cleaned runs/logs, runs/pipeline_state.json, runs/pipeline_exports (checkpoints/ untouched)."
fi
