#!/usr/bin/env bash
# Reopen the Rich terminal dashboard on the active pipeline log (after SSH / new tab).
# Does NOT stop or restart the pipeline — safe to run anytime the driver is already running.
# Log resolution: same as pipeline_dashboard.py — prefer runs/logs/current.log (symlink from the
#   running driver); else newest runs/logs/pipeline_*.log. Override: python scripts/pipeline_dashboard.py --file PATH
# Refresh ~1s while this process runs.
# Plain text: tail -f runs/logs/current.log
# Requires an interactive TTY. Without one, falls back to tail -f on current.log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${ROOT}/runs/logs"
CURRENT="${LOG_DIR}/current.log"

if [[ ! -t 1 ]]; then
  echo "No TTY: streaming ${CURRENT} (Ctrl+C to stop)."
  mkdir -p "${LOG_DIR}"
  touch "${CURRENT}" 2>/dev/null || true
  exec tail -n 80 -f "${CURRENT}"
fi

exec bash "${ROOT}/scripts/start_dashboard.sh" "$@"
