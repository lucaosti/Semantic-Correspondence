#!/usr/bin/env bash
# Stop any previous run, start pipeline in background, then open the dashboard (foreground).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "${ROOT}/scripts/kill_pipeline.sh"
sleep 1
bash "${ROOT}/scripts/start_pipeline_detached.sh"
# Brief pause so current.log exists
sleep 2
exec "${ROOT}/scripts/start_dashboard.sh"
