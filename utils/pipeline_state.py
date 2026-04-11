"""
Persistent pipeline progress for :mod:`scripts.run_pipeline`.

Saves ``runs/pipeline_state.json`` (gitignored with ``runs/``) so interrupted runs can skip
completed stages. A **fingerprint** hashes training/eval configuration; if settings change, saved
progress is ignored so checkpoints are not mixed across incompatible configs.

Environment:

* ``SEMANTIC_CORRESPONDENCE_PIPELINE_RESET`` — if non-empty, delete state at pipeline start
  (full restart while keeping checkpoints on disk).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

SCHEMA_VERSION = 1


def state_path(repo_root: Path) -> Path:
    custom = os.environ.get("SEMANTIC_CORRESPONDENCE_STATE_PATH", "").strip()
    if custom:
        return Path(custom)
    return repo_root / "runs" / "pipeline_state.json"


def stage_events_path(repo_root: Path) -> Path:
    return repo_root / "runs" / "logs" / "stage_events.jsonl"


def fingerprint_from_config(config: Mapping[str, Any]) -> str:
    """Stable SHA-256 hex digest of JSON-serializable config keys."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(repo_root: Path) -> Optional[Dict[str, Any]]:
    path = state_path(repo_root)
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None


def save_state(repo_root: Path, data: Dict[str, Any]) -> None:
    path = state_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {**data, "schema_version": SCHEMA_VERSION, "updated_at_utc": _utc_iso()}
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def should_reset_from_env() -> bool:
    return bool(os.environ.get("SEMANTIC_CORRESPONDENCE_PIPELINE_RESET", "").strip())


def append_stage_event(repo_root: Path, record: Dict[str, Any]) -> None:
    """Append one JSON object per line for dashboards and post-mortems."""
    p = stage_events_path(repo_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts_utc": _utc_iso(), **record}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def is_step_done(completed: List[str], step_id: str) -> bool:
    return step_id in completed


def mark_step_done(repo_root: Path, state: Dict[str, Any], step_id: str) -> None:
    comp = state.setdefault("completed", [])
    if step_id not in comp:
        comp.append(step_id)
    save_state(repo_root, state)


__all__ = [
    "SCHEMA_VERSION",
    "append_stage_event",
    "fingerprint_from_config",
    "is_step_done",
    "load_state",
    "mark_step_done",
    "save_state",
    "should_reset_from_env",
    "stage_events_path",
    "state_path",
]
