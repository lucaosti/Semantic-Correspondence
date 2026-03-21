#!/usr/bin/env python3
"""
Terminal dashboard for monitoring a running pipeline (log tail + parsed metrics).

Run from the repository root in a **second** terminal while ``run_pipeline.py`` or training
scripts are running::

    pip install -e ".[dashboard]"
    python scripts/pipeline_dashboard.py

Press Ctrl+C to exit. By default it follows ``runs/logs/current.log`` (symlink to the active run) if present,
else the newest ``runs/logs/pipeline_*.log``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from rich import box
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:  # pragma: no cover - optional dependency
    print(
        "The dashboard requires Rich. Install with:\n"
        "  pip install -e \".[dashboard]\"\n"
        "or: pip install rich>=13",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_pipeline_log(log_dir: Path) -> Optional[Path]:
    if not log_dir.is_dir():
        return None
    logs = list(log_dir.glob("pipeline_*.log"))
    if not logs:
        return None
    return max(logs, key=lambda p: p.stat().st_mtime)


def _resolve_log_file(log_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    """Prefer ``--file``, then ``current.log`` symlink, then newest ``pipeline_*.log``."""
    if explicit is not None:
        return explicit if explicit.is_file() else None
    current = log_dir / "current.log"
    if current.exists():
        try:
            resolved = current.resolve()
            if resolved.is_file():
                return resolved
        except OSError:
            pass
    return _find_latest_pipeline_log(log_dir)


def _read_tail_lines(path: Path, max_lines: int, max_bytes: int = 2_000_000) -> List[str]:
    """Return up to ``max_lines`` non-empty tail lines (read from end if file is huge)."""
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= max_bytes:
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-max_lines:] if len(lines) > max_lines else lines

    with open(path, "rb") as f:
        f.seek(0, 2)
        end = f.tell()
        chunk = min(max_bytes, end)
        f.seek(end - chunk)
        data = f.read().decode("utf-8", errors="replace")
    lines = data.splitlines()
    return lines[-max_lines:]


def _parse_metrics(lines: List[str]) -> Dict[str, str]:
    """Extract best-effort fields from training log lines."""
    out: Dict[str, str] = {}
    joined = "\n".join(lines)

    m = re.search(
        r"train_(?:finetune|lora):\s*device=(\S+)\s+backbone=(\S+)\s+"
        r"epochs=(\d+)\s+batches_per_epoch=(\d+)",
        joined,
    )
    if m:
        out["device"] = m.group(1)
        out["backbone"] = m.group(2)
        out["epochs_target"] = m.group(3)
        out["batches_per_epoch"] = m.group(4)

    m = re.search(r"epoch\s+(\d+)/(\d+)\s+\(train batches:\s+(\d+)\)", joined)
    if m:
        out["epoch_current"] = m.group(1)
        out["epoch_total"] = m.group(2)
        out["train_batches"] = m.group(3)

    m = None
    for line in reversed(lines):
        if "epoch=" in line and "train_loss=" in line and "val_loss=" in line:
            m = re.search(
                r"epoch=(\d+)\s+train_loss=([\d.eE+-]+)\s+val_loss=([\d.eE+-]+)",
                line,
            )
            if m:
                out["last_completed_epoch"] = m.group(1)
                out["last_train_loss"] = m.group(2)
                out["last_val_loss"] = m.group(3)
                break

    for line in reversed(lines):
        if re.search(r"batch\s+\d+/\d+", line):
            m = re.search(r"batch\s+(\d+)/(\d+)", line)
            if m:
                out["batch_progress"] = f"{m.group(1)} / {m.group(2)}"
                break

    return out


def _read_manifest_tail(manifest: Path, n: int = 6) -> List[str]:
    if not manifest.is_file():
        return []
    lines = manifest.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return lines[-n:] if len(lines) > n else lines


def _build_panel(
    *,
    log_path: Path,
    log_lines: List[str],
    metrics: Dict[str, str],
    manifest_lines: List[str],
    utc_now: str,
) -> Layout:
    root = Layout()

    meta = Table.grid(padding=(0, 2))
    meta.add_column(justify="right", style="dim")
    meta.add_column()
    meta.add_row("Log file", str(log_path))
    meta.add_row("Updated (UTC)", utc_now)
    if log_path.is_file():
        try:
            meta.add_row("Size (bytes)", str(log_path.stat().st_size))
        except OSError:
            pass

    tbl = Table(title="Parsed status", box=box.ROUNDED, expand=True)
    tbl.add_column("Field", style="cyan", no_wrap=True)
    tbl.add_column("Value")
    order = [
        ("device", "Device"),
        ("backbone", "Backbone"),
        ("epochs_target", "Epochs (target)"),
        ("epoch_current", "Epoch (in progress)"),
        ("epoch_total", "Epochs (total)"),
        ("train_batches", "Batches / epoch"),
        ("batch_progress", "Batch (last reported)"),
        ("last_completed_epoch", "Last finished epoch #"),
        ("last_train_loss", "Last train loss"),
        ("last_val_loss", "Last val loss"),
    ]
    for key, label in order:
        if key in metrics:
            tbl.add_row(label, metrics[key])
    if len(tbl.rows) == 0:
        tbl.add_row("—", "No metrics parsed yet (training may still be loading weights).")

    log_text = Text("\n".join(log_lines) if log_lines else "(empty log)")
    log_panel = Panel(log_text, title="Log tail", border_style="blue")

    man_text = Text("\n".join(manifest_lines) if manifest_lines else "(no manifest)")
    man_panel = Panel(man_text, title="manifest.tsv (tail)", border_style="magenta")

    upper = Panel(Group(meta, tbl), title="Pipeline dashboard", border_style="green")
    root.split_column(
        Layout(upper, name="upper"),
        Layout(log_panel, name="log", ratio=2),
        Layout(man_panel, name="manifest", size=max(4, min(10, len(manifest_lines) + 3))),
    )
    return root


def main() -> int:
    p = argparse.ArgumentParser(description="Live terminal dashboard for pipeline logs.")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory containing pipeline_*.log (default: <repo>/runs/logs).",
    )
    p.add_argument("--file", type=Path, default=None, help="Follow this log file instead of the newest.")
    p.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds.")
    p.add_argument("--tail", type=int, default=40, help="Number of log lines to show.")
    args = p.parse_args()

    cwd = _repo_root()
    log_dir = args.log_dir or (cwd / "runs" / "logs")
    manifest = cwd / "runs" / "logs" / "manifest.tsv"

    log_path = _resolve_log_file(log_dir, args.file)
    if log_path is None or not log_path.is_file():
        print(f"No pipeline log found under {log_dir}", file=sys.stderr)
        print("Start training or pass --file /path/to/pipeline_*.log", file=sys.stderr)
        return 2

    console = Console()

    def render() -> Layout:
        lines = _read_tail_lines(log_path, args.tail)
        metrics = _parse_metrics(lines)
        man = _read_manifest_tail(manifest, 8)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return _build_panel(
            log_path=log_path,
            log_lines=lines,
            metrics=metrics,
            manifest_lines=man,
            utc_now=now,
        )

    console.print(
        "[dim]Following[/dim]", Text(str(log_path), style="bold"), "[dim]— Ctrl+C to quit[/dim]"
    )
    rps = min(10.0, max(0.25, 1.0 / max(args.interval, 0.05)))
    try:
        with Live(render(), console=console, refresh_per_second=rps) as live:
            while True:
                time.sleep(args.interval)
                live.update(render())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard closed.[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
