#!/usr/bin/env python3
"""
Terminal dashboard for the detached pipeline workflow.

**Architecture:** ``run_pipeline.py`` (often via ``start_pipeline_detached.sh``) appends *all* stage
output—including subprocesses—to ``runs/logs/pipeline_*.log`` and symlinks ``current.log``.
This script re-reads that file every second (default), parses training lines (epoch, batch, losses),
draws **progress bars** (global across epochs + batch within the current epoch), and shows the tail of
``stage_events.jsonl`` for the stage timeline.

Install Rich (once): ``pip install -e ".[dashboard]"``

Run in a second terminal while the pipeline is running::

    bash scripts/reconnect_dashboard.sh
    # or: python scripts/pipeline_dashboard.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
        '  pip install -e ".[dashboard]"\n'
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
    return lines[-max_lines:] if len(lines) > max_lines else lines


def _parse_metrics(lines: List[str]) -> Dict[str, str]:
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

    m = re.search(r"resume_save_interval=(\d+)", joined)
    if m:
        out["resume_save_interval"] = m.group(1)

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


@dataclass
class TrainingProgress:
    """Derived from the log tail; None means unknown."""

    epoch_current: Optional[int] = None  # 1-based, from "epoch k/total (train batches:"
    epoch_total: Optional[int] = None
    batches_per_epoch: Optional[int] = None
    batch_current: Optional[int] = None  # last "batch i/n" during train
    batch_total: Optional[int] = None


def _parse_training_progress(lines: List[str]) -> TrainingProgress:
    p = TrainingProgress()
    joined = "\n".join(lines)

    m = re.search(r"epoch\s+(\d+)/(\d+)\s+\(train batches:\s+(\d+)\)", joined)
    if m:
        p.epoch_current = int(m.group(1))
        p.epoch_total = int(m.group(2))
        p.batches_per_epoch = int(m.group(3))

    for line in reversed(lines):
        if re.search(r"\bbatch\s+\d+/\d+", line):
            m = re.search(r"\bbatch\s+(\d+)/(\d+)", line)
            if m:
                p.batch_current = int(m.group(1))
                p.batch_total = int(m.group(2))
                break

    if p.batch_total is None and p.batches_per_epoch is not None:
        p.batch_total = p.batches_per_epoch

    return p


def _bar_markup(frac: float, width: int = 32) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(width * frac))
    filled = min(width, max(0, filled))
    bar = "[green]" + "█" * filled + "[/][dim]" + "░" * (width - filled) + "[/]"
    pct = f"{100.0 * frac:.1f}%"
    return f"{bar} [bold]{pct}[/]"


def _global_training_fraction(p: TrainingProgress) -> Optional[float]:
    """Completed fraction across all epochs (train phase only; best-effort)."""
    et = p.epoch_total
    ec = p.epoch_current
    if et is None or ec is None or et <= 0:
        return None
    ec = max(1, min(ec, et + 1))
    within = 0.0
    bt = p.batch_total
    bc = p.batch_current
    if bt and bt > 0 and bc is not None:
        within = max(0.0, min(1.0, bc / bt))
    done_epochs = max(0, ec - 1)
    return max(0.0, min(1.0, (done_epochs + within) / et))


def _progress_panel_lines(p: TrainingProgress) -> List[str]:
    lines: List[str] = []
    if p.epoch_current is not None and p.epoch_total is not None:
        lines.append(
            f"[cyan]Epoca[/] [bold]{p.epoch_current}[/] / [bold]{p.epoch_total}[/]  "
            f"(batch totali per epoca: {p.batches_per_epoch or '?'})"
        )
    else:
        lines.append("[dim]Epoca in corso: non ancora nel log (caricamento pesi / verify / altro stage).[/]")

    g = _global_training_fraction(p)
    if g is not None:
        lines.append("[yellow]Progress globale (tutte le epoche, fase train)[/] " + _bar_markup(g))
    else:
        lines.append("[dim]Barra globale: attendere la riga «epoch k/total (train batches: …)».[/]")

    bt = p.batch_total
    bc = p.batch_current
    if bt and bt > 0 and bc is not None:
        frac_b = max(0.0, min(1.0, bc / bt))
        lines.append(f"[magenta]Nell'epoca corrente (batch)[/] {bc:,} / {bt:,}  " + _bar_markup(frac_b))
    elif p.epoch_current is not None:
        lines.append(
            "[dim]Batch nell’epoca: nessuna riga «batch i/n» nel tail — "
            "il training stampa ogni LOG_BATCH_INTERVAL step (es. 2500).[/]"
        )
    else:
        lines.append("[dim]Batch nell’epoca: —[/]")

    return lines


def _read_jsonl_tail(path: Path, max_lines: int) -> List[Dict[str, Any]]:
    if not path.is_file() or max_lines <= 0:
        return []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows: List[Dict[str, Any]] = []
    for line in raw[-max_lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _read_pipeline_state_summary(cwd: Path) -> str:
    p = cwd / "runs" / "pipeline_state.json"
    if not p.is_file():
        return "No pipeline_state.json (fresh run or cleared)."
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "(unreadable pipeline_state.json)"
    comp = data.get("completed")
    fp = data.get("fingerprint")
    if not isinstance(comp, list):
        comp = []
    fp_s = str(fp)[:16] + "…" if isinstance(fp, str) and len(str(fp)) > 16 else str(fp)
    return f"Completed stages: {len(comp)}  |  fingerprint: {fp_s}"


def _read_manifest_tail(manifest: Path, n: int = 6) -> List[str]:
    if not manifest.is_file():
        return []
    lines = manifest.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    return lines[-n:] if len(lines) > n else lines


def _build_layout(
    *,
    log_path: Path,
    log_lines: List[str],
    metrics: Dict[str, str],
    progress_lines: List[str],
    events_rows: List[Dict[str, Any]],
    state_summary: str,
    manifest_lines: List[str],
    utc_now: str,
    tail_n: int,
) -> Layout:
    root = Layout()

    how = (
        "[cyan]run_pipeline.py[/cyan] → one log file + [magenta]stage_events.jsonl[/magenta]. "
        f"Refresh ~1s; tail [green]{tail_n}[/green] lines."
    )
    how_panel = Panel(Text.from_markup(how), title="Wiring", border_style="yellow")

    meta = Table.grid(padding=(0, 2))
    meta.add_column(justify="right", style="dim")
    meta.add_column()
    meta.add_row("Primary log", str(log_path))
    meta.add_row("Updated (UTC)", utc_now)
    if log_path.is_file():
        try:
            meta.add_row("Log size (bytes)", str(log_path.stat().st_size))
        except OSError:
            pass
    meta.add_row("Resume state", state_summary)

    tbl = Table(title="Training (parsed from log tail)", box=box.ROUNDED, expand=True)
    tbl.add_column("Field", style="cyan", no_wrap=True)
    tbl.add_column("Value")
    order = [
        ("device", "Device"),
        ("backbone", "Backbone"),
        ("epochs_target", "Epochs (target)"),
        ("epoch_current", "Epoch (in progress, 1-based)"),
        ("epoch_total", "Epochs (total)"),
        ("train_batches", "Batches / epoch"),
        ("resume_save_interval", "Resume save every N batches"),
        ("batch_progress", "Batch (last reported)"),
        ("last_completed_epoch", "Last finished epoch #"),
        ("last_train_loss", "Last train loss"),
        ("last_val_loss", "Last val loss"),
    ]
    for key, label in order:
        if key in metrics:
            tbl.add_row(label, metrics[key])
    if len(tbl.rows) == 0:
        tbl.add_row("—", "No training lines in tail yet (verify / download / model load).")

    prog_panel = Panel(
        Group(*[Text.from_markup(x) for x in progress_lines]),
        title="Avanzamento training",
        border_style="bright_blue",
    )

    ev = Table(title="stage_events.jsonl (tail)", box=box.SIMPLE, expand=True)
    ev.add_column("time (UTC)", style="dim", max_width=22)
    ev.add_column("action", style="magenta")
    ev.add_column("stage_id")
    ev.add_column("extra", max_width=36)
    for row in events_rows:
        ts = str(row.get("ts_utc", ""))[:19]
        act = str(row.get("action", ""))
        sid = str(row.get("stage_id", ""))
        extra_keys = sorted(k for k in row if k not in ("ts_utc", "action", "stage_id"))
        extra = " ".join(f"{k}={row[k]!s}"[:40] for k in extra_keys[:2])
        ev.add_row(ts, act, sid, extra)
    if len(events_rows) == 0:
        ev.add_row("—", "—", "—", "(empty or missing)")

    log_text = Text("\n".join(log_lines) if log_lines else "(empty log)")
    log_panel = Panel(log_text, title=f"Log tail (last {tail_n} lines)", border_style="blue")

    man_text = Text("\n".join(manifest_lines) if manifest_lines else "(no manifest)")
    man_panel = Panel(man_text, title="manifest.tsv (tail)", border_style="magenta")

    upper = Panel(Group(how_panel, meta, prog_panel, tbl), title="Pipeline dashboard", border_style="green")
    root.split_column(
        Layout(upper, name="upper"),
        Layout(Panel(ev, border_style="cyan"), name="events", ratio=1),
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
    p.add_argument(
        "--tail",
        type=int,
        default=96,
        help="Log lines to load (higher = epoch header / batch lines stay visible longer).",
    )
    p.add_argument(
        "--events-tail",
        type=int,
        default=14,
        help="How many last lines of stage_events.jsonl to parse.",
    )
    args = p.parse_args()

    cwd = _repo_root()
    log_dir = args.log_dir or (cwd / "runs" / "logs")
    manifest = cwd / "runs" / "logs" / "manifest.tsv"
    events_path = cwd / "runs" / "logs" / "stage_events.jsonl"

    log_path = _resolve_log_file(log_dir, args.file)
    if log_path is None:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"No pipeline log yet under {log_dir}.\n"
            "Start the runner:  bash scripts/start_pipeline_detached.sh\n"
            "Then reopen this dashboard.",
            file=sys.stderr,
        )
        return 2
    if not log_path.is_file():
        print(f"Log path is not a file: {log_path}", file=sys.stderr)
        return 2

    console = Console()

    def render() -> Layout:
        lines = _read_tail_lines(log_path, args.tail)
        metrics = _parse_metrics(lines)
        progress_lines = _progress_panel_lines(_parse_training_progress(lines))
        events_rows = _read_jsonl_tail(events_path, args.events_tail)
        state_summary = _read_pipeline_state_summary(cwd)
        man = _read_manifest_tail(manifest, 8)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return _build_layout(
            log_path=log_path,
            log_lines=lines,
            metrics=metrics,
            progress_lines=progress_lines,
            events_rows=events_rows,
            state_summary=state_summary,
            manifest_lines=man,
            utc_now=now,
            tail_n=args.tail,
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
