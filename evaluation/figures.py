"""
Paper-grade figures and tables from pipeline exports.

Loads ``runs/pipeline_exports/*.json`` written by :func:`scripts.run_pipeline` and
emits Markdown / LaTeX / CSV tables and matplotlib figures (PDF + PNG) consumed by
``AML_Analysis.ipynb``.

The run-name parser is the single source of truth for mapping the evaluator's
``name`` field (e.g. ``dinov2_vitb14_ft_lb2_wsa``) into a structured record.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PCK_ALPHAS_DEFAULT: Tuple[float, ...] = (0.05, 0.1, 0.2)
BACKBONES_ORDER: Tuple[str, ...] = ("dinov2_vitb14", "dinov3_vitb16", "sam_vit_b")
METHODS_ORDER: Tuple[str, ...] = ("baseline", "ft_lb1", "ft_lb2", "ft_lb4", "lora")

_DIFFICULTY_FLAGS: Tuple[str, ...] = (
    "viewpoint_variation",
    "scale_variation",
    "truncation",
    "occlusion",
)


# ---------------------------------------------------------------------------
# Run-name parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunInfo:
    """Structured representation of an eval run name."""

    name: str
    backbone: str
    method: str  # one of: baseline, ft_lbN, lora
    last_blocks: Optional[int]
    wsa: bool
    wsa_window: Optional[int] = None
    layer_index: Optional[int] = None


_RUN_NAME_RE = re.compile(
    r"^(?P<bb>dinov2_vitb14|dinov3_vitb16|sam_vit_b)"
    r"_(?P<rest>.+)$"
)


def parse_run_name(name: str) -> Optional[RunInfo]:
    """Parse a run name produced by ``scripts/run_pipeline.py``."""
    m = _RUN_NAME_RE.match(name)
    if not m:
        return None
    bb = m.group("bb")
    rest = m.group("rest")

    m_wsa_sweep = re.match(r"^baseline_wsa_w(\d+)$", rest)
    if m_wsa_sweep:
        return RunInfo(name, bb, "baseline", None, True, wsa_window=int(m_wsa_sweep.group(1)))
    m_layer = re.match(r"^baseline_layer(\d+)$", rest)
    if m_layer:
        return RunInfo(name, bb, "baseline", None, False, layer_index=int(m_layer.group(1)))

    wsa = rest.endswith("_wsa")
    if wsa:
        rest = rest[: -len("_wsa")]
    if rest == "baseline":
        return RunInfo(name, bb, "baseline", None, wsa)
    if rest == "lora":
        return RunInfo(name, bb, "lora", None, wsa)
    m2 = re.match(r"^ft_lb(\d+)$", rest)
    if m2:
        n = int(m2.group(1))
        return RunInfo(name, bb, f"ft_lb{n}", n, wsa)
    return None


# ---------------------------------------------------------------------------
# Export loading
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pck_exports(exports_dir: Path) -> Dict[str, Any]:
    """Load every JSON export into one dict.

    Returns a dict with these keys (each is ``None`` if the file is missing):
    ``pck_results``, ``per_category``, ``by_difficulty_flag``, ``wsa_sweep``,
    ``layer_sweep``. The ``available`` sub-dict reports presence on disk.
    """
    exports_dir = Path(exports_dir)
    files = {
        "pck_results": exports_dir / "pck_results.json",
        "per_category": exports_dir / "pck_results_per_category.json",
        "by_difficulty_flag": exports_dir / "pck_results_by_difficulty_flag.json",
        "wsa_sweep": exports_dir / "pck_results_wsa_sweep.json",
        "layer_sweep": exports_dir / "pck_results_layer_sweep.json",
    }
    payload: Dict[str, Any] = {k: _read_json(p) for k, p in files.items()}
    payload["pck_results"] = payload.get("pck_results") or []
    payload["available"] = {k: p.is_file() for k, p in files.items()}
    return payload


# ---------------------------------------------------------------------------
# Master table
# ---------------------------------------------------------------------------


def build_master_table(pck_data: Dict[str, Any]) -> "Any":
    """Build a paper-ready master comparison DataFrame.

    Rows: ``(backbone, method, wsa)`` triples; columns: ``pck@α`` (image macro)
    and ``pck_pt@α`` (point micro) per α reported in each run.
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for r in pck_data.get("pck_results", []) or []:
        info = parse_run_name(str(r.get("name", "")))
        if info is None:
            continue
        row: Dict[str, Any] = {
            "backbone": info.backbone,
            "method": info.method,
            "last_blocks": info.last_blocks,
            "wsa": info.wsa,
        }
        row.update({k: v for k, v in (r.get("metrics") or {}).items()
                    if isinstance(v, (int, float))})
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["backbone", "method", "last_blocks", "wsa"])
    df = pd.DataFrame(rows)

    bb_order = {bb: i for i, bb in enumerate(BACKBONES_ORDER)}
    method_order = {m: i for i, m in enumerate(METHODS_ORDER)}
    df["_bb_order"] = df["backbone"].map(bb_order).fillna(99)
    df["_method_order"] = df["method"].map(method_order).fillna(99)
    df = df.sort_values(["_bb_order", "_method_order", "wsa"], kind="stable").drop(
        columns=["_bb_order", "_method_order"]
    )
    return df.reset_index(drop=True)


def _bold_max_per_column(df: "Any", metric_cols: Sequence[str], fmt: str) -> "Any":
    """Wrap the column-wise max in LaTeX ``\\textbf{...}``; format the rest with ``fmt``."""
    import numpy as np
    import pandas as pd

    out = df.copy()
    for c in metric_cols:
        if c not in out.columns:
            continue
        vals = pd.to_numeric(out[c], errors="coerce")
        if not vals.notna().any():
            continue
        best = vals.max()
        formatted = vals.map(lambda v: fmt.format(v) if pd.notna(v) else "--")
        mask = (vals == best) & vals.notna()
        out[c] = np.where(mask, formatted.map(lambda s: r"\textbf{" + s + "}"), formatted)
    return out


def dataframe_to_latex(
    df: "Any",
    *,
    metric_cols: Optional[Sequence[str]] = None,
    fmt: str = "{:.4f}",
    caption: str = "PCK comparison.",
    label: str = "tab:pck",
) -> str:
    """Render a master DataFrame as a LaTeX ``tabular`` with column-wise bold for the best."""
    import pandas as pd

    if df.empty:
        return "% empty table\n"
    if metric_cols is None:
        metric_cols = [c for c in df.columns if str(c).startswith(("pck@", "pck_pt@"))]
    df_bold = _bold_max_per_column(df, metric_cols, fmt)

    for c in df_bold.columns:
        if c in metric_cols:
            continue
        df_bold[c] = df_bold[c].map(lambda v: "" if pd.isna(v) else str(v))

    n_cols = len(df_bold.columns)
    col_spec = "l" * (n_cols - len(metric_cols)) + "r" * len(metric_cols)
    header = " & ".join(str(c).replace("_", r"\_") for c in df_bold.columns) + r" \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for _, row in df_bold.iterrows():
        cells = " & ".join(
            str(v).replace("_", r"\_") if not str(v).startswith(r"\textbf") else str(v)
            for v in row.tolist()
        )
        lines.append(cells + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def dataframe_to_markdown(df: "Any", *, fmt: str = "{:.4f}") -> str:
    """Render a master DataFrame as a Markdown table."""
    import pandas as pd

    if df.empty:
        return "_(empty)_\n"
    out = df.copy()
    metric_cols = [c for c in out.columns if str(c).startswith(("pck@", "pck_pt@"))]
    for c in metric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").map(
            lambda v: "--" if pd.isna(v) else fmt.format(v)
        )
    for c in out.columns:
        if c in metric_cols:
            continue
        out[c] = out[c].map(lambda v: "" if pd.isna(v) else str(v))
    return out.to_markdown(index=False)


# ---------------------------------------------------------------------------
# Plotting style + IO
# ---------------------------------------------------------------------------


def apply_paper_style() -> None:
    """Set matplotlib rcParams for paper-ready figures (serif, larger ticks)."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def save_figure_dual(fig: "Any", out_path_no_ext: Path, *, dpi: int = 300) -> Tuple[Path, Path]:
    """Save ``fig`` as PDF (vector) + PNG (raster, 300 dpi by default)."""
    out_path_no_ext = Path(out_path_no_ext)
    out_path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_path_no_ext.with_suffix(".pdf")
    png = out_path_no_ext.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=int(dpi))
    return pdf, png


# ---------------------------------------------------------------------------
# Plot: WSA gain
# ---------------------------------------------------------------------------


def _master_pivot(df_master: "Any", metric: str) -> "Any":
    """Pivot so we get ``(backbone, method) -> {without_wsa, with_wsa}``."""
    keep = df_master[df_master[metric].notna()].copy()
    keep["bm"] = keep["backbone"] + "/" + keep["method"]
    return keep.pivot_table(index="bm", columns="wsa", values=metric, aggfunc="first")


def plot_wsa_gain(df_master: "Any", out_path: Path, *, alpha: float = 0.1) -> Tuple[Path, Path]:
    """Paired bars showing PCK without/with WSA per ``(backbone, method)``."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_style()
    metric = f"pck@{alpha:g}"
    pivot = _master_pivot(df_master, metric)
    if pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    pivot = pivot.reindex(columns=[c for c in (False, True) if c in pivot.columns])
    labels = list(pivot.index)
    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 4))
    if False in pivot.columns:
        ax.bar(x - width / 2, pivot[False].values, width, label="argmax")
    if True in pivot.columns:
        ax.bar(x + width / 2, pivot[True].values, width, label="+ WSA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"WSA gain at α = {alpha:g}")
    ax.legend()
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


# ---------------------------------------------------------------------------
# Plot: fine-tune depth
# ---------------------------------------------------------------------------


def plot_ft_depth(
    df_master: "Any", out_path: Path, *, alphas: Sequence[float] = PCK_ALPHAS_DEFAULT
) -> Tuple[Path, Path]:
    """PCK vs unfrozen-blocks for each backbone (PDF Stage 2)."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    df = df_master[(df_master["method"].str.startswith("ft_lb")) & (~df_master["wsa"])].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No fine-tune runs", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    backbones = [bb for bb in BACKBONES_ORDER if bb in set(df["backbone"])]
    fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), 4), squeeze=False)
    for i, bb in enumerate(backbones):
        ax = axes[0, i]
        sub = df[df["backbone"] == bb].sort_values("last_blocks")
        for a in alphas:
            col = f"pck@{a:g}"
            if col in sub.columns:
                ax.plot(sub["last_blocks"], sub[col], marker="o", label=col)
        ax.set_xlabel("Unfrozen last blocks")
        ax.set_ylabel("PCK (image macro)")
        ax.set_title(bb)
        ax.set_xticks(sorted(sub["last_blocks"].dropna().unique().astype(int)))
        ax.legend(fontsize=8)
    fig.suptitle("Fine-tune depth effect (no WSA)")
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


# ---------------------------------------------------------------------------
# Trainable-param accounting + LoRA-vs-FT scatter
# ---------------------------------------------------------------------------


def estimate_trainable_params(
    backbone: str,
    method: str,
    *,
    ckpt_dir: Path,
    last_blocks: Optional[int] = None,
    rank: int = 8,
) -> Optional[int]:
    """Best-effort estimate of *trainable* parameters per (backbone, method).

    For ``ft_lbN`` reports the number of params under the last N transformer blocks
    in the saved state dict; for ``lora`` reports only the ``lora_a`` / ``lora_b``
    tensors. Returns ``None`` if the checkpoint is missing or unreadable.
    """
    try:
        import torch
    except Exception:
        return None
    ckpt_dir = Path(ckpt_dir)
    if method == "lora":
        path = ckpt_dir / f"{backbone}_lora_r{rank}_best.pt"
    elif method.startswith("ft_lb") and last_blocks is not None:
        path = ckpt_dir / f"{backbone}_lastblocks{last_blocks}_best.pt"
    else:
        return None
    if not path.is_file():
        return None
    try:
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(str(path), map_location="cpu")
    except Exception:
        return None
    state = blob.get("model") if isinstance(blob, dict) else None
    if not isinstance(state, dict):
        return None
    if method == "lora":
        return sum(int(v.numel()) for k, v in state.items() if "lora_" in k and hasattr(v, "numel"))
    last_blocks = int(last_blocks) if last_blocks is not None else 0
    block_keys = [k for k in state.keys() if ".blocks." in k]
    block_indices = sorted({int(k.split(".blocks.")[1].split(".")[0]) for k in block_keys
                            if k.split(".blocks.")[1].split(".")[0].isdigit()})
    if not block_indices:
        return None
    cutoff = block_indices[-last_blocks] if last_blocks <= len(block_indices) else block_indices[0]
    return sum(
        int(v.numel()) for k, v in state.items()
        if ".blocks." in k
        and k.split(".blocks.")[1].split(".")[0].isdigit()
        and int(k.split(".blocks.")[1].split(".")[0]) >= cutoff
        and hasattr(v, "numel")
    )


def plot_lora_vs_ft(
    df_master: "Any",
    ckpt_dir: Path,
    out_path: Path,
    *,
    alpha: float = 0.1,
    rank: int = 8,
) -> Tuple[Path, Path]:
    """Scatter trainable-param-count vs PCK for adapted methods (log-x)."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    metric = f"pck@{alpha:g}"
    df = df_master[(~df_master["wsa"]) & (df_master["method"] != "baseline")].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No adapted runs", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for bb in [b for b in BACKBONES_ORDER if b in set(df["backbone"])]:
        sub = df[df["backbone"] == bb]
        xs: List[int] = []
        ys: List[float] = []
        labels_pts: List[str] = []
        for _, row in sub.iterrows():
            n = estimate_trainable_params(
                row["backbone"], row["method"], ckpt_dir=ckpt_dir,
                last_blocks=row.get("last_blocks"), rank=rank,
            )
            v = row.get(metric)
            if n is None or v is None or n <= 0:
                continue
            xs.append(n)
            ys.append(float(v))
            labels_pts.append(f"{row['method']}")
        if xs:
            ax.scatter(xs, ys, label=bb, s=70, alpha=0.8)
            for x, y, lbl in zip(xs, ys, labels_pts):
                ax.annotate(lbl, (x, y), xytext=(4, 2), textcoords="offset points",
                            fontsize=7, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log)")
    ax.set_ylabel(metric)
    ax.set_title(f"Parameter efficiency at α = {alpha:g}")
    ax.legend(title="backbone")
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


# ---------------------------------------------------------------------------
# Per-category heatmap
# ---------------------------------------------------------------------------


def per_category_table(pck_data: Dict[str, Any], *, alpha: float = 0.1) -> "Any":
    """Build a (run × category) table for the requested α (image-macro)."""
    import pandas as pd

    per_cat = pck_data.get("per_category") or []
    metric_key = f"pck@{alpha:g}"
    rows: List[Dict[str, Any]] = []
    for entry in per_cat:
        for cat, alphas in (entry.get("categories") or {}).items():
            v = alphas.get(metric_key)
            if v is None:
                continue
            rows.append({"run": entry.get("name"), "category": cat, "value": float(v)})
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .pivot_table(index="run", columns="category", values="value", aggfunc="first")
    )


def plot_per_category_heatmap(
    pck_data: Dict[str, Any], out_path: Path, *, alpha: float = 0.1
) -> Tuple[Path, Path]:
    """Per-category heatmap (rows = run, cols = SPair category)."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_style()
    pivot = per_category_table(pck_data, alpha=alpha)
    if pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No per-category data", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.6),
                                    max(5, pivot.shape[0] * 0.32)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="black" if 0.3 < v < 0.85 else "white")
    fig.colorbar(im, ax=ax, label=f"pck@{alpha:g}")
    ax.set_title(f"Per-category PCK at α = {alpha:g}")
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


# ---------------------------------------------------------------------------
# Per-difficulty
# ---------------------------------------------------------------------------


def per_difficulty_table(pck_data: Dict[str, Any], *, alpha: float = 0.1) -> "Any":
    """Long table of per-flag, per-bucket PCK values."""
    import pandas as pd

    by_diff = pck_data.get("by_difficulty_flag") or []
    metric_key = f"custom_pck{alpha}"
    rows: List[Dict[str, Any]] = []
    for entry in by_diff:
        run = entry.get("name")
        for flag, buckets in (entry.get("data") or {}).items():
            for bucket, info in buckets.items():
                summary = ((info or {}).get("summary") or {}).get("image") or {}
                v = (summary.get(metric_key) or {}).get("all")
                if v is None:
                    continue
                rows.append({"run": run, "flag": flag, "bucket": int(bucket),
                             "value": float(v)})
    return pd.DataFrame(rows)


def plot_per_difficulty_bars(
    pck_data: Dict[str, Any], out_path: Path, *, alpha: float = 0.1
) -> Tuple[Path, Path]:
    """Grouped bars: low (0) vs high (1) difficulty per flag for each run."""
    import matplotlib.pyplot as plt
    import numpy as np

    apply_paper_style()
    df = per_difficulty_table(pck_data, alpha=alpha)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No difficulty data", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    flags = [f for f in _DIFFICULTY_FLAGS if f in set(df["flag"])]
    runs = sorted(set(df["run"]))
    fig, axes = plt.subplots(1, len(flags), figsize=(4 * len(flags), 4), squeeze=False, sharey=True)
    for i, flag in enumerate(flags):
        ax = axes[0, i]
        sub = df[df["flag"] == flag]
        x = np.arange(len(runs))
        w = 0.4
        low = [float(sub[(sub["run"] == r) & (sub["bucket"] == 0)]["value"].mean()) if (
            (sub["run"] == r) & (sub["bucket"] == 0)).any() else float("nan") for r in runs]
        high = [float(sub[(sub["run"] == r) & (sub["bucket"] == 1)]["value"].mean()) if (
            (sub["run"] == r) & (sub["bucket"] == 1)).any() else float("nan") for r in runs]
        ax.bar(x - w / 2, low, w, label="bucket 0 (easy)")
        ax.bar(x + w / 2, high, w, label="bucket 1 (hard)")
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=80, ha="right", fontsize=7)
        ax.set_title(flag.replace("_", " "))
        if i == 0:
            ax.set_ylabel(f"pck@{alpha:g}")
            ax.legend(fontsize=8)
    fig.suptitle(f"Per-difficulty PCK at α = {alpha:g}")
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def load_training_histories(history_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load every ``*_history.jsonl`` under ``history_dir`` into a dict."""
    history_dir = Path(history_dir)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for f in sorted(history_dir.glob("*_history.jsonl")):
        recs: List[Dict[str, Any]] = []
        with open(f, "r", encoding="utf-8") as fp:
            for raw in fp:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    recs.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue
        if recs:
            out[f.stem.replace("_history", "")] = recs
    return out


def plot_training_curves(history_dir: Path, out_path: Path) -> Tuple[Path, Path]:
    """One subplot per history file: train + val loss with the best epoch annotated."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    histories = load_training_histories(history_dir)
    if not histories:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    n = len(histories)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    for i, (name, recs) in enumerate(sorted(histories.items())):
        ax = axes[i // cols][i % cols]
        ep = [r["epoch"] for r in recs]
        train = [r.get("train_loss") for r in recs]
        val = [r.get("val_loss") for r in recs]
        ax.plot(ep, train, marker=".", label="train")
        ax.plot(ep, val, marker=".", label="val")
        valid = [(e, v) for e, v in zip(ep, val) if v is not None]
        if valid:
            be, bv = min(valid, key=lambda t: t[1])
            ax.axvline(be, color="grey", linestyle=":", alpha=0.6)
            ax.annotate(f"best ep={be}\nval={bv:.4f}", xy=(be, bv), xytext=(6, -22),
                        textcoords="offset points", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=0.6))
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Training curves")
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


# ---------------------------------------------------------------------------
# Sensitivity plots (WSA window, DINO layer)
# ---------------------------------------------------------------------------


def _plot_sweep(
    sweep: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    alpha: float,
    selector: str,  # "wsa_window" or "layer_index"
    xlabel: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt

    apply_paper_style()
    if not sweep:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No sweep data", ha="center", va="center", transform=ax.transAxes)
        return save_figure_dual(fig, out_path)

    metric = f"pck@{alpha:g}"
    by_bb: Dict[str, Dict[int, float]] = {}
    for r in sweep:
        info = parse_run_name(str(r.get("name", "")))
        if info is None:
            continue
        x = getattr(info, selector)
        if x is None:
            continue
        v = float((r.get("metrics") or {}).get(metric, float("nan")))
        by_bb.setdefault(info.backbone, {})[int(x)] = v

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, bb in enumerate(sorted(by_bb.keys())):
        items = sorted(by_bb[bb].items())
        xs = [k for k, _ in items]
        ys = [v for _, v in items]
        c = cmap(i / max(len(by_bb) - 1, 1))
        ax.plot(xs, ys, marker="o", color=c, label=bb, linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_xticks(sorted({k for d in by_bb.values() for k in d}))
    ax.legend()
    fig.tight_layout()
    return save_figure_dual(fig, out_path)


def plot_wsa_window_sensitivity(
    pck_data: Dict[str, Any], out_path: Path, *, alpha: float = 0.1
) -> Tuple[Path, Path]:
    """One curve per backbone of PCK as a function of the WSA window size."""
    return _plot_sweep(
        pck_data.get("wsa_sweep") or [],
        out_path,
        alpha=alpha,
        selector="wsa_window",
        xlabel="WSA window size (px on feature grid)",
        title="WSA window sensitivity",
    )


def plot_dino_layer_sensitivity(
    pck_data: Dict[str, Any], out_path: Path, *, alpha: float = 0.1
) -> Tuple[Path, Path]:
    """One curve per backbone of PCK as a function of the DINO intermediate-layer index."""
    return _plot_sweep(
        pck_data.get("layer_sweep") or [],
        out_path,
        alpha=alpha,
        selector="layer_index",
        xlabel="DINO intermediate-layer index",
        title="DINO layer-index sensitivity (training-free)",
    )


__all__ = [
    "BACKBONES_ORDER",
    "METHODS_ORDER",
    "PCK_ALPHAS_DEFAULT",
    "RunInfo",
    "apply_paper_style",
    "build_master_table",
    "dataframe_to_latex",
    "dataframe_to_markdown",
    "estimate_trainable_params",
    "load_pck_exports",
    "load_training_histories",
    "parse_run_name",
    "per_category_table",
    "per_difficulty_table",
    "plot_dino_layer_sensitivity",
    "plot_ft_depth",
    "plot_lora_vs_ft",
    "plot_per_category_heatmap",
    "plot_per_difficulty_bars",
    "plot_training_curves",
    "plot_wsa_gain",
    "plot_wsa_window_sensitivity",
    "save_figure_dual",
]
