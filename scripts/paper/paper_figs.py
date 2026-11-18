"""Paper figures from runs/pipeline_exports/compact_metrics.json. Run from the repo root."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.figures import apply_paper_style

apply_paper_style()
IMG = Path("report/tex/img")
D = json.load(open("runs/pipeline_exports/compact_metrics.json"))
R = {e["name"]: e for e in D["pck_results"]}
BB = [("dinov2_vitb14", "DINOv2", "C0"), ("dinov3_vitb16", "DINOv3", "C1"), ("sam_vit_b", "SAM", "C2")]


def pck(name, a="0.1"):
    return R[name]["metrics"][f"pck@{a}"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# 1) Inference sensitivity on val: WSA (window x temperature) and DINO layer index.
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
ls = {0.02: "-", 0.05: "--", 0.1: ":"}
for b, lab, c in BB:
    rows = [e for e in D["pck_results_wsa_sweep"] if e["name"].startswith(b)]
    for t in sorted(ls):
        pts = sorted((e["w"], e["metrics"]["pck@0.1"]) for e in rows if abs(e["t"] - t) < 1e-9)
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], ls[t], color=c, marker="o", ms=3,
                   label=lab if t == 0.02 else None)
for t, s in ls.items():
    ax[0].plot([], [], s, color="gray", label=f"$\\tau$={t:g}")
ax[0].set(xlabel="WSA window $w$", ylabel="PCK@0.1 (val)", xticks=[3, 5, 7, 9], title="(a) WSA window / temperature")
ax[0].legend(fontsize=6, ncol=2)
for b, lab, c in BB[:2]:
    pts = sorted((e["layer"] + 1, e["metrics"]["pck@0.1"]) for e in D["pck_results_layer_sweep"] if e["name"].startswith(b))
    ax[1].plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, ms=3, label=lab)
ax[1].set(xlabel="block (1-indexed, 12 = last)", ylabel="PCK@0.1 (val)", title="(b) frozen feature depth")
ax[1].legend(fontsize=7)
save(fig, "sensitivity_combined")

# 2) Adaptation depth: FT vs LoRA, with WSA, test split.
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4), sharex=True)
for k, a in enumerate(["0.05", "0.1"]):
    for b, lab, c in BB:
        ax[k].plot([1, 2, 4], [pck(f"{b}_ft_lb{n}_wsa", a) for n in (1, 2, 4)], "-o", color=c, ms=3, label=f"{lab} FT")
        ax[k].plot([1, 2, 4], [pck(f"{b}_lora_lb{n}_wsa", a) for n in (1, 2, 4)], "--s", color=c, ms=3, label=f"{lab} LoRA")
        ax[k].axhline(pck(f"{b}_baseline_wsa", a), color=c, lw=0.8, ls=":")
    ax[k].set(xlabel="adapted blocks $N$", ylabel=f"PCK@{a} (test)", xticks=[1, 2, 4])
ax[0].set_title("(a) $\\alpha$=0.05"); ax[1].set_title("(b) $\\alpha$=0.1")
ax[1].legend(fontsize=6, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
save(fig, "depth_combined")

# 3) Difficulty factors: frozen baseline vs best model (both with WSA).
FACT = [("viewpoint_variation", "viewpoint"), ("scale_variation", "scale"), ("truncation", "truncation"), ("occlusion", "occlusion")]
fig, ax = plt.subplots(figsize=(3.4, 2.2))
wbar = 0.2
for j, (name, lab, col) in enumerate([("dinov2_vitb14_baseline_wsa", "frozen", "C7"), ("dinov2_vitb14_ft_lb2_wsa", "FT $N$=2", "C0")]):
    df = R[name]["diff"]
    easy = [df[f]["0"]["summary"]["image"]["custom_pck0.1"]["all"] for f, _ in FACT]
    hard = [df[f]["1"]["summary"]["image"]["custom_pck0.1"]["all"] for f, _ in FACT]
    x = [i + (2 * j - 1) * wbar for i in range(len(FACT))]
    ax.bar([v - wbar / 2 for v in x], easy, wbar, color=col, alpha=0.45, label=f"{lab}, flag unset")
    ax.bar([v + wbar / 2 for v in x], hard, wbar, color=col, label=f"{lab}, flag set")
ax.set_xticks(range(len(FACT)), [l for _, l in FACT], fontsize=7)
ax.set(ylabel="PCK@0.1 (test)", ylim=(0, 1))
ax.legend(fontsize=6, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
save(fig, "per_difficulty_a0")

# Print the difficulty numbers and levels used, for the text.
for name in ("dinov2_vitb14_baseline_wsa", "dinov2_vitb14_ft_lb2_wsa"):
    df = R[name]["diff"]
    print(name, {f: {k: round(v["summary"]["image"]["custom_pck0.1"]["all"], 3) for k, v in df[f].items()} for f, _ in FACT})

# 4) Per-category PCK@0.1 (supplement): frozen vs best FT, both with WSA.
import numpy as np
PC = {e["name"]: e["categories"] for e in json.load(open("runs/pipeline_exports/pck_results_per_category.json"))}
rows = [("dinov2_vitb14_baseline_wsa", "DINOv2 frozen"), ("dinov2_vitb14_ft_lb2_wsa", "DINOv2 FT $N$=2"),
        ("dinov3_vitb16_baseline_wsa", "DINOv3 frozen"), ("dinov3_vitb16_ft_lb2_wsa", "DINOv3 FT $N$=2"),
        ("sam_vit_b_baseline_wsa", "SAM frozen"), ("sam_vit_b_ft_lb4_wsa", "SAM FT $N$=4")]
cats = sorted(PC[rows[0][0]])
M = np.array([[PC[n][c]["pck@0.1"] * 100 for c in cats] for n, _ in rows])
fig, ax = plt.subplots(figsize=(7.0, 2.3))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5)
ax.set_xticks(range(len(cats)), [c.replace("pottedplant", "plant").replace("tvmonitor", "tv") for c in cats], rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=6.5)
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
save(fig, "per_category_a0")

# 5) Training curves (supplement): train/val loss per epoch for representative runs.
H = Path("runs/histories")
runs = [("dinov2_vitb14_ft_lb2", "DINOv2 FT $N$=2", "C0", "-"), ("dinov3_vitb16_ft_lb2", "DINOv3 FT $N$=2", "C1", "-"),
        ("sam_vit_b_ft_lb4", "SAM FT $N$=4", "C2", "-"), ("dinov2_vitb14_lora_r8_lb4", "DINOv2 LoRA $N$=4", "C0", "--"),
        ("dinov3_vitb16_lora_r8_lb4", "DINOv3 LoRA $N$=4", "C1", "--"), ("sam_vit_b_lora_r8_lb4", "SAM LoRA $N$=4", "C2", "--")]
fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.4))
for name, lab, c, s in runs:
    recs = [json.loads(l) for l in open(H / f"{name}_history.jsonl") if l.strip()]
    ep = [r["epoch"] + 1 for r in recs]
    ax[0].plot(ep, [r["train_loss"] for r in recs], s, color=c, marker="o", ms=2.5, label=lab)
    ax[1].plot(ep, [r["val_loss"] for r in recs], s, color=c, marker="o", ms=2.5)
ax[0].set(xlabel="epoch", ylabel="training loss", title="(a) training"); ax[1].set(xlabel="epoch", ylabel="validation loss", title="(b) validation")
ax[0].legend(fontsize=5.5, ncol=2)
save(fig, "training_curves")
