"""Write LaTeX result tables from runs/pipeline_exports/compact_metrics.json (run from the repo root)."""
import json
from pathlib import Path

D = json.load(open("runs/pipeline_exports/compact_metrics.json"))
R = {e["name"]: e["metrics"] for e in D["pck_results"]}
PF = {e["name"]: e["metrics"] for e in D.get("pck_results_pf_willow", [])}
PFP = {e["name"]: e["metrics"] for e in D.get("pck_results_pf_pascal", [])}
OUT = Path("report/tex/tab"); OUT.mkdir(exist_ok=True)
BB = [("dinov2_vitb14", "DINOv2"), ("dinov3_vitb16", "DINOv3"), ("sam_vit_b", "SAM")]
METHODS = [("baseline", "Frozen", "--", "0")] + \
          [(f"ft_lb{n}", "FT", str(n), {1: "7.1M", 2: "14.2M", 4: "28.4M"}[n]) for n in (1, 2, 4)] + \
          [(f"lora_lb{n}", "LoRA", str(n), {1: "61K", 2: "123K", 4: "246K"}[n]) for n in (1, 2, 4)]
AL = ("0.05", "0.1", "0.2")


def fmt(v, best):
    s = f"{100 * v:.1f}"
    return f"\\textbf{{{s}}}" if abs(v - best) < 1e-12 else s


def main_table(key, path, caption, label):
    rows = []
    for b, lab in BB:
        names = [f"{b}_{m}" for m, *_ in METHODS]
        best = {(w, a): max(R[n + w][f"{key}@{a}"] for n in names) for w in ("", "_wsa") for a in AL}
        for i, (m, meth, n, par) in enumerate(METHODS):
            cells = [fmt(R[f"{b}_{m}{w}"][f"{key}@{a}"], best[(w, a)]) for w in ("", "_wsa") for a in AL]
            head = f"\\multirow{{7}}{{*}}{{{lab}}}" if i == 0 else ""
            rows.append(f"{head} & {meth} & {n} & {par} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    rows[-1] = r"\bottomrule"
    body = "\n".join(rows)
    Path(path).write_text(f"""\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\small
\\begin{{tabular}}{{llcr ccc ccc}}
\\toprule
 & & & Trainable & \\multicolumn{{3}}{{c}}{{Argmax}} & \\multicolumn{{3}}{{c}}{{WSA}} \\\\
\\cmidrule(lr){{5-7}}\\cmidrule(lr){{8-10}}
Backbone & Method & $N$ & params & @0.05 & @0.1 & @0.2 & @0.05 & @0.1 & @0.2 \\\\
\\midrule
{body}
\\end{{tabular}}
\\end{{table*}}
""")


main_table("pck", OUT / "main.tex",
           "PCK (\\%) on the SPair-71k test split, averaged per image. FT: fine-tuning of the last $N$ blocks; LoRA: rank-8 adapters on the MLPs of the last $N$ blocks; WSA: window soft-argmax with $(w,\\tau)$ selected on the validation split. Bold: best per backbone and column.",
           "tab:main")
main_table("pck_pt", OUT / "main_pt.tex",
           "PCK (\\%) on the SPair-71k test split, averaged per keypoint (micro). Same runs as \\cref{tab:main}.",
           "tab:main_pt")

# Augmentation ablation (FT, WSA, per image @0.1).
rows = []
for b, lab in BB:
    for n in (1, 2, 4):
        a, na = R[f"{b}_ft_lb{n}_wsa"]["pck@0.1"], R[f"{b}_ft_lb{n}_noaug_wsa"]["pck@0.1"]
        rows.append(f"{lab} & {n} & {100*a:.1f} & {100*na:.1f} & {100*(a-na):+.1f} \\\\")
Path(OUT / "aug.tex").write_text("""\\begin{table}[t]
\\centering
\\caption{Photometric augmentation during fine-tuning: PCK@0.1 (\\%, per image, WSA) on the SPair-71k test split.}
\\label{tab:aug}
\\small
\\begin{tabular}{lcccc}
\\toprule
Backbone & $N$ & with aug. & without & $\\Delta$ \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table}
""")

# Cross-dataset (argmax, as exported).
def pf_rows(src, suffix):
    out = []
    for b, lab in BB:
        for m, meth, n, _ in METHODS:
            k = f"{b}_{m}_{suffix}"
            if k in src:
                v = src[k]
                out.append(f"{lab} & {meth} & {n} & " + " & ".join(f"{100*v[f'pck@{a}']:.1f}" for a in AL) + r" \\")
        out.append(r"\midrule")
    if out:
        out[-1] = r"\bottomrule"
    return "\n".join(out)

for src, suffix, name, thr in ((PF, "pf_willow", "PF-Willow", "the keypoint bounding box"), (PFP, "pf_pascal", "PF-Pascal", "the image size")):
    if not src:
        continue
    Path(OUT / f"{suffix}.tex").write_text(f"""\\begin{{table}}[t]
\\centering
\\caption{{{name}: PCK (\\%, per image, argmax) of SPair-71k models without retraining; threshold relative to {thr}.}}
\\label{{tab:{suffix}}}
\\small
\\begin{{tabular}}{{llcccc}}
\\toprule
Backbone & Method & $N$ & @0.05 & @0.1 & @0.2 \\\\
\\midrule
{pf_rows(src, suffix)}
\\end{{tabular}}
\\end{{table}}
""")
print(sorted(p.name for p in OUT.iterdir()))
