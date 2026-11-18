"""Qualitative figure for the paper: DINOv2 baseline vs FT lb2 + WSA on SPair-71k test pairs.

Run from the repo root on a GPU node. Pairs are picked from the exported per-image PCK@0.1:
largest gain of FT lb2+WSA over the frozen baseline, one pair per category.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from data.dataset import PreprocessMode, SPair71kPairDataset
from evaluation.qualitative import MethodSpec, load_method_extractors, render_method_comparison_grid

OUT = Path("runs/paper_figs"); OUT.mkdir(parents=True, exist_ok=True)
CK = Path("checkpoints")
HW = (518, 518)
W, T = 7, 0.05  # WSA setting selected on val for DINOv2 (wsa_selection.json)

res = {e["name"]: e for e in json.load(open("runs/pipeline_exports/pck_results.json"))}
base = res["dinov2_vitb14_baseline"]["sd4match_per_image"]["custom_pck0.1"]["all"]
ft = res["dinov2_vitb14_ft_lb2_wsa"]["sd4match_per_image"]["custom_pck0.1"]["all"]
ds = SPair71kPairDataset(spair_root="data/SPair-71k", split="test", preprocess=PreprocessMode.FIXED_RESIZE,
                         output_size_hw=HW, normalize=True)
assert len(ds) == len(base) == len(ft), (len(ds), len(base), len(ft))

order = sorted(range(len(ds)), key=lambda i: ft[i] - base[i], reverse=True)
picked, cats = [], set()
for i in order:
    s = ds[i]
    if s["category"] in cats or int(s["n_valid_keypoints"]) < 5:
        continue
    cats.add(s["category"]); picked.append(i)
    if len(picked) == 3:
        break

specs = [MethodSpec("dinov2_vitb14", "baseline"), MethodSpec("dinov2_vitb14", "ft_lb2", wsa=True)]
ext = load_method_extractors(specs, ckpt_dir=CK, pretrained_paths={"dinov2_vitb14": str(CK / "dinov2_vitb14_pretrain.pth")},
                             device=torch.device("cuda"))
for i in picked:
    s = ds[i]
    print(f"pair {i} {s['pair_id_str']} cat={s['category']} stored PCK@0.1 base={base[i]:.3f} ft_wsa={ft[i]:.3f}")
    fig = render_method_comparison_grid(s, ext, img_hw=HW, use_wsa_per_label={sp.label: sp.wsa for sp in specs}, wsa_window=W, wsa_temperature=T, alpha=0.1)
    fig.savefig(OUT / f"qual_{i}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"qual_{i}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
