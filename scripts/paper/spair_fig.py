"""SPair-71k example figure (raw test pairs with GT keypoint matches). Run from the repo root (CPU)."""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image

ROOT = Path("data/SPair-71k")
OUT = Path("runs/paper_figs"); OUT.mkdir(parents=True, exist_ok=True)
pairs = [l.strip() for l in open(ROOT / "Layout/large/test.txt") if l.strip()]
random.seed(0)
want = {"cat": None, "aeroplane": None, "motorbike": None, "bottle": None}
for p in random.sample(pairs, len(pairs)):
    a = json.load(open(ROOT / "PairAnnotation/test" / f"{p}.json"))
    c = a["category"]
    if c in want and want[c] is None and len(a["src_kps"]) >= 5 and (a["viewpoint_variation"] > 0 or a["scale_variation"] > 0):
        want[c] = a
    if all(want.values()):
        break

fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
for col, (c, a) in enumerate(want.items()):
    ims = [Image.open(ROOT / "JPEGImages" / c / a[k]) for k in ("src_imname", "trg_imname")]
    for r in range(2):
        axes[r, col].imshow(ims[r]); axes[r, col].axis("off")
    cols = plt.cm.tab10.colors
    for i, (s, t) in enumerate(zip(a["src_kps"], a["trg_kps"])):
        col_i = cols[i % 10]
        axes[0, col].plot(*s, "o", ms=5, mec="white", mfc=col_i, mew=0.8)
        axes[1, col].plot(*t, "o", ms=5, mec="white", mfc=col_i, mew=0.8)
        fig.add_artist(ConnectionPatch(xyA=s, xyB=t, coordsA="data", coordsB="data", axesA=axes[0, col],
                                       axesB=axes[1, col], color=col_i, lw=0.6, alpha=0.7))
    flags = f"vp={a['viewpoint_variation']} sc={a['scale_variation']} tr={a['truncation']} oc={a['occlusion']}"
    axes[0, col].set_title(f"{c}\n{flags}", fontsize=8)
    print(a["filename"] if "filename" in a else (a["src_imname"], a["trg_imname"]), flags)
fig.tight_layout()
fig.savefig(OUT / "spair_examples.pdf", bbox_inches="tight", dpi=200)
fig.savefig(OUT / "spair_examples.png", bbox_inches="tight", dpi=110)
