"""Collapse the large pipeline exports into runs/pipeline_exports/compact_metrics.json (run from the repo root).

The paper tables and figures (scripts/paper/paper_tables.py, paper_figs.py) read only this file.
"""
import json
from pathlib import Path

EXP = Path("runs/pipeline_exports")
out = {}
for f in ("pck_results_layer_sweep", "pck_results_wsa_sweep", "pck_results_pf_willow", "pck_results_pf_pascal", "pck_results"):
    p = EXP / f"{f}.json"
    if not p.is_file():
        continue
    out[f] = [{
        "name": e["name"], "split": e["spec"].get("split"), "wsa": e["spec"].get("use_window_soft_argmax"),
        "w": e["spec"].get("wsa_window"), "t": e["spec"].get("wsa_temperature"), "layer": e["spec"].get("dino_layer"),
        "metrics": e["metrics"], "thr": e.get("throughput_img_per_sec"), "mem": e.get("peak_memory_mb"),
        "diff": e.get("sd4match_by_difficulty_flag") if f == "pck_results" else None,
    } for e in json.load(open(p))]
(EXP / "compact_metrics.json").write_text(json.dumps(out))
print({k: len(v) for k, v in out.items()})
