"""
Re-evaluate LoRA configurations with the fixed checkpoint loader.

This script is a one-shot fix: the original eval_and_export stage silently
skipped LoRA weights because checkpoint_loader.py loaded the state dict into
a vanilla (non-LoRA) encoder.  After applying the fix to checkpoint_loader.py,
run this script to regenerate correct PCK numbers for all LoRA runs and
overwrite the relevant pipeline exports.

Usage (from repo root):
    PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/reeval_lora.py

The script writes to:
    runs/pipeline_exports/pck_results.json
    runs/pipeline_exports/pck_results.csv
    runs/pipeline_exports/pck_results_per_category.json
    runs/pipeline_exports/pck_results_by_difficulty_flag.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from evaluation.experiment_runner import EvalRunSpec, run_spair_pck_eval

CKPTS = REPO / "checkpoints"
EXPORTS = REPO / "runs" / "pipeline_exports"

# ---------------------------------------------------------------------------
# Locate pretrained weights
# ---------------------------------------------------------------------------
DINOV2_PRETRAIN = str(CKPTS / "dinov2_vitb14_pretrain.pth")
# Accept either the versioned or un-versioned dinov3 filename
_dv3_candidates = list(CKPTS.glob("dinov3_vitb16_pretrain*.pth"))
DINOV3_PRETRAIN = str(_dv3_candidates[0]) if _dv3_candidates else ""
SAM_PRETRAIN = str(CKPTS / "sam_vit_b_01ec64.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Build LoRA EvalRunSpecs
# ---------------------------------------------------------------------------
_BACKBONES = [
    (
        "dinov2_vitb14",
        {"dinov2_weights": DINOV2_PRETRAIN, "dinov3_weights": None, "sam_checkpoint": None},
        518, 518,
    ),
    (
        "dinov3_vitb16",
        {"dinov2_weights": None, "dinov3_weights": DINOV3_PRETRAIN, "sam_checkpoint": None},
        512, 512,
    ),
    (
        "sam_vit_b",
        {"dinov2_weights": None, "dinov3_weights": None, "sam_checkpoint": SAM_PRETRAIN},
        512, 512,
    ),
]
_LORA_NBS = [1, 2, 4]
# Batch sizes: SAM encoder is large — keep bs=1 on CUDA to avoid OOM during inference.
_BS = 4 if DEVICE.type in ("mps", "cpu") else 8
_SAM_BS = 1  # SAM OOMs at bs>=2 on 16 GB VRAM; use 1 unconditionally
# MPS/CPU: num_workers=0 avoids DataLoader subprocess spawn issues on macOS.
# The subprocess tries to re-run module-level code, which errors on multiprocessing.
_NW = 0 if DEVICE.type in ("mps", "cpu") else 4

specs_to_run: list[EvalRunSpec] = []

for backbone, weights, h, w in _BACKBONES:
    bs = _SAM_BS if backbone.startswith("sam") else _BS
    for nb in _LORA_NBS:
        ck_path = CKPTS / f"{backbone}_lora_r8_lb{nb}_best.pt"
        if not ck_path.exists():
            print(f"  WARNING: checkpoint not found, skipping: {ck_path.name}")
            continue
        base = dict(
            backbone=backbone,
            split="test",
            height=h,
            width=w,
            num_workers=_NW,
            batch_size=bs,
            dino_layer_indices=4,
            checkpoint=str(ck_path),
            **weights,
        )
        specs_to_run.append(EvalRunSpec(name=f"{backbone}_lora_lb{nb}", **base))
        specs_to_run.append(EvalRunSpec(
            name=f"{backbone}_lora_lb{nb}_wsa",
            use_window_soft_argmax=True,
            wsa_window=5,
            wsa_temperature=1.0,
            **base,
        ))

print(f"Running {len(specs_to_run)} LoRA evaluation specs.")

# ---------------------------------------------------------------------------
# Per-spec checkpoint (resume support)
# Results are written to a JSONL file after each spec so a crash does not
# lose completed work.  On re-run, already-done specs are skipped.
# ---------------------------------------------------------------------------
CKPT_JSONL = REPO / "runs" / "reeval_lora_checkpoint.jsonl"
CKPT_JSONL.parent.mkdir(parents=True, exist_ok=True)

done_map: dict[str, dict] = {}
if CKPT_JSONL.exists():
    for line in CKPT_JSONL.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entry = json.loads(line)
                done_map[entry["name"]] = entry
            except Exception:
                pass
    if done_map:
        print(f"  Resuming: {len(done_map)} spec(s) already done — skipping.")

# ---------------------------------------------------------------------------
# Run evaluations
# ---------------------------------------------------------------------------
new_results: list[dict] = list(done_map.values())
for spec in specs_to_run:
    if spec.name in done_map:
        pck01 = done_map[spec.name]["metrics"].get("pck@0.1", float("nan"))
        print(f"  [skip] {spec.name}  (checkpoint: PCK@0.1 = {pck01:.4f})", flush=True)
        continue
    print(f"  [{len(new_results)+1}/{len(specs_to_run)}] {spec.name} ...", flush=True)
    r = run_spair_pck_eval(spec, device=DEVICE)
    pck01 = r["metrics"].get("pck@0.1", float("nan"))
    print(f"    PCK@0.1 = {pck01:.4f}  (load_info: {r.get('checkpoint_load')})")
    new_results.append(r)
    # Write checkpoint immediately so progress survives a crash
    with open(CKPT_JSONL, "a") as _f:
        _f.write(json.dumps(r, default=str) + "\n")

# ---------------------------------------------------------------------------
# Merge into existing pck_results.json
# ---------------------------------------------------------------------------
print("\nUpdating pipeline exports ...")
pck_results_path = EXPORTS / "pck_results.json"
with open(pck_results_path) as f:
    all_results: list[dict] = json.load(f)

new_result_map = {r["name"]: r for r in new_results}
updated_results: list[dict] = []
replaced_names: set[str] = set()
for entry in all_results:
    name = entry.get("name", "")
    if name in new_result_map:
        updated_results.append(new_result_map[name])
        replaced_names.add(name)
    else:
        updated_results.append(entry)
# Append any new entries not previously present
for name, r in new_result_map.items():
    if name not in replaced_names:
        updated_results.append(r)

print(f"  Replaced {len(replaced_names)} LoRA entries in pck_results.json")
with open(pck_results_path, "w") as f:
    json.dump(updated_results, f)
print(f"  Saved {pck_results_path}")

# ---------------------------------------------------------------------------
# Regenerate pck_results.csv
# ---------------------------------------------------------------------------
csv_rows: list[dict] = []
for r in updated_results:
    spec_d = r.get("spec", {})
    m = r.get("metrics", {})
    csv_rows.append({
        "name": r.get("name", ""),
        "split": spec_d.get("split", ""),
        "wsa": spec_d.get("use_window_soft_argmax", False),
        "pck@0.05": m.get("pck@0.05", float("nan")),
        "pck_pt@0.05": m.get("pck_pt@0.05", float("nan")),
        "pck@0.1": m.get("pck@0.1", float("nan")),
        "pck_pt@0.1": m.get("pck_pt@0.1", float("nan")),
        "pck@0.2": m.get("pck@0.2", float("nan")),
        "pck_pt@0.2": m.get("pck_pt@0.2", float("nan")),
    })

csv_path = EXPORTS / "pck_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"  Saved {csv_path}")

# ---------------------------------------------------------------------------
# Update pck_results_per_category.json
# ---------------------------------------------------------------------------
per_cat_path = EXPORTS / "pck_results_per_category.json"
with open(per_cat_path) as f:
    per_cat_all: list[dict] = json.load(f)

def _build_per_cat_entry(result: dict) -> dict:
    """Convert sd4match_per_image per-category data to per_category.json format."""
    per_image = result.get("sd4match_per_image", {})
    alphas_present = [0.05, 0.1, 0.2]
    categories: dict[str, dict[str, float]] = {}
    for a in alphas_present:
        key = f"custom_pck{a}"
        if key not in per_image:
            continue
        for cat, vals in per_image[key].items():
            if cat == "all":
                continue
            if cat not in categories:
                categories[cat] = {}
            if vals:
                categories[cat][f"pck@{a:g}"] = float(sum(vals) / len(vals))
    # Add micro (per_point) data
    per_point = result.get("sd4match_per_point", {})
    for a in alphas_present:
        key = f"custom_pck{a}"
        if key not in per_point:
            continue
        for cat, vals in per_point[key].items():
            if cat == "all":
                continue
            if cat not in categories:
                categories[cat] = {}
            if vals:
                categories[cat][f"pck_pt@{a:g}"] = float(sum(vals) / len(vals))
    return {"name": result["name"], "categories": categories}

new_per_cat_map = {r["name"]: _build_per_cat_entry(r) for r in new_results}
updated_per_cat: list[dict] = []
replaced_cat: set[str] = set()
for entry in per_cat_all:
    name = entry.get("name", "")
    if name in new_per_cat_map:
        updated_per_cat.append(new_per_cat_map[name])
        replaced_cat.add(name)
    else:
        updated_per_cat.append(entry)
for name, e in new_per_cat_map.items():
    if name not in replaced_cat:
        updated_per_cat.append(e)

print(f"  Replaced {len(replaced_cat)} entries in pck_results_per_category.json")
with open(per_cat_path, "w") as f:
    json.dump(updated_per_cat, f)
print(f"  Saved {per_cat_path}")

# ---------------------------------------------------------------------------
# Update pck_results_by_difficulty_flag.json
# ---------------------------------------------------------------------------
diff_path = EXPORTS / "pck_results_by_difficulty_flag.json"
with open(diff_path) as f:
    diff_all: list[dict] = json.load(f)

def _build_diff_entry(result: dict) -> dict:
    spec_d = result.get("spec", {})
    return {
        "name": result["name"],
        "split": spec_d.get("split", "test"),
        "data": result.get("sd4match_by_difficulty_flag", {}),
    }

new_diff_map = {r["name"]: _build_diff_entry(r) for r in new_results}
updated_diff: list[dict] = []
replaced_diff: set[str] = set()
for entry in diff_all:
    name = entry.get("name", "")
    if name in new_diff_map:
        updated_diff.append(new_diff_map[name])
        replaced_diff.add(name)
    else:
        updated_diff.append(entry)
for name, e in new_diff_map.items():
    if name not in replaced_diff:
        updated_diff.append(e)

print(f"  Replaced {len(replaced_diff)} entries in pck_results_by_difficulty_flag.json")
with open(diff_path, "w") as f:
    json.dump(updated_diff, f)
print(f"  Saved {diff_path}")

print("\nAll LoRA evaluation results updated successfully.")
print("Run AML_Analysis.ipynb next to regenerate all figures and tables.")
