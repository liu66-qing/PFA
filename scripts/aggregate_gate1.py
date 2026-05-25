"""Aggregate multi-case Gate 1 results into a single verdict."""
import csv
import json
import sys
from pathlib import Path
import numpy as np

results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/vulnerability_multi")

all_rows = []
case_summaries = []

for case_dir in sorted(results_dir.iterdir()):
    csv_path = case_dir / "sam_med3d_gate1_rows.csv"
    if not csv_path.exists():
        continue
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            r["dice"] = float(r["dice"])
            r["noise"] = int(r["noise"])
            r["organ_id"] = int(r["organ_id"])
            r["case"] = case_dir.name
            rows.append(r)
            all_rows.append(r)
    
    # Per-case summary
    clean = np.mean([r["dice"] for r in rows if r["noise"] == 0])
    s10 = np.mean([r["dice"] for r in rows if r["noise"] == 10])
    s20 = np.mean([r["dice"] for r in rows if r["noise"] == 20])
    case_summaries.append({
        "case": case_dir.name,
        "n_organs": len(set(r["organ_id"] for r in rows)),
        "clean": clean,
        "shift10": s10,
        "shift20": s20,
        "drop10": (clean - s10) * 100,
        "drop20": (clean - s20) * 100,
    })

if not all_rows:
    print("No results found.")
    sys.exit(1)

print(f"Cases: {len(case_summaries)}")
print(f"Total organ-noise-trial rows: {len(all_rows)}")
print()

# Overall
for noise in [0, 5, 10, 15, 20]:
    vals = [r["dice"] for r in all_rows if r["noise"] == noise]
    if vals:
        clean_mean = np.mean([r["dice"] for r in all_rows if r["noise"] == 0])
        mean = np.mean(vals)
        std = np.std(vals)
        drop = (clean_mean - mean) * 100
        print(f"  Shift {noise:2d}: {mean:.4f} +/- {std:.4f} (drop {drop:+.1f}pts, n={len(vals)})")

# Functional organs only (clean > 0.25 per organ per case)
print("\n--- Functional organs (per-organ clean > 0.25) ---")
functional_rows = []
for case in set(r["case"] for r in all_rows):
    for oid in set(r["organ_id"] for r in all_rows if r["case"] == case):
        organ_clean = np.mean([r["dice"] for r in all_rows 
                              if r["case"] == case and r["organ_id"] == oid and r["noise"] == 0])
        if organ_clean > 0.25:
            functional_rows.extend([r for r in all_rows if r["case"] == case and r["organ_id"] == oid])

if functional_rows:
    for noise in [0, 5, 10, 15, 20]:
        vals = [r["dice"] for r in functional_rows if r["noise"] == noise]
        if vals:
            clean_mean = np.mean([r["dice"] for r in functional_rows if r["noise"] == 0])
            mean = np.mean(vals)
            drop = (clean_mean - mean) * 100
            print(f"  Shift {noise:2d}: {mean:.4f} (drop {drop:+.1f}pts, n={len(vals)})")

# Gate decision
clean_all = np.mean([r["dice"] for r in all_rows if r["noise"] == 0])
s10_all = np.mean([r["dice"] for r in all_rows if r["noise"] == 10])
s20_all = np.mean([r["dice"] for r in all_rows if r["noise"] == 20])
drop10 = (clean_all - s10_all) * 100
drop20 = (clean_all - s20_all) * 100

func_clean = np.mean([r["dice"] for r in functional_rows if r["noise"] == 0]) if functional_rows else 0
func_s10 = np.mean([r["dice"] for r in functional_rows if r["noise"] == 10]) if functional_rows else 0
func_drop10 = (func_clean - func_s10) * 100

print(f"\n=== GATE 1 VERDICT ===")
print(f"All organs: clean={clean_all:.4f}, s10={s10_all:.4f}, drop10={drop10:.1f}pts, drop20={drop20:.1f}pts")
if functional_rows:
    print(f"Functional: clean={func_clean:.4f}, s10={func_s10:.4f}, drop10={func_drop10:.1f}pts")

if func_drop10 >= 8 or drop20 >= 8:
    print("STATUS: GREEN (vulnerability confirmed)")
elif func_drop10 >= 4 or drop20 >= 5:
    print("STATUS: YELLOW (partial vulnerability, proceed with caution)")
else:
    print("STATUS: RED (insufficient vulnerability)")

# Save aggregate
agg = {
    "n_cases": len(case_summaries),
    "n_rows": len(all_rows),
    "all_organs": {"clean": float(clean_all), "s10": float(s10_all), "s20": float(s20_all), "drop10": float(drop10), "drop20": float(drop20)},
    "case_summaries": case_summaries,
}
out_path = results_dir / "gate1_aggregate.json"
with out_path.open("w") as f:
    json.dump(agg, f, indent=2, default=float)
print(f"\nSaved: {out_path}")
