"""Gate 1 Directional Vulnerability Test.

Tests SAM-Med3D vulnerability with xyz-separated perturbations on multiple cases.
Key additions over the original test:
- Directional perturbation (x-only, y-only, z-only, isotropic)
- Multiple validation cases
- Focus on organs with clean Dice > 0.3 (meaningful baseline)
- Statistical summary with per-direction analysis
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/autodl-tmp/SAM-Med3D")
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")

from src.sam_med3d_gate1_wrapper import (
    require_simpleitk,
    load_model,
    read_image,
    resample_sitk,
    crop_or_pad_around_mask,
    z_normalize_like_official,
    choose_clean_point,
    infer_point,
    paste_roi,
    dice_score,
)


def perturb_directional(point, noise, shape, rng, direction="iso"):
    """Perturb point in a specific direction only."""
    if noise <= 0:
        return point.copy()
    shift = np.zeros(3, dtype=int)
    if direction == "iso":
        shift = rng.integers(-noise, noise + 1, size=3)
    elif direction == "z":
        shift[0] = rng.integers(-noise, noise + 1)
    elif direction == "y":
        shift[1] = rng.integers(-noise, noise + 1)
    elif direction == "x":
        shift[2] = rng.integers(-noise, noise + 1)
    return np.clip(point + shift, 0, np.array(shape) - 1).astype(int)


def run_case(model, image_path, label_path, args, device):
    """Run vulnerability test on a single case."""
    sitk = require_simpleitk()
    rng = np.random.default_rng(args.seed)

    image_sitk, image_orig = read_image(image_path)
    label_sitk, label_orig = read_image(label_path)

    target_spacing = [1.5, 1.5, 1.5]
    image_resampled = resample_sitk(image_sitk, target_spacing, sitk.sitkLinear,
                                     default_value=float(np.min(image_orig)))
    label_resampled = resample_sitk(label_sitk, target_spacing, sitk.sitkNearestNeighbor)
    image_arr = sitk.GetArrayFromImage(image_resampled).astype(np.float32)
    label_arr = sitk.GetArrayFromImage(label_resampled).astype(np.int16)

    organs = [int(v) for v in np.unique(label_arr) if int(v) > 0]
    rows = []

    for organ_id in organs:
        organ_label = (label_arr == organ_id).astype(np.uint8)
        voxels = int(organ_label.sum())
        if voxels < 100:
            continue

        try:
            roi_image, roi_label, crop_info = crop_or_pad_around_mask(
                image_arr, organ_label, args.crop_size)
        except ValueError:
            continue

        roi_image = z_normalize_like_official(roi_image)
        clean_point = choose_clean_point(roi_label, "center", rng)

        for direction in ["iso", "z", "y", "x"]:
            for noise in args.noise_levels:
                for trial in range(args.trials):
                    trial_rng = np.random.default_rng(
                        args.seed + organ_id * 10000 + noise * 100 + trial
                        + hash(direction) % 1000)
                    point = perturb_directional(
                        clean_point, noise, roi_label.shape, trial_rng, direction)
                    pred_roi, _ = infer_point(model, roi_image, point, device)
                    pred_full = paste_roi(pred_roi, label_arr.shape, crop_info)
                    dice = dice_score(pred_full, organ_label)
                    inside = bool(roi_label[tuple(point)] > 0)

                    rows.append({
                        "case": Path(image_path).stem,
                        "organ_id": organ_id,
                        "roi_voxels": voxels,
                        "direction": direction,
                        "noise": noise,
                        "trial": trial,
                        "dice": dice,
                        "point_inside_gt": inside,
                    })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/amos22_raw/amos22")
    parser.add_argument("--split", default="Va", choices=["Tr", "Va"])
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--output-dir", default="results/gate1_directional")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--noise-levels", type=int, nargs="+",
                        default=[0, 3, 5, 8, 10, 15, 20])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(
        "/root/autodl-tmp/SAM-Med3D",
        "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth",
        "vit_b_ori", device)

    img_dir = Path(args.data_dir) / f"images{args.split}"
    lbl_dir = Path(args.data_dir) / f"labels{args.split}"
    cases = sorted(img_dir.glob("*.nii.gz"))[:args.num_cases]

    print(f"Running Gate 1 directional on {len(cases)} cases")
    print(f"Noise levels: {args.noise_levels}, Trials: {args.trials}")
    print(f"Directions: iso, z, y, x")
    print("=" * 60)

    all_rows = []
    for i, img_path in enumerate(cases):
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            print(f"  SKIP {img_path.name}: no label")
            continue
        print(f"  [{i+1}/{len(cases)}] {img_path.name}...", end=" ", flush=True)
        try:
            rows = run_case(model, str(img_path), str(lbl_path), args, device)
            all_rows.extend(rows)
            n_organs = len(set(r["organ_id"] for r in rows))
            print(f"{n_organs} organs, {len(rows)} measurements")
        except Exception as e:
            print(f"ERROR: {e}")

    if not all_rows:
        print("No results!")
        return

    csv_path = output_dir / "gate1_directional_rows.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary: filter to organs with clean Dice > 0.3
    print("\n" + "=" * 60)
    print("GATE 1 DIRECTIONAL VULNERABILITY SUMMARY")
    print("=" * 60)

    organ_clean = defaultdict(list)
    for r in all_rows:
        if r["noise"] == 0 and r["direction"] == "iso":
            organ_clean[(r["case"], r["organ_id"])].append(r["dice"])

    valid_organs = {k for k, v in organ_clean.items() if np.mean(v) > 0.3}
    valid_rows = [r for r in all_rows
                  if (r["case"], r["organ_id"]) in valid_organs]

    print(f"\nTotal organs tested: {len(organ_clean)}")
    print(f"Organs with clean Dice > 0.3: {len(valid_organs)}")

    print(f"\n{'Direction':>10} {'Noise':>6} {'Mean Dice':>10} {'Drop':>8} {'N':>5}")
    print("-" * 45)

    summary = {}
    for direction in ["iso", "z", "y", "x"]:
        summary[direction] = {}
        clean_vals = [r["dice"] for r in valid_rows
                      if r["direction"] == direction and r["noise"] == 0]
        clean_mean = np.mean(clean_vals) if clean_vals else 0

        for noise in args.noise_levels:
            vals = [r["dice"] for r in valid_rows
                    if r["direction"] == direction and r["noise"] == noise]
            if not vals:
                continue
            mean_val = np.mean(vals)
            drop = (clean_mean - mean_val) * 100
            summary[direction][noise] = {
                "mean_dice": float(mean_val),
                "drop_pts": float(drop),
                "n": len(vals),
            }
            print(f"{direction:>10} {noise:>6} {mean_val:>10.4f} "
                  f"{drop:>+8.1f} {len(vals):>5}")

    # Gate decision
    iso_drop10 = summary.get("iso", {}).get(10, {}).get("drop_pts", 0)
    iso_drop15 = summary.get("iso", {}).get(15, {}).get("drop_pts", 0)
    z_drop10 = summary.get("z", {}).get(10, {}).get("drop_pts", 0)

    print(f"\n{'='*60}")
    print("GATE 1 DECISION METRICS (organs with clean Dice > 0.3 only):")
    print(f"  Isotropic shift-10 drop: {iso_drop10:+.1f} pts")
    print(f"  Isotropic shift-15 drop: {iso_drop15:+.1f} pts")
    print(f"  Z-only shift-10 drop:    {z_drop10:+.1f} pts")

    max_drop = max(iso_drop10, iso_drop15, z_drop10)
    if max_drop >= 8:
        status = "GREEN"
    elif max_drop >= 5:
        status = "YELLOW"
    else:
        status = "RED"

    print(f"  Max drop in critical range: {max_drop:+.1f} pts")
    print(f"  GATE 1 STATUS: {status}")

    print(f"\n  Anisotropy (shift-10): "
          f"z={summary.get('z',{}).get(10,{}).get('drop_pts',0):+.1f}  "
          f"y={summary.get('y',{}).get(10,{}).get('drop_pts',0):+.1f}  "
          f"x={summary.get('x',{}).get(10,{}).get('drop_pts',0):+.1f}")

    result = {
        "num_cases": len(cases),
        "num_valid_organs": len(valid_organs),
        "summary_by_direction": summary,
        "gate1_status": status,
        "max_drop_critical_range": float(max_drop),
        "anisotropy_shift10": {
            "z": summary.get("z", {}).get(10, {}).get("drop_pts", 0),
            "y": summary.get("y", {}).get(10, {}).get("drop_pts", 0),
            "x": summary.get("x", {}).get(10, {}).get("drop_pts", 0),
        },
    }
    json_path = output_dir / "gate1_directional_summary.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
