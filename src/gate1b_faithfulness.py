"""Gate 1B: Prompt Faithfulness Diagnostic.

For each case-organ pair, sample prompts at different semantic locations
and measure whether the model's output is causally controlled by prompt location.

Key metrics:
- Semantic Switch Accuracy: does prompt in organ k produce mask best-matching organ k?
- Target Margin: Dice(pred, prompted_organ) - max(Dice(pred, other_organs))
- Cross-prompt Mask IoU: do different prompts produce different masks?
- Prompt Grounding Rate: is the prompt point inside the predicted mask?
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Gate 1B: Prompt Faithfulness Diagnostic")
    parser.add_argument("--sam-root", default="/root/autodl-tmp/SAM-Med3D")
    parser.add_argument("--checkpoint", default="/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth")
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/amos22_raw/amos22")
    parser.add_argument("--split", default="Va", choices=["Tr", "Va"])
    parser.add_argument("--output-dir", default="results/gate1b_faithfulness")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--samples-per-region", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--target-spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model_and_deps(args):
    sys.path.insert(0, args.sam_root)
    sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")
    from src.sam_med3d_gate1_wrapper import (
        load_model, require_simpleitk, read_image,
        resample_sitk, crop_or_pad_around_mask, z_normalize_like_official,
        infer_point, dice_score
    )
    device = torch.device(args.device)
    model = load_model(args.sam_root, args.checkpoint, "vit_b_ori", device)
    return model, device


def sample_points_in_mask(mask, n, rng, mode="random"):
    """Sample n points inside a binary mask."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return []
    if mode == "center":
        center = coords.mean(axis=0)
        idx = np.argmin(np.sum((coords - center) ** 2, axis=1))
        return [coords[idx].astype(int)]
    indices = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
    return [coords[i].astype(int) for i in indices]


def sample_background_points(label_arr, n, rng):
    """Sample points in background (label == 0), far from any organ."""
    bg_mask = (label_arr == 0)
    coords = np.argwhere(bg_mask)
    if len(coords) == 0:
        return []
    indices = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
    return [coords[i].astype(int) for i in indices]


def run_diagnostic(args):
    import SimpleITK as sitk
    sys.path.insert(0, args.sam_root)
    sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")
    from src.sam_med3d_gate1_wrapper import (
        load_model, resample_sitk, crop_or_pad_around_mask,
        z_normalize_like_official, infer_point, dice_score
    )

    device = torch.device(args.device)
    model = load_model(args.sam_root, args.checkpoint, "vit_b_ori", device)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.data_dir) / f"images{args.split}"
    lbl_dir = Path(args.data_dir) / f"labels{args.split}"
    cases = sorted(img_dir.glob("*.nii.gz"))[:args.num_cases]

    all_rows = []
    confusion_data = []  # For confusion matrix

    for ci, img_path in enumerate(cases):
        lbl_path = lbl_dir / img_path.name
        case_name = img_path.stem.replace(".nii", "")
        print(f"[{ci+1}/{len(cases)}] {case_name}")

        # Load and resample
        image_sitk = sitk.ReadImage(str(img_path))
        label_sitk = sitk.ReadImage(str(lbl_path))
        spacing_xyz = tuple(args.target_spacing)

        image_res = resample_sitk(image_sitk, spacing_xyz, sitk.sitkLinear,
                                  float(sitk.GetArrayFromImage(image_sitk).min()))
        label_res = resample_sitk(label_sitk, spacing_xyz, sitk.sitkNearestNeighbor)
        image_arr = sitk.GetArrayFromImage(image_res).astype(np.float32)
        label_arr = sitk.GetArrayFromImage(label_res).astype(np.int16)

        organs = [int(v) for v in np.unique(label_arr) if v > 0]
        organ_masks = {oid: (label_arr == oid).astype(np.uint8) for oid in organs}
        valid_organs = [oid for oid in organs if organ_masks[oid].sum() >= 200]

        if len(valid_organs) < 2:
            print(f"  Skipping: only {len(valid_organs)} valid organs")
            continue

        # For each valid organ as the "crop target" (determines the ROI)
        for target_oid in valid_organs:
            target_mask = organ_masks[target_oid]
            roi_image, roi_label, crop_info = crop_or_pad_around_mask(
                image_arr, target_mask, args.crop_size)
            roi_image = z_normalize_like_official(roi_image)

            # Get all organ masks in this ROI
            roi_organ_masks = {}
            for oid in valid_organs:
                roi_mask = np.zeros((args.crop_size,)*3, dtype=np.uint8)
                src_sl = tuple(slice(int(s), int(e))
                               for s, e in zip(crop_info.src_start, crop_info.src_end))
                dst_sl = tuple(slice(int(s), int(e))
                               for s, e in zip(crop_info.dst_start, crop_info.dst_end))
                full_mask = organ_masks[oid]
                roi_mask[dst_sl] = full_mask[src_sl]
                if roi_mask.sum() >= 50:
                    roi_organ_masks[oid] = roi_mask

            if target_oid not in roi_organ_masks:
                continue

            # Sample prompts from different regions
            prompt_sources = {}

            # 1. Target organ (inside)
            pts = sample_points_in_mask(roi_organ_masks[target_oid],
                                        args.samples_per_region, rng)
            prompt_sources[f"target_{target_oid}"] = (pts, target_oid)

            # 2. Neighboring organs
            for other_oid in roi_organ_masks:
                if other_oid == target_oid:
                    continue
                pts = sample_points_in_mask(roi_organ_masks[other_oid],
                                            args.samples_per_region, rng)
                prompt_sources[f"other_{other_oid}"] = (pts, other_oid)

            # 3. Background in ROI
            roi_bg = ((roi_label == 0) & (roi_image != 0)).astype(np.uint8)
            if roi_bg.sum() > 100:
                pts = sample_points_in_mask(roi_bg, args.samples_per_region, rng)
                prompt_sources["background"] = (pts, 0)

            # Run inference for each prompt and compute metrics
            predictions = {}  # source_key -> list of pred masks
            for source_key, (points, prompted_oid) in prompt_sources.items():
                for pi, point in enumerate(points):
                    pred_roi, _ = infer_point(model, roi_image, point, device)

                    # Compute Dice to ALL organs in ROI
                    dices_to_organs = {}
                    for oid, omask in roi_organ_masks.items():
                        dices_to_organs[oid] = dice_score(pred_roi, omask)

                    # Best matching organ
                    if dices_to_organs:
                        best_organ = max(dices_to_organs, key=dices_to_organs.get)
                        best_dice = dices_to_organs[best_organ]
                    else:
                        best_organ = -1
                        best_dice = 0.0

                    # Target margin
                    target_dice = dices_to_organs.get(prompted_oid, 0.0)
                    other_dices = [d for oid, d in dices_to_organs.items()
                                   if oid != prompted_oid]
                    max_other = max(other_dices) if other_dices else 0.0
                    target_margin = target_dice - max_other

                    # Grounding: is prompt point inside prediction?
                    point_inside_pred = bool(pred_roi[tuple(point)] > 0)

                    # Predicted mask volume
                    pred_volume = int(pred_roi.sum())

                    row = {
                        "case": case_name,
                        "crop_target_organ": target_oid,
                        "prompt_source": source_key,
                        "prompted_organ": prompted_oid,
                        "point_idx": pi,
                        "point_zyx": point.tolist(),
                        "best_match_organ": best_organ,
                        "best_match_dice": best_dice,
                        "target_dice": target_dice,
                        "target_margin": target_margin,
                        "point_inside_pred": point_inside_pred,
                        "pred_volume": pred_volume,
                        "switch_correct": (best_organ == prompted_oid),
                    }
                    # Add per-organ dices
                    for oid, d in dices_to_organs.items():
                        row[f"dice_organ_{oid}"] = d

                    all_rows.append(row)

                    # Store prediction for cross-prompt IoU
                    predictions.setdefault(source_key, []).append(pred_roi)

            # Compute cross-prompt mask IoU (between target and other organs)
            target_key = f"target_{target_oid}"
            if target_key in predictions and predictions[target_key]:
                target_pred = predictions[target_key][0]  # first prediction
                for source_key, preds in predictions.items():
                    if source_key == target_key:
                        continue
                    for pred in preds:
                        inter = np.logical_and(target_pred > 0, pred > 0).sum()
                        union = np.logical_or(target_pred > 0, pred > 0).sum()
                        iou = float(inter / (union + 1e-6))
                        confusion_data.append({
                            "case": case_name,
                            "crop_organ": target_oid,
                            "source_a": target_key,
                            "source_b": source_key,
                            "cross_mask_iou": iou,
                        })

    # Save results
    if all_rows:
        csv_path = output_dir / "faithfulness_rows.csv"
        keys = list(all_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    if confusion_data:
        csv_path2 = output_dir / "cross_prompt_iou.csv"
        with csv_path2.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(confusion_data[0].keys()))
            writer.writeheader()
            writer.writerows(confusion_data)

    # Compute summary statistics
    print("\n" + "=" * 60)
    print("PROMPT FAITHFULNESS DIAGNOSTIC SUMMARY")
    print("=" * 60)

    # Switch accuracy (only for prompts inside labeled organs, not background)
    organ_prompts = [r for r in all_rows if r["prompted_organ"] > 0]
    if organ_prompts:
        correct = sum(1 for r in organ_prompts if r["switch_correct"])
        total = len(organ_prompts)
        switch_acc = 100.0 * correct / total
        print(f"\nSemantic Switch Accuracy: {correct}/{total} = {switch_acc:.1f}%")

        # Target margin
        margins = [r["target_margin"] for r in organ_prompts]
        print(f"Target Margin: mean={np.mean(margins):.4f}, "
              f"median={np.median(margins):.4f}, "
              f"% positive={100*np.mean([m>0 for m in margins]):.1f}%")

        # Grounding rate
        grounded = sum(1 for r in organ_prompts if r["point_inside_pred"])
        print(f"Prompt Grounding Rate: {grounded}/{total} = {100*grounded/total:.1f}%")

    # Cross-prompt IoU
    if confusion_data:
        ious = [r["cross_mask_iou"] for r in confusion_data]
        print(f"\nCross-prompt Mask IoU (target vs other): "
              f"mean={np.mean(ious):.4f}, median={np.median(ious):.4f}")
        high_iou = sum(1 for x in ious if x > 0.5)
        print(f"  High IoU (>0.5): {high_iou}/{len(ious)} = "
              f"{100*high_iou/len(ious):.1f}% (prompt insensitivity indicator)")

    # Per-organ switch accuracy
    print("\nPer-organ Switch Accuracy:")
    from collections import defaultdict
    organ_correct = defaultdict(lambda: [0, 0])
    for r in organ_prompts:
        oid = r["prompted_organ"]
        organ_correct[oid][1] += 1
        if r["switch_correct"]:
            organ_correct[oid][0] += 1
    for oid in sorted(organ_correct):
        c, t = organ_correct[oid]
        print(f"  organ {oid:2d}: {c}/{t} = {100*c/t:.1f}%")

    # Decision
    print("\n" + "=" * 60)
    if organ_prompts:
        if switch_acc < 40:
            print("VERDICT: STRONG prompt faithfulness failure. PROCEED with pivot.")
        elif switch_acc < 70:
            print("VERDICT: MODERATE prompt faithfulness issue. Pivot viable.")
        else:
            print("VERDICT: Model IS prompt-faithful. Pivot to controllability is WEAK.")
    print("=" * 60)

    # Save summary
    summary = {
        "num_cases": len(cases),
        "num_rows": len(all_rows),
        "switch_accuracy": switch_acc if organ_prompts else None,
        "mean_target_margin": float(np.mean(margins)) if organ_prompts else None,
        "grounding_rate": 100 * grounded / total if organ_prompts else None,
        "mean_cross_prompt_iou": float(np.mean(ious)) if confusion_data else None,
    }
    with (output_dir / "faithfulness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
