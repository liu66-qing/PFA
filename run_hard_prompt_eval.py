#!/usr/bin/env python3
"""
Hard Prompt Evaluation Benchmark
=================================
Three difficulty levels beyond the standard interior-point evaluation:
1. Boundary clicks: points within 1-3 voxels of organ boundary
2. Noisy clicks: standard points + uniform noise ±5 voxels
3. Adjacent-organ-only switch: only test switch between neighboring organs

Evaluates both seg-only baseline and PFA (ours) to show robustness.
"""
import sys
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg/src")
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import Dataset, DataLoader

from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point

DEVICE = "cuda:1"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"
SAVE_DIR = "/root/autodl-tmp/PEA-MedSeg/results/formal/hard_prompt_eval"

# Checkpoints to evaluate
CHECKPOINTS = {
    "seg_only": "/root/autodl-tmp/PEA-MedSeg/results/formal/seg_only_seed42/epoch050.pth",
    "pfa_ours": "/root/autodl-tmp/PEA-MedSeg/results/formal/switch_loss_seed42/best.pth",
}

ORGAN_ADJACENCY = {
    1: [2, 3, 8, 9],      # spleen
    2: [1, 3, 9, 10],     # kidney_L
    3: [1, 2, 8],         # kidney_R
    6: [7, 8, 9, 10],     # liver
    7: [6, 1, 4],         # stomach
    4: [7, 8, 9],         # pancreas
    8: [6, 4, 9],         # aorta
    9: [6, 8, 10],        # IVC
    10: [6, 9],           # portal_vein
}

SEED = 42
NUM_SAMPLES = 80
CROP_SIZE = 128


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_eval(ckpt_path, device):
    """Load SAM-Med3D + LoRA with a specific checkpoint."""
    model = load_model(SAM_ROOT, CKPT, "vit_b_ori", device)
    apply_lora_to_sam_med3d(model, rank=8)
    model.to(device)
    # Load LoRA weights
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def sample_boundary_point(mask, rng, boundary_dist=(1, 3)):
    """Sample a point within 1-3 voxels of the organ boundary."""
    if mask.sum() < 50:
        return None
    # Distance transform from background
    dist_from_bg = distance_transform_edt(mask)
    # Boundary zone: 1 <= dist <= 3
    boundary_zone = (dist_from_bg >= boundary_dist[0]) & (dist_from_bg <= boundary_dist[1])
    coords = np.argwhere(boundary_zone)
    if len(coords) == 0:
        # Fallback: just use dist == 1
        boundary_zone = dist_from_bg == 1
        coords = np.argwhere(boundary_zone)
    if len(coords) == 0:
        return None
    idx = rng.integers(0, len(coords))
    return coords[idx]


def sample_noisy_point(mask, rng, noise_range=5):
    """Sample interior point then add uniform noise ±5 voxels."""
    eroded = binary_erosion(mask, iterations=3)
    coords = np.argwhere(eroded)
    if len(coords) < 10:
        coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    idx = rng.integers(0, len(coords))
    point = coords[idx].astype(np.float64)
    # Add noise
    noise = rng.uniform(-noise_range, noise_range, size=3)
    point = point + noise
    # Clamp to volume bounds
    point = np.clip(point, 0, np.array(mask.shape) - 1).astype(np.int64)
    return point


def sample_interior_point(mask, rng):
    """Standard interior point (eroded by 3)."""
    eroded = binary_erosion(mask, iterations=3)
    coords = np.argwhere(eroded)
    if len(coords) < 10:
        coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    idx = rng.integers(0, len(coords))
    return coords[idx]


def dice_score(pred, gt):
    """Binary Dice."""
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 0.0
    return (2.0 * inter / union).item()


def evaluate_single_prompt(model, image, gt_mask, point, device):
    """Run inference with a single point prompt, return predicted mask."""
    img_t = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
    point_t = torch.tensor(point, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = infer_with_point(model, img_t, point_t, device, CROP_SIZE)

    pred_bin = (pred.squeeze() > 0.5).cpu().numpy()
    return pred_bin


def evaluate_hard_prompts(model, device, rng):
    """
    Evaluate model under three hard prompt conditions:
    1. Boundary clicks → Dice
    2. Noisy clicks → Dice
    3. Adjacent-organ switch → Switch Accuracy

    Also evaluates standard interior clicks as reference.
    """
    img_dir = Path(DATA_DIR) / "imagesVa"
    lbl_dir = Path(DATA_DIR) / "labelsVa"

    cases = sorted(img_dir.glob("*.npy"))
    if not cases:
        cases = sorted(img_dir.glob("*.nii.gz"))

    results = {
        "interior": {"dices": [], "n": 0},
        "boundary": {"dices": [], "n": 0},
        "noisy": {"dices": [], "n": 0},
        "adjacent_switch": {"correct": 0, "total": 0, "dices_A": [], "dices_B": []},
    }

    n_evaluated = 0

    for case_path in cases:
        if n_evaluated >= NUM_SAMPLES:
            break

        # Load volume
        image = np.load(str(case_path)).astype(np.float32)
        lbl_path = lbl_dir / case_path.name
        label = np.load(str(lbl_path)).astype(np.int16)

        # Find organs present
        organs_present = [o for o in np.unique(label) if o > 0 and o in ORGAN_ADJACENCY]
        if len(organs_present) < 2:
            continue

        # Random crop
        for _ in range(4):  # 4 crops per volume
            if n_evaluated >= NUM_SAMPLES:
                break

            # Pick a random organ and crop around it
            organ_id = rng.choice(organs_present)
            organ_mask = (label == organ_id)
            coords = np.argwhere(organ_mask)
            if len(coords) < 200:
                continue

            # Center crop
            center = coords[rng.integers(0, len(coords))]
            half = CROP_SIZE // 2
            slices = tuple(
                slice(max(0, c - half), max(0, c - half) + CROP_SIZE)
                for c in center
            )

            img_crop = image[slices]
            lbl_crop = label[slices]

            # Pad if needed
            if img_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
                pad_img = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.float32)
                pad_lbl = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=np.int16)
                s = img_crop.shape
                pad_img[:s[0], :s[1], :s[2]] = img_crop
                pad_lbl[:s[0], :s[1], :s[2]] = lbl_crop
                img_crop, lbl_crop = pad_img, pad_lbl

            gt_mask = (lbl_crop == organ_id).astype(np.uint8)
            if gt_mask.sum() < 100:
                continue

            # --- 1. Interior click (reference) ---
            pt = sample_interior_point(gt_mask, rng)
            if pt is not None:
                pred = evaluate_single_prompt(model, img_crop, gt_mask, pt, device)
                d = dice_score(pred, gt_mask)
                results["interior"]["dices"].append(d)
                results["interior"]["n"] += 1

            # --- 2. Boundary click ---
            pt_b = sample_boundary_point(gt_mask, rng)
            if pt_b is not None:
                pred = evaluate_single_prompt(model, img_crop, gt_mask, pt_b, device)
                d = dice_score(pred, gt_mask)
                results["boundary"]["dices"].append(d)
                results["boundary"]["n"] += 1

            # --- 3. Noisy click ---
            pt_n = sample_noisy_point(gt_mask, rng)
            if pt_n is not None:
                pred = evaluate_single_prompt(model, img_crop, gt_mask, pt_n, device)
                d = dice_score(pred, gt_mask)
                results["noisy"]["dices"].append(d)
                results["noisy"]["n"] += 1

            # --- 4. Adjacent-organ switch ---
            adj_organs = [o for o in ORGAN_ADJACENCY.get(organ_id, [])
                         if o in organs_present and (lbl_crop == o).sum() > 100]
            if adj_organs:
                organ_B = rng.choice(adj_organs)
                gt_B = (lbl_crop == organ_B).astype(np.uint8)

                pt_A = sample_interior_point(gt_mask, rng)
                pt_B = sample_interior_point(gt_B, rng)

                if pt_A is not None and pt_B is not None:
                    pred_A = evaluate_single_prompt(model, img_crop, gt_mask, pt_A, device)
                    pred_B = evaluate_single_prompt(model, img_crop, gt_B, pt_B, device)

                    # Switch test: pred_A should match gt_A more than gt_B
                    d_AA = dice_score(pred_A, gt_mask)
                    d_AB = dice_score(pred_A, gt_B)
                    d_BB = dice_score(pred_B, gt_B)
                    d_BA = dice_score(pred_B, gt_mask)

                    switch_ok = (d_AA > d_AB) and (d_BB > d_BA)
                    results["adjacent_switch"]["correct"] += int(switch_ok)
                    results["adjacent_switch"]["total"] += 1
                    results["adjacent_switch"]["dices_A"].append(d_AA)
                    results["adjacent_switch"]["dices_B"].append(d_BB)

            n_evaluated += 1

    # Summarize
    summary = {
        "interior_dice": np.mean(results["interior"]["dices"]) if results["interior"]["dices"] else 0,
        "boundary_dice": np.mean(results["boundary"]["dices"]) if results["boundary"]["dices"] else 0,
        "noisy_dice": np.mean(results["noisy"]["dices"]) if results["noisy"]["dices"] else 0,
        "adjacent_switch_acc": (results["adjacent_switch"]["correct"] /
                                max(results["adjacent_switch"]["total"], 1)),
        "adjacent_switch_dice": np.mean(
            results["adjacent_switch"]["dices_A"] + results["adjacent_switch"]["dices_B"]
        ) if results["adjacent_switch"]["dices_A"] else 0,
        "n_interior": results["interior"]["n"],
        "n_boundary": results["boundary"]["n"],
        "n_noisy": results["noisy"]["n"],
        "n_switch_pairs": results["adjacent_switch"]["total"],
    }
    return summary


def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("HARD PROMPT EVALUATION BENCHMARK")
    print("Conditions: Interior | Boundary | Noisy | Adjacent Switch")
    print("=" * 60)

    all_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'─' * 40}")
        print(f"Evaluating: {name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'─' * 40}")

        model = load_model_eval(ckpt_path, DEVICE)
        results = evaluate_hard_prompts(model, DEVICE, rng)
        all_results[name] = results

        print(f"  Interior Dice:       {results['interior_dice']:.3f} (n={results['n_interior']})")
        print(f"  Boundary Dice:       {results['boundary_dice']:.3f} (n={results['n_boundary']})")
        print(f"  Noisy Dice:          {results['noisy_dice']:.3f} (n={results['n_noisy']})")
        print(f"  Adjacent Switch Acc: {results['adjacent_switch_acc']:.3f} (n={results['n_switch_pairs']})")
        print(f"  Adjacent Switch Dice:{results['adjacent_switch_dice']:.3f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Reset RNG for fair comparison
        rng = np.random.default_rng(SEED)

    # Save results
    with open(os.path.join(SAVE_DIR, "hard_prompt_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Metric':<25} {'seg_only':<12} {'pfa_ours':<12} {'Delta':<10}")
    print("-" * 60)

    for metric in ["interior_dice", "boundary_dice", "noisy_dice",
                   "adjacent_switch_acc", "adjacent_switch_dice"]:
        v_seg = all_results["seg_only"][metric]
        v_pfa = all_results["pfa_ours"][metric]
        delta = v_pfa - v_seg
        sign = "+" if delta > 0 else ""
        print(f"  {metric:<23} {v_seg:<12.3f} {v_pfa:<12.3f} {sign}{delta:.3f}")

    print("=" * 60)
    print(f"Results saved: {SAVE_DIR}/hard_prompt_results.json")


if __name__ == "__main__":
    main()

