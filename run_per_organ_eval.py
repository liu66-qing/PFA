#!/usr/bin/env python3
"""
Per-Organ Evaluation + Runtime Analysis
========================================
1. Per-organ Dice and Switch Accuracy for seg-only vs PFA (ours)
2. Runtime/compute overhead comparison
"""
import sys
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg/src")
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path
from scipy.ndimage import binary_erosion

from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point

DEVICE = "cuda:0"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"
SAVE_DIR = "/root/autodl-tmp/PEA-MedSeg/results/formal/per_organ_eval"

CHECKPOINTS = {
    "seg_only": "/root/autodl-tmp/PEA-MedSeg/results/formal/seg_only_seed42/epoch050.pth",
    "pfa_ours": "/root/autodl-tmp/PEA-MedSeg/results/formal/switch_loss_seed42/best.pth",
}

ORGAN_NAMES = {
    1: "Spleen", 2: "Kidney_L", 3: "Kidney_R", 4: "Pancreas",
    6: "Liver", 7: "Stomach", 8: "Aorta", 9: "IVC",
    10: "Portal_Vein", 11: "Adrenal_L", 12: "Adrenal_R",
    13: "Gallbladder", 14: "Bladder",
}

ORGAN_ADJACENCY = {
    1: [2, 3, 8, 9], 2: [1, 3, 9, 10], 3: [1, 2, 8],
    6: [7, 8, 9, 10], 7: [6, 1, 4], 4: [7, 8, 9],
    8: [6, 4, 9], 9: [6, 8, 10], 10: [6, 9],
}

SEED = 42
CROP_SIZE = 128
NUM_EVAL_CROPS = 120


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_eval(ckpt_path, device):
    model = load_model(SAM_ROOT, CKPT, "vit_b_ori", device)
    apply_lora_to_sam_med3d(model, rank=8)
    model.to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def sample_interior_point(mask, rng):
    eroded = binary_erosion(mask, iterations=3)
    coords = np.argwhere(eroded)
    if len(coords) < 10:
        coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    return coords[rng.integers(0, len(coords))]


def dice_score(pred, gt):
    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    if union == 0:
        return 0.0
    return (2.0 * inter / union).item() if torch.is_tensor(inter) else 2.0 * inter / union


def per_organ_evaluation(model, device, rng):
    """Evaluate Dice and Switch per organ."""
    img_dir = Path(DATA_DIR) / "imagesVa"
    lbl_dir = Path(DATA_DIR) / "labelsVa"
    cases = sorted(img_dir.glob("*.npy"))

    # Per-organ metrics
    organ_dices = {oid: [] for oid in ORGAN_NAMES}
    organ_switch = {oid: {"correct": 0, "total": 0} for oid in ORGAN_NAMES}

    n_evaluated = 0

    for case_path in cases:
        if n_evaluated >= NUM_EVAL_CROPS:
            break

        image = np.load(str(case_path)).astype(np.float32)
        lbl_path = lbl_dir / case_path.name
        label = np.load(str(lbl_path)).astype(np.int16)

        organs_present = [o for o in np.unique(label) if o in ORGAN_NAMES and (label == o).sum() > 200]
        if len(organs_present) < 1:
            continue

        for organ_id in organs_present:
            if n_evaluated >= NUM_EVAL_CROPS:
                break

            organ_mask = (label == organ_id)
            coords = np.argwhere(organ_mask)
            if len(coords) < 200:
                continue

            # Center crop around organ
            center = coords[rng.integers(0, len(coords))]
            half = CROP_SIZE // 2
            slices = tuple(slice(max(0, c - half), max(0, c - half) + CROP_SIZE) for c in center)

            img_crop = image[slices]
            lbl_crop = label[slices]

            if img_crop.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
                pad_img = np.zeros((CROP_SIZE,) * 3, dtype=np.float32)
                pad_lbl = np.zeros((CROP_SIZE,) * 3, dtype=np.int16)
                s = img_crop.shape
                pad_img[:s[0], :s[1], :s[2]] = img_crop
                pad_lbl[:s[0], :s[1], :s[2]] = lbl_crop
                img_crop, lbl_crop = pad_img, pad_lbl

            gt_A = (lbl_crop == organ_id).astype(np.uint8)
            if gt_A.sum() < 100:
                continue

            # Dice evaluation
            pt_A = sample_interior_point(gt_A, rng)
            if pt_A is None:
                continue

            img_t = torch.from_numpy(img_crop).float().unsqueeze(0).unsqueeze(0).to(device)
            pt_t = torch.tensor(pt_A, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                pred, _ = infer_with_point(model, img_t, pt_t, device)
            pred_bin = (pred.squeeze() > 0.5).cpu().numpy()
            d = dice_score(pred_bin, gt_A)
            organ_dices[organ_id].append(d)

            # Switch evaluation: find another organ in same crop
            other_organs = [o for o in np.unique(lbl_crop)
                           if o in ORGAN_NAMES and o != organ_id and (lbl_crop == o).sum() > 100]
            if other_organs:
                organ_B = rng.choice(other_organs)
                gt_B = (lbl_crop == organ_B).astype(np.uint8)
                pt_B = sample_interior_point(gt_B, rng)
                if pt_B is not None:
                    pt_Bt = torch.tensor(pt_B, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_B, _ = infer_with_point(model, img_t, pt_Bt, device)
                    pred_B_bin = (pred_B.squeeze() > 0.5).cpu().numpy()

                    d_AA = dice_score(pred_bin, gt_A)
                    d_AB = dice_score(pred_bin, gt_B)
                    d_BB = dice_score(pred_B_bin, gt_B)
                    d_BA = dice_score(pred_B_bin, gt_A)

                    switch_ok = (d_AA > d_AB) and (d_BB > d_BA)
                    organ_switch[organ_id]["total"] += 1
                    organ_switch[organ_id]["correct"] += int(switch_ok)

            n_evaluated += 1

    # Summarize
    summary = {}
    for oid, name in ORGAN_NAMES.items():
        dices = organ_dices[oid]
        sw = organ_switch[oid]
        summary[name] = {
            "dice_mean": float(np.mean(dices)) if dices else 0.0,
            "dice_std": float(np.std(dices)) if dices else 0.0,
            "n_dice": len(dices),
            "switch_acc": sw["correct"] / max(sw["total"], 1),
            "n_switch": sw["total"],
        }
    return summary


def runtime_analysis(model, device):
    """Measure inference time, memory, and training overhead."""
    import torch.cuda

    results = {}

    # Model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["total_params_M"] = total_params / 1e6
    results["trainable_params_M"] = trainable_params / 1e6
    results["trainable_pct"] = 100.0 * trainable_params / total_params

    # Inference time (single point prompt)
    dummy_img = torch.randn(1, 1, CROP_SIZE, CROP_SIZE, CROP_SIZE, device=device)
    dummy_pt = torch.tensor([[64.0, 64.0, 64.0]], device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = infer_with_point(model, dummy_img, dummy_pt, device)

    torch.cuda.synchronize()

    # Measure inference
    n_runs = 20
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = infer_with_point(model, dummy_img, dummy_pt, device)
    torch.cuda.synchronize()
    t1 = time.time()
    results["inference_ms"] = (t1 - t0) / n_runs * 1000

    # Measure encoder vs decoder separately
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            emb = model.image_encoder(dummy_img)
    torch.cuda.synchronize()
    t1 = time.time()
    results["encoder_ms"] = (t1 - t0) / n_runs * 1000

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            pt_tensor = dummy_pt.view(1, 1, 3)
            lbl_tensor = torch.ones((1, 1), dtype=torch.long, device=device)
            low_res = torch.zeros((1, 1, CROP_SIZE//4, CROP_SIZE//4, CROP_SIZE//4), device=device)
            sparse, dense = model.prompt_encoder(
                points=[pt_tensor, lbl_tensor], boxes=None, masks=low_res)
            _ = model.mask_decoder(
                image_embeddings=emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False)
    torch.cuda.synchronize()
    t1 = time.time()
    results["decoder_ms"] = (t1 - t0) / n_runs * 1000

    # GPU memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = infer_with_point(model, dummy_img, dummy_pt, device)
    results["peak_memory_MB"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # Training time from history
    seg_hist = "/root/autodl-tmp/PEA-MedSeg/results/formal/seg_only_seed42/history.json"
    pfa_hist = "/root/autodl-tmp/PEA-MedSeg/results/formal/switch_loss_seed42/history.json"

    if os.path.exists(seg_hist):
        with open(seg_hist) as f:
            seg_h = json.load(f)
        results["seg_only_time_per_epoch"] = np.mean([e["time"] for e in seg_h])
        results["seg_only_total_time_h"] = sum(e["time"] for e in seg_h) / 3600

    if os.path.exists(pfa_hist):
        with open(pfa_hist) as f:
            pfa_h = json.load(f)
        results["pfa_time_per_epoch"] = np.mean([e["time"] for e in pfa_h])
        results["pfa_total_time_h"] = sum(e["time"] for e in pfa_h) / 3600

    # Checkpoint size
    ckpt_path = CHECKPOINTS["pfa_ours"]
    results["checkpoint_size_MB"] = os.path.getsize(ckpt_path) / 1024 / 1024

    return results


def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("PER-ORGAN EVALUATION + RUNTIME ANALYSIS")
    print("=" * 70)

    all_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'─' * 50}")
        print(f"Evaluating: {name}")
        print(f"{'─' * 50}")

        rng = np.random.default_rng(SEED)
        model = load_model_eval(ckpt_path, DEVICE)
        organ_results = per_organ_evaluation(model, DEVICE, rng)
        all_results[name] = organ_results

        # Print per-organ table
        print(f"\n{'Organ':<15} {'Dice':<12} {'Switch':<12} {'n':<5}")
        print("-" * 45)
        for organ_name, metrics in sorted(organ_results.items(), key=lambda x: -x[1]["dice_mean"]):
            d = metrics["dice_mean"]
            s = metrics["switch_acc"]
            n = metrics["n_dice"]
            if n > 0:
                print(f"  {organ_name:<13} {d:.3f}±{metrics['dice_std']:.3f}  {s:.3f}        {n}")

        del model
        torch.cuda.empty_cache()

    # Runtime analysis (use PFA model)
    print(f"\n{'─' * 50}")
    print("RUNTIME ANALYSIS")
    print(f"{'─' * 50}")

    model = load_model_eval(CHECKPOINTS["pfa_ours"], DEVICE)
    runtime = runtime_analysis(model, DEVICE)
    all_results["runtime"] = runtime

    print(f"  Total params:        {runtime['total_params_M']:.1f}M")
    print(f"  Trainable params:    {runtime['trainable_params_M']:.2f}M ({runtime['trainable_pct']:.2f}%)")
    print(f"  Inference time:      {runtime['inference_ms']:.1f}ms")
    print(f"  Encoder time:        {runtime['encoder_ms']:.1f}ms")
    print(f"  Decoder time:        {runtime['decoder_ms']:.1f}ms")
    print(f"  Peak GPU memory:     {runtime['peak_memory_MB']:.0f}MB")
    print(f"  Checkpoint size:     {runtime['checkpoint_size_MB']:.1f}MB")

    if "seg_only_time_per_epoch" in runtime:
        print(f"  Seg-only time/epoch: {runtime['seg_only_time_per_epoch']:.0f}s")
        print(f"  PFA time/epoch:      {runtime['pfa_time_per_epoch']:.0f}s")
        overhead = (runtime['pfa_time_per_epoch'] - runtime['seg_only_time_per_epoch']) / runtime['seg_only_time_per_epoch'] * 100
        print(f"  Training overhead:   +{overhead:.1f}%")
        print(f"  Seg-only total:      {runtime['seg_only_total_time_h']:.2f}h")
        print(f"  PFA total:           {runtime['pfa_total_time_h']:.2f}h")

    # Save
    with open(os.path.join(SAVE_DIR, "per_organ_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n{'=' * 70}")
    print(f"Results saved: {SAVE_DIR}/per_organ_results.json")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

