"""Fixed Prompt Bank Evaluation v2 - bypasses dataset RNG.
Loads volumes directly, uses bank organ IDs for deterministic GT construction.
Covers P0: fixed eval + stats + per-organ.
"""
import sys, os, json, torch, time
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, "src")
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, infer_with_embedding

DEVICE = "cuda:0"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
BASE_CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
DATA_DIR = Path("/root/autodl-tmp/data/amos22_cached")
BANK_PATH = "results/prompt_bank_amos.json"
OUTPUT_DIR = Path("results/fixed_eval_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CROP_SIZE = 128

CHECKPOINTS = {
    "frozen": None,
    "seg_only_seed42": "results/formal/seg_only_seed42/epoch050.pth",
    "seg_only_seed123": "results/formal/seg_only_seed123/epoch050.pth",
    "seg_only_seed7": "results/formal/seg_only_seed7/epoch050.pth",
    "switch_loss_seed42_best": "results/formal/switch_loss_seed42/best.pth",
    "switch_loss_seed123_best": "results/formal/switch_loss_seed123/best.pth",
    "switch_loss_seed7_best": "results/formal/switch_loss_seed7/best.pth",
}


def crop_around_organ(image, label, organ_id, crop_size=128):
    mask = (label == organ_id)
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None, None
    center = np.rint(coords.mean(axis=0)).astype(int)
    half = crop_size // 2
    start = center - half
    end = start + crop_size
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.array(image.shape))
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    roi_image = np.zeros((crop_size,)*3, dtype=np.float32)
    roi_label = np.zeros((crop_size,)*3, dtype=np.int16)
    src_sl = tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_end))
    dst_sl = tuple(slice(int(s), int(e)) for s, e in zip(dst_start, dst_end))
    roi_image[dst_sl] = image[src_sl]
    roi_label[dst_sl] = label[src_sl]
    return roi_image, roi_label


def dice_score(pred, gt):
    inter = (pred * gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    if union == 0:
        return 0.0
    return 2 * inter / (union + 1e-6)


def load_case(case_idx):
    img_dir = DATA_DIR / "imagesVa"
    lbl_dir = DATA_DIR / "labelsVa"
    cases = sorted(img_dir.glob("*.npy"))
    if case_idx >= len(cases):
        return None, None
    img = np.load(str(cases[case_idx])).astype(np.float32)
    lbl = np.load(str(lbl_dir / cases[case_idx].name)).astype(np.int16)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img, lbl


def evaluate_checkpoint(model, bank, device):
    model.eval()
    results = []
    cache = {}
    with torch.no_grad():
        for entry in bank:
            case_idx = entry["case_idx"]
            organ_A = entry["primary_organ_id"]
            organ_B = entry["cross_organ_id"]
            if case_idx not in cache:
                img, lbl = load_case(case_idx)
                if img is None:
                    continue
                cache[case_idx] = (img, lbl)
            else:
                img, lbl = cache[case_idx]

            roi_img, roi_lbl = crop_around_organ(img, lbl, organ_A, CROP_SIZE)
            if roi_img is None:
                continue
            gt_A = (roi_lbl == organ_A).astype(np.float32)
            gt_B = (roi_lbl == organ_B).astype(np.float32)
            if gt_A.sum() < 50 or gt_B.sum() < 50:
                continue

            coords_A = np.argwhere(gt_A > 0)
            coords_B = np.argwhere(gt_B > 0)
            pt_A = coords_A.mean(axis=0).astype(int)
            pt_B = coords_B.mean(axis=0).astype(int)
            if gt_A[pt_A[0], pt_A[1], pt_A[2]] == 0:
                idx = np.argmin(np.sum((coords_A - pt_A)**2, axis=1))
                pt_A = coords_A[idx]
            if gt_B[pt_B[0], pt_B[1], pt_B[2]] == 0:
                idx = np.argmin(np.sum((coords_B - pt_B)**2, axis=1))
                pt_B = coords_B[idx]

            img_t = torch.from_numpy(roi_img).unsqueeze(0).unsqueeze(0).to(device)
            gt_A_t = torch.from_numpy(gt_A).unsqueeze(0).unsqueeze(0).to(device)
            gt_B_t = torch.from_numpy(gt_B).unsqueeze(0).unsqueeze(0).to(device)
            pt_A_t = torch.tensor(pt_A, dtype=torch.float32).to(device)
            pt_B_t = torch.tensor(pt_B, dtype=torch.float32).to(device)

            logits_A, img_emb = infer_with_point(model, img_t, pt_A_t, device)
            pred_A = (torch.sigmoid(logits_A) > 0.5).float()
            logits_B = infer_with_embedding(model, img_emb, pt_B_t, device, CROP_SIZE)
            pred_B = (torch.sigmoid(logits_B) > 0.5).float()

            d_A_A = dice_score(pred_A, gt_A_t)
            d_A_B = dice_score(pred_A, gt_B_t)
            d_B_B = dice_score(pred_B, gt_B_t)
            d_B_A = dice_score(pred_B, gt_A_t)

            switch_ok = (d_A_A > d_A_B) and (d_B_B > d_B_A)
            results.append({
                "organ_A": organ_A, "organ_B": organ_B,
                "dice_A": d_A_A, "dice_B": d_B_B,
                "switch": 1 if switch_ok else 0,
                "case_idx": case_idx,
            })
    return results


def main():
    print("=" * 60)
    print("P0: FIXED EVAL V2")
    print("=" * 60)
    with open(BANK_PATH) as f:
        bank = json.load(f)
    print(f"Prompt bank: {len(bank)} entries")

    all_results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        model = load_model(SAM_ROOT, BASE_CKPT, "vit_b_ori", torch.device(DEVICE))
        apply_lora_to_sam_med3d(model)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        results = evaluate_checkpoint(model, bank, DEVICE)
        sw = np.mean([r["switch"] for r in results]) if results else 0
        dices = [r["dice_A"] for r in results] + [r["dice_B"] for r in results]
        dice_mean = np.mean(dices) if dices else 0
        elapsed = time.time() - t0
        print(f"  Switch={sw:.3f} Dice={dice_mean:.3f} n={len(results)} ({elapsed:.0f}s)")
        all_results[name] = results
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump({"switch": sw, "dice": dice_mean, "n": len(results)}, f, indent=2)
        del model
        torch.cuda.empty_cache()

    # Stats
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    for seed in [42, 123, 7]:
        sw_key = f"switch_loss_seed{seed}_best"
        seg_key = f"seg_only_seed{seed}"
        if sw_key not in all_results or seg_key not in all_results:
            continue
        sw_s = [r["switch"] for r in all_results[sw_key]]
        seg_s = [r["switch"] for r in all_results[seg_key]]
        n = min(len(sw_s), len(seg_s))
        n01 = sum(1 for a, b in zip(sw_s[:n], seg_s[:n]) if a == 1 and b == 0)
        n10 = sum(1 for a, b in zip(sw_s[:n], seg_s[:n]) if a == 0 and b == 1)
        if n01 + n10 > 0:
            stat = (abs(n01 - n10) - 1)**2 / (n01 + n10)
            p = 1 - stats.chi2.cdf(stat, 1)
        else:
            p = 1.0
        print(f"  Seed {seed}: McNemar p={p:.6f} (sw>seg={n01}, seg>sw={n10})")

    # Per-organ
    print("\n" + "=" * 60)
    print("PER-ORGAN")
    print("=" * 60)
    for method in ["switch_loss_seed42_best", "seg_only_seed42"]:
        if method not in all_results:
            continue
        print(f"\n  {method}:")
        per_organ = {}
        for r in all_results[method]:
            for oid, d in [(r["organ_A"], r["dice_A"]), (r["organ_B"], r["dice_B"])]:
                k = str(oid)
                if k not in per_organ:
                    per_organ[k] = {"dices": [], "switches": []}
                per_organ[k]["dices"].append(d)
                per_organ[k]["switches"].append(r["switch"])
        for oid in sorted(per_organ.keys(), key=lambda x: int(x)):
            data = per_organ[oid]
            print(f"    Organ {oid:>2}: Dice={np.mean(data['dices']):.3f} Switch={np.mean(data['switches']):.3f} (n={len(data['dices'])})")
        with open(OUTPUT_DIR / f"per_organ_{method}.json", "w") as f:
            json.dump({k: {"dice": float(np.mean(v["dices"])), "switch": float(np.mean(v["switches"])), "n": len(v["dices"])} for k, v in per_organ.items()}, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name in CHECKPOINTS:
        if name in all_results:
            r = all_results[name]
            sw = np.mean([x["switch"] for x in r])
            d = np.mean([x["dice_A"] for x in r] + [x["dice_B"] for x in r])
            print(f"  {name:30s}: Switch={sw:.3f} Dice={d:.3f} (n={len(r)})")
    print("\nDONE.")


if __name__ == "__main__":
    main()
