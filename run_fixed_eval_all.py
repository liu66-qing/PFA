"""Batch fixed-prompt evaluation + per-organ analysis + statistical tests.
Covers P0: fixed prompt bank, stats, per-organ breakdown.
"""
import sys, os, json, torch, time
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, "src")
from pea_dataset import PEADataset
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, infer_with_embedding

DEVICE = "cuda:0"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
BASE_CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"
BANK_PATH = "results/prompt_bank_amos.json"
OUTPUT_DIR = Path("results/fixed_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS = {
    "frozen": None,
    "seg_only_seed42": "results/formal/seg_only_seed42/epoch050.pth",
    "seg_only_seed123": "results/formal/seg_only_seed123/epoch050.pth",
    "seg_only_seed7": "results/formal/seg_only_seed7/epoch050.pth",
    "switch_loss_seed42_best": "results/formal/switch_loss_seed42/best.pth",
    "switch_loss_seed123_best": "results/formal/switch_loss_seed123/best.pth",
    "switch_loss_seed7_best": "results/formal/switch_loss_seed7/best.pth",
}


def dice_score(pred, gt):
    inter = (pred * gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    if union == 0:
        return 0.0
    return 2 * inter / (union + 1e-6)


def evaluate_with_bank(model, bank, ds, device):
    model.eval()
    results = []
    with torch.no_grad():
        for entry in bank:
            idx = entry["sample_idx"]
            if idx >= len(ds):
                continue
            sample = ds[idx]
            if sample["primary_organ_id"] == -1:
                continue
            if not sample["has_cross"]:
                continue

            image = sample["image"].unsqueeze(0).to(device)
            gt_A = sample["gt_primary"].unsqueeze(0).to(device)
            gt_B = sample["gt_cross"].unsqueeze(0).to(device)

            pos_pt = torch.tensor(entry["clean_point"], dtype=torch.float32).to(device)
            cross_pt = torch.tensor(entry["cross_point"], dtype=torch.float32).to(device)

            logits_A, img_emb = infer_with_point(model, image, pos_pt, device)
            pred_A = (torch.sigmoid(logits_A) > 0.5).float()

            logits_B = infer_with_embedding(model, img_emb, cross_pt, device, image.shape[-1])
            pred_B = (torch.sigmoid(logits_B) > 0.5).float()

            d_A_A = dice_score(pred_A, gt_A)
            d_A_B = dice_score(pred_A, gt_B)
            d_B_B = dice_score(pred_B, gt_B)
            d_B_A = dice_score(pred_B, gt_A)

            A_ok = d_A_A > d_A_B
            B_ok = d_B_B > d_B_A

            results.append({
                "organ_A": entry["primary_organ_id"],
                "organ_B": entry["cross_organ_id"],
                "dice_A": d_A_A, "dice_B": d_B_B,
                "dice_A_cross": d_A_B, "dice_B_cross": d_B_A,
                "switch": 1 if (A_ok and B_ok) else 0,
                "case_idx": entry["case_idx"],
            })
    return results


def main():
    print("=" * 60)
    print("P0: FIXED PROMPT BANK EVALUATION")
    print("=" * 60)

    with open(BANK_PATH) as f:
        bank = json.load(f)
    print(f"Prompt bank: {len(bank)} entries")

    ds = PEADataset(DATA_DIR, split="Va", samples_per_volume=5, seed=999)
    print(f"Dataset: {len(ds)} samples")

    all_results = {}

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        model = load_model(SAM_ROOT, BASE_CKPT, "vit_b_ori", torch.device(DEVICE))
        apply_lora_to_sam_med3d(model)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state, strict=False)

        results = evaluate_with_bank(model, bank, ds, DEVICE)
        sw = np.mean([r["switch"] for r in results])
        dices = [r["dice_A"] for r in results] + [r["dice_B"] for r in results]
        dice_mean = np.mean(dices)
        elapsed = time.time() - t0
        print(f"  Switch={sw:.3f} Dice={dice_mean:.3f} n={len(results)} ({elapsed:.0f}s)")

        all_results[name] = results
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump({"switch": sw, "dice": dice_mean, "n": len(results), "raw": results}, f)

        del model
        torch.cuda.empty_cache()

    # === STATISTICAL TESTS ===
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    stat_out = {}
    for seed in [42, 123, 7]:
        sw_key = f"switch_loss_seed{seed}_best"
        seg_key = f"seg_only_seed{seed}"
        if sw_key not in all_results or seg_key not in all_results:
            continue
        sw_switches = [r["switch"] for r in all_results[sw_key]]
        seg_switches = [r["switch"] for r in all_results[seg_key]]
        min_n = min(len(sw_switches), len(seg_switches))
        sw_switches = sw_switches[:min_n]
        seg_switches = seg_switches[:min_n]

        n01 = sum(1 for a, b in zip(sw_switches, seg_switches) if a == 1 and b == 0)
        n10 = sum(1 for a, b in zip(sw_switches, seg_switches) if a == 0 and b == 1)
        if n01 + n10 > 0:
            mcnemar_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10)
            p_val = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            p_val = 1.0
        print(f"  Seed {seed} McNemar: p={p_val:.6f} (sw>seg={n01}, seg>sw={n10})")
        stat_out[f"mcnemar_seed{seed}"] = {"p": p_val, "n01": n01, "n10": n10}

    # Pooled Wilcoxon on Switch
    all_sw_switch = []
    all_seg_switch = []
    for seed in [42, 123, 7]:
        sw_key = f"switch_loss_seed{seed}_best"
        seg_key = f"seg_only_seed{seed}"
        if sw_key in all_results and seg_key in all_results:
            all_sw_switch.extend([r["switch"] for r in all_results[sw_key]])
            all_seg_switch.extend([r["switch"] for r in all_results[seg_key]])
    if all_sw_switch:
        min_n = min(len(all_sw_switch), len(all_seg_switch))
        try:
            w_stat, w_p = stats.wilcoxon(all_sw_switch[:min_n], all_seg_switch[:min_n])
            print(f"  Pooled Wilcoxon Switch: p={w_p:.8f}")
            stat_out["wilcoxon_switch_pooled"] = {"p": float(w_p)}
        except Exception as e:
            print(f"  Wilcoxon failed: {e}")

    # === PER-ORGAN ANALYSIS ===
    print("\n" + "=" * 60)
    print("PER-ORGAN ANALYSIS")
    print("=" * 60)

    for method in ["switch_loss_seed42_best", "seg_only_seed42"]:
        if method not in all_results:
            continue
        print(f"\n  {method}:")
        per_organ = {}
        for r in all_results[method]:
            for oid, d in [(r["organ_A"], r["dice_A"]), (r["organ_B"], r["dice_B"])]:
                key = str(oid)
                if key not in per_organ:
                    per_organ[key] = {"dices": [], "switches": []}
                per_organ[key]["dices"].append(d)
                per_organ[key]["switches"].append(r["switch"])

        for oid in sorted(per_organ.keys(), key=lambda x: int(x)):
            data = per_organ[oid]
            print(f"    Organ {oid:>2}: Dice={np.mean(data['dices']):.3f} Switch={np.mean(data['switches']):.3f} (n={len(data['dices'])})")

        with open(OUTPUT_DIR / f"per_organ_{method}.json", "w") as f:
            json.dump({k: {"dice": float(np.mean(v["dices"])), "switch": float(np.mean(v["switches"])), "n": len(v["dices"])} for k, v in per_organ.items()}, f, indent=2)

    # Save stats
    with open(OUTPUT_DIR / "statistics.json", "w") as f:
        json.dump(stat_out, f, indent=2)

    # === SUMMARY TABLE ===
    print("\n" + "=" * 60)
    print("FINAL SUMMARY (Fixed Prompt Bank)")
    print("=" * 60)
    for name in CHECKPOINTS:
        if name in all_results:
            sw = np.mean([r["switch"] for r in all_results[name]])
            d = np.mean([r["dice_A"] for r in all_results[name]] + [r["dice_B"] for r in all_results[name]])
            print(f"  {name:30s}: Switch={sw:.3f} Dice={d:.3f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
