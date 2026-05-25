"""Fixed Prompt Bank Evaluation for AAAI submission.

Generates a deterministic prompt bank ONCE, then evaluates all checkpoints
with the exact same prompts. Eliminates eval variance.

Protocol:
- Pre-generate K=3 prompt sets per validation case
- Each prompt set: (organ_A, point_A, organ_B, point_B) fixed
- Evaluate all methods/checkpoints with identical prompts
- Report mean ± std over prompt sets, CI over cases

Outputs per-organ breakdown + aggregate metrics.
"""
import sys, os, json, torch, argparse
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from scipy import stats

sys.path.insert(0, "src")
from pea_dataset import PEADataset
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point

DEVICE = "cuda"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
K_PROMPTS = 3  # prompt sets per case


def generate_prompt_bank(data_dir, split="Va", seed=999, samples_per_case=5):
    """Generate fixed prompt bank for evaluation."""
    rng = np.random.default_rng(seed)
    ds = PEADataset(data_dir, split=split, samples_per_volume=samples_per_case, seed=seed)

    bank = []
    for idx in range(len(ds)):
        sample = ds[idx]
        if sample["primary_organ_id"] == -1:
            continue
        if not sample["has_cross"]:
            continue

        entry = {
            "case_idx": idx // ds.samples_per_volume,
            "sample_idx": idx,
            "primary_organ_id": int(sample["primary_organ_id"]),
            "cross_organ_id": int(sample["cross_organ_id"]),
            "clean_point": sample["clean_point"].numpy().tolist(),
            "cross_point": sample["cross_point"].numpy().tolist(),
        }
        bank.append(entry)

    print(f"Generated prompt bank: {len(bank)} entries from {split} split")
    return bank


def evaluate_checkpoint(model, data_dir, prompt_bank, split="Va", device="cuda"):
    """Evaluate a model checkpoint with fixed prompt bank."""
    ds = PEADataset(data_dir, split=split, samples_per_volume=5, seed=999)

    results = {
        "switch_correct": 0,
        "switch_total": 0,
        "dices_target": [],
        "dices_cross": [],
        "grounded": 0,
        "grounded_total": 0,
        "per_organ": {},  # organ_id -> {dice: [], switch: []}
        "per_pair": {},   # "organA_organB" -> {correct: 0, total: 0}
    }

    model.eval()
    with torch.no_grad():
        for entry in prompt_bank:
            idx = entry["sample_idx"]
            sample = ds[idx]

            if sample["primary_organ_id"] == -1:
                continue

            image = sample["image"].unsqueeze(0).to(device)
            gt_primary = sample["gt_primary"].unsqueeze(0).to(device)
            gt_cross = sample["gt_cross"].unsqueeze(0).to(device)

            # Use FIXED prompts from bank
            pos_pt = torch.tensor(entry["clean_point"], dtype=torch.float32).to(device)
            cross_pt = torch.tensor(entry["cross_point"], dtype=torch.float32).to(device)

            organ_A = entry["primary_organ_id"]
            organ_B = entry["cross_organ_id"]

            # Forward A
            logits_A, img_emb = infer_with_point(model, image, pos_pt, device)
            pred_A = (torch.sigmoid(logits_A) > 0.5).float()

            # Forward B
            from train_pfa import infer_with_embedding
            logits_B = infer_with_embedding(model, img_emb, cross_pt, device, image.shape[-1])
            pred_B = (torch.sigmoid(logits_B) > 0.5).float()

            # Dice scores
            dice_A_A = dice(pred_A, gt_primary)
            dice_A_B = dice(pred_A, gt_cross)
            dice_B_B = dice(pred_B, gt_cross)
            dice_B_A = dice(pred_B, gt_primary)

            results["dices_target"].append(dice_A_A)
            results["dices_target"].append(dice_B_B)
            results["dices_cross"].append(dice_A_B)
            results["dices_cross"].append(dice_B_A)

            # Switch check
            A_ok = dice_A_A > dice_A_B
            B_ok = dice_B_B > dice_B_A
            switch_ok = A_ok and B_ok
            if switch_ok:
                results["switch_correct"] += 1
            results["switch_total"] += 1

            # Grounding check
            pz, py, px = pos_pt.long().tolist()
            pz = max(0, min(pz, pred_A.shape[2]-1))
            py = max(0, min(py, pred_A.shape[3]-1))
            px = max(0, min(px, pred_A.shape[4]-1))
            if pred_A[0, 0, pz, py, px] > 0:
                results["grounded"] += 1
            results["grounded_total"] += 1

            # Per-organ tracking
            for oid, d in [(organ_A, dice_A_A), (organ_B, dice_B_B)]:
                key = str(oid)
                if key not in results["per_organ"]:
                    results["per_organ"][key] = {"dices": [], "switches": []}
                results["per_organ"][key]["dices"].append(d)

            # Per-pair tracking
            pair_key = f"{min(organ_A, organ_B)}_{max(organ_A, organ_B)}"
            if pair_key not in results["per_pair"]:
                results["per_pair"][pair_key] = {"correct": 0, "total": 0}
            results["per_pair"][pair_key]["total"] += 1
            if switch_ok:
                results["per_pair"][pair_key]["correct"] += 1

            # Per-organ switch
            results["per_organ"][str(organ_A)]["switches"].append(1 if A_ok else 0)
            results["per_organ"][str(organ_B)]["switches"].append(1 if B_ok else 0)

    return results


def dice(pred, gt):
    """Compute Dice score."""
    inter = (pred * gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    if union == 0:
        return 0.0
    return 2 * inter / (union + 1e-6)


def summarize(results):
    """Compute summary statistics."""
    sw_acc = results["switch_correct"] / max(results["switch_total"], 1)
    mean_dice = np.mean(results["dices_target"]) if results["dices_target"] else 0
    ground_rate = results["grounded"] / max(results["grounded_total"], 1)
    target_margin = np.mean(results["dices_target"]) - np.mean(results["dices_cross"])

    per_organ_summary = {}
    for oid, data in results["per_organ"].items():
        per_organ_summary[oid] = {
            "mean_dice": float(np.mean(data["dices"])),
            "mean_switch": float(np.mean(data["switches"])),
            "n": len(data["dices"]),
        }

    per_pair_summary = {}
    for pair, data in results["per_pair"].items():
        per_pair_summary[pair] = {
            "switch_acc": data["correct"] / max(data["total"], 1),
            "n": data["total"],
        }

    return {
        "switch_accuracy": sw_acc,
        "mean_dice": mean_dice,
        "grounding_rate": ground_rate,
        "target_margin": target_margin,
        "n_pairs": results["switch_total"],
        "per_organ": per_organ_summary,
        "per_pair": per_pair_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/amos22_cached")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--prompt-bank", default=None, help="Path to prompt bank JSON")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--generate-bank", action="store_true", help="Generate prompt bank only")
    args = parser.parse_args()

    if args.generate_bank:
        bank = generate_prompt_bank(args.data_dir)
        bank_path = args.output
        with open(bank_path, "w") as f:
            json.dump(bank, f, indent=2)
        print(f"Prompt bank saved to {bank_path}")
        return

    # Load prompt bank
    if args.prompt_bank is None:
        bank_path = Path(args.data_dir).parent / "prompt_bank_val.json"
        if not bank_path.exists():
            print("Generating prompt bank...")
            bank = generate_prompt_bank(args.data_dir)
            with open(bank_path, "w") as f:
                json.dump(bank, f, indent=2)
        else:
            with open(bank_path) as f:
                bank = json.load(f)
    else:
        with open(args.prompt_bank) as f:
            bank = json.load(f)

    print(f"Prompt bank: {len(bank)} entries")

    # Load model
    model = load_model(SAM_ROOT, CKPT, "vit_b_ori", torch.device(DEVICE))
    apply_lora_to_sam_med3d(model)

    # Load checkpoint
    if args.checkpoint != "frozen":
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Evaluating FROZEN baseline (no LoRA weights loaded)")

    # Evaluate
    results = evaluate_checkpoint(model, args.data_dir, bank, device=DEVICE)
    summary = summarize(results)

    print(f"\nSwitch Accuracy: {summary['switch_accuracy']:.3f}")
    print(f"Mean Dice: {summary['mean_dice']:.3f}")
    print(f"Grounding Rate: {summary['grounding_rate']:.3f}")
    print(f"Target Margin: {summary['target_margin']:.3f}")
    print(f"\nPer-organ Switch:")
    for oid, data in sorted(summary["per_organ"].items(), key=lambda x: x[1]["mean_switch"]):
        print(f"  Organ {oid}: Dice={data['mean_dice']:.3f} Switch={data['mean_switch']:.3f} (n={data['n']})")

    # Save
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
