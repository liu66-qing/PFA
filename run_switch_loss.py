"""Direct Switch Loss Training: Paired prompt-contrastive adaptation.

Core idea: directly optimize the model to switch predictions when prompts switch.
For each crop with two organs (A, B):
  - prompt_A -> pred_A should match GT_A, not GT_B
  - prompt_B -> pred_B should match GT_B, not GT_A

L = 0.7 * [SegLoss(pred_A, GT_A) + SegLoss(pred_B, GT_B)] / 2
  + 2.0 * L_switch
  + 0.5 * L_ground (applied to both predictions)

L_switch = max(0, m + Dice(pred_A, GT_B) - Dice(pred_A, GT_A))
         + max(0, m + Dice(pred_B, GT_A) - Dice(pred_B, GT_B))

m = 0.2 (margin)
No L_stability. 5 epochs. Eval every epoch.
"""
import sys, os, time, json, torch
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, "src")

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, infer_with_embedding
from pfa_loss import SegmentationLoss, PromptGroundingLoss
from eval_faithfulness_v2 import evaluate_faithfulness_v2
from pathlib import Path

device = "cuda"
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"

LAMBDA_SEG = 0.7
LAMBDA_SWITCH = 2.0
LAMBDA_GROUND = 0.5
SWITCH_MARGIN = 0.2
TOTAL_EPOCHS = 5
ACCUM_STEPS = 4


def soft_dice(probs, target):
    """Differentiable soft Dice score."""
    probs_flat = probs.view(-1)
    target_flat = target.view(-1).float()
    intersection = (probs_flat * target_flat).sum()
    union = probs_flat.sum() + target_flat.sum()
    return (2.0 * intersection + 1.0) / (union + 1.0)


def switch_loss(pred_A, pred_B, gt_A, gt_B, margin=0.2):
    """Direct switch loss: pred_A should match GT_A not GT_B, and vice versa."""
    probs_A = torch.sigmoid(pred_A)
    probs_B = torch.sigmoid(pred_B)

    dice_A_targetA = soft_dice(probs_A, gt_A)
    dice_A_targetB = soft_dice(probs_A, gt_B)
    dice_B_targetB = soft_dice(probs_B, gt_B)
    dice_B_targetA = soft_dice(probs_B, gt_A)

    # pred_A should prefer GT_A over GT_B by margin
    violation_A = F.relu(margin + dice_A_targetB - dice_A_targetA)
    # pred_B should prefer GT_B over GT_A by margin
    violation_B = F.relu(margin + dice_B_targetA - dice_B_targetB)

    return violation_A + violation_B


def train_one_epoch_switch(model, dataloader, optimizer, device):
    model.train()
    seg_fn = SegmentationLoss()
    ground_fn = PromptGroundingLoss()

    epoch_losses = {k: 0.0 for k in ["total", "seg", "switch", "grounding"]}
    n_batches = 0
    n_switch = 0

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        if batch["primary_organ_id"] == -1:
            continue

        image = batch["image"].to(device)
        gt_A = batch["gt_primary"].to(device)
        point_A = batch["clean_point"].to(device)

        # Forward pass A (compute image embedding once)
        pred_A, img_emb = infer_with_point(model, image, point_A, device)

        # Seg loss for pred_A
        l_seg_A = seg_fn(pred_A, gt_A)

        # Grounding for pred_A
        pt_A_g = point_A.unsqueeze(0) if point_A.dim() == 1 else point_A
        l_ground_A = ground_fn(pred_A, pt_A_g, None)

        # If cross-organ pair available, compute switch loss
        l_switch = torch.tensor(0.0, device=device)
        l_seg_B = torch.tensor(0.0, device=device)
        l_ground_B = torch.tensor(0.0, device=device)
        has_pair = False

        if batch["has_cross"]:
            gt_B = batch["gt_cross"].to(device)
            point_B = batch["cross_point"].to(device)

            if gt_B.sum() > 10:
                # Forward pass B (reuse image embedding)
                pred_B = infer_with_embedding(
                    model, img_emb, point_B, device, image.shape[-1])

                # Seg loss for pred_B
                l_seg_B = seg_fn(pred_B, gt_B)

                # Grounding for pred_B
                pt_B_g = point_B.unsqueeze(0) if point_B.dim() == 1 else point_B
                l_ground_B = ground_fn(pred_B, pt_B_g, None)

                # Switch loss
                l_switch = switch_loss(pred_A, pred_B, gt_A, gt_B, SWITCH_MARGIN)
                has_pair = True
                n_switch += 1

        # Total loss
        if has_pair:
            l_seg = LAMBDA_SEG * (l_seg_A + l_seg_B) / 2.0
        else:
            l_seg = LAMBDA_SEG * l_seg_A

        l_ground = LAMBDA_GROUND * (l_ground_A + l_ground_B) / 2.0 if has_pair else LAMBDA_GROUND * l_ground_A
        total = l_seg + LAMBDA_SWITCH * l_switch + l_ground

        scaled = total / ACCUM_STEPS
        scaled.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_losses["total"] += total.item()
        epoch_losses["seg"] += (l_seg_A.item() + l_seg_B.item()) / (2 if has_pair else 1)
        epoch_losses["switch"] += l_switch.item()
        epoch_losses["grounding"] += l_ground_A.item()
        n_batches += 1

    if n_batches % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    if n_batches > 0:
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

    epoch_losses["switch_pairs_ratio"] = n_switch / max(n_batches, 1)
    return epoch_losses


def main():
    print("=" * 60)
    print("DIRECT SWITCH LOSS TRAINING")
    print(f"L = {LAMBDA_SEG}*seg + {LAMBDA_SWITCH}*switch(m={SWITCH_MARGIN}) + {LAMBDA_GROUND}*ground")
    print("No stability loss. Paired prompt-contrastive.")
    print("=" * 60)

    output_dir = Path("results/switch_loss")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(SAM_ROOT, CKPT, "vit_b_ori", device)
    lora_info = apply_lora_to_sam_med3d(model)
    print(f"LoRA: {lora_info['trainable_params']:,} params ({lora_info['trainable_pct']:.2f}%)")

    train_ds = PEADataset(DATA_DIR, split="Tr", samples_per_volume=4, seed=42)
    val_ds = PEADataset(DATA_DIR, split="Va", samples_per_volume=2, seed=43)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)

    history = []
    best_score = -1

    for epoch in range(1, TOTAL_EPOCHS + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{TOTAL_EPOCHS} ---")

        train_losses = train_one_epoch_switch(model, train_loader, optimizer, device)
        scheduler.step()

        metrics = evaluate_faithfulness_v2(model, val_loader, device, infer_with_point, max_batches=120)

        elapsed = time.time() - t0
        record = {"epoch": epoch, "time": elapsed, **train_losses, **metrics}
        history.append(record)

        switch_acc = metrics.get("switch_accuracy", metrics.get("switch_acc", 0))
        dice = metrics.get("mean_dice", metrics.get("dice", 0))

        print(f"Ep {epoch} | loss={train_losses['total']:.4f} "
              f"(seg={train_losses['seg']:.3f} switch={train_losses['switch']:.3f} "
              f"gnd={train_losses['grounding']:.3f}) "
              f"pairs={train_losses['switch_pairs_ratio']:.2f}")
        print(f"       | Switch={switch_acc:.3f} Dice={dice:.3f} | {elapsed:.0f}s")

        # Save checkpoint
        state = {k: v for k, v in model.state_dict().items()
                 if any(t in k for t in ["lora", "point_embed", "not_a_point",
                                         "mask_downscaling", "output_hyper",
                                         "iou_prediction", "output_upscaling"])}
        torch.save(state, output_dir / f"switch_ep{epoch}.pth")

        # Best by constrained criterion
        score = switch_acc if dice >= 0.55 else switch_acc * 0.5
        if score > best_score:
            best_score = score
            torch.save(state, output_dir / "best_switch.pth")
            print(f"  >> New best! Score={score:.3f}")

    with (output_dir / "switch_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("SWITCH LOSS TRAINING COMPLETE")
    print("=" * 60)
    for r in history:
        sw = r.get("switch_accuracy", r.get("switch_acc", 0))
        d = r.get("mean_dice", r.get("dice", 0))
        print(f"  Ep {r['epoch']}: Switch={sw:.3f} Dice={d:.3f} L_switch={r['switch']:.4f}")

    final = history[-1]
    sw_final = final.get("switch_accuracy", final.get("switch_acc", 0))
    d_final = final.get("mean_dice", final.get("dice", 0))
    print(f"\nGate check: Switch={sw_final:.3f} (need>=0.62), Dice={d_final:.3f} (need>=0.55)")
    if sw_final >= 0.62 and d_final >= 0.55:
        print(">>> PASS - proceed to 3 seeds")
    elif sw_final >= 0.55:
        print(">>> BORDERLINE - try m=0.3 or higher lambda_switch")
    else:
        print(">>> FAIL - pivot to Pareto frontier paper")


if __name__ == "__main__":
    main()
