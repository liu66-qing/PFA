"""Curriculum PFA Training: Phase-based loss scheduling.

Recipe from adversarial review (Codex debate 2026-05-21):
  Phase 1 (ep 1-2): 0.2*L_seg + 1.0*L_ground + 1.5*L_contrast + 0.5*L_stability
  Phase 2 (ep 3-5): 0.7*L_seg + 1.0*L_ground + 2.0*L_contrast + 0.25*L_stability

Hard switch at epoch 3. Same LR as pfa_full. Eval every epoch.
Success: Switch>=0.62, Dice>=0.55
"""
import sys, os, time, json, torch
import numpy as np
sys.path.insert(0, "src")

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, infer_with_embedding
from pfa_loss import SegmentationLoss, PromptGroundingLoss, SemanticContrastLoss, IntraOrganStabilityLoss
from eval_faithfulness_v2 import evaluate_faithfulness_v2
from pathlib import Path
import torch.nn.functional as F

device = "cuda"
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"

PHASE1_WEIGHTS = {"seg": 0.2, "ground": 1.0, "contrast": 1.5, "stability": 0.5}
PHASE2_WEIGHTS = {"seg": 0.7, "ground": 1.0, "contrast": 2.0, "stability": 0.25}
PHASE_SWITCH_EPOCH = 3
TOTAL_EPOCHS = 5
ACCUM_STEPS = 4


def get_weights(epoch):
    if epoch < PHASE_SWITCH_EPOCH:
        return PHASE1_WEIGHTS
    return PHASE2_WEIGHTS


def train_one_epoch_curriculum(model, dataloader, optimizer, epoch, device):
    model.train()
    weights = get_weights(epoch)

    seg_fn = SegmentationLoss()
    ground_fn = PromptGroundingLoss()
    contrast_fn = SemanticContrastLoss(margin=0.1)
    stability_fn = IntraOrganStabilityLoss()

    epoch_losses = {k: 0.0 for k in ["total", "seg", "grounding", "contrast", "stability"]}
    n_batches = 0

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        if batch["primary_organ_id"] == -1:
            continue

        image = batch["image"].to(device)
        gt = batch["gt_primary"].to(device)
        pos_pt = batch["clean_point"].to(device)

        logits_primary, img_emb = infer_with_point(model, image, pos_pt, device)

        l_seg = seg_fn(logits_primary, gt)

        neg_pt = None
        if batch["has_cross"]:
            neg_pt = batch["cross_point"].to(device)
            if neg_pt.dim() == 1:
                neg_pt = neg_pt.unsqueeze(0)
        pos_pt_g = pos_pt.unsqueeze(0) if pos_pt.dim() == 1 else pos_pt
        l_ground = ground_fn(logits_primary, pos_pt_g, neg_pt)

        l_contrast = torch.tensor(0.0, device=device)
        if batch["has_cross"]:
            gt_cross = batch["gt_cross"].to(device)
            if gt_cross.sum() > 10:
                l_contrast = contrast_fn(logits_primary, gt, [gt_cross])

        l_stability = torch.tensor(0.0, device=device)
        noisy_pt = batch["noisy_point"].to(device)
        logits_second = infer_with_embedding(model, img_emb, noisy_pt, device, image.shape[-1])
        l_stability = stability_fn(logits_primary, logits_second)

        total = (weights["seg"] * l_seg
                 + weights["ground"] * l_ground
                 + weights["contrast"] * l_contrast
                 + weights["stability"] * l_stability)

        scaled = total / ACCUM_STEPS
        scaled.backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_losses["total"] += total.item()
        epoch_losses["seg"] += l_seg.item()
        epoch_losses["grounding"] += l_ground.item()
        epoch_losses["contrast"] += l_contrast.item()
        epoch_losses["stability"] += l_stability.item()
        n_batches += 1

    if n_batches % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    if n_batches > 0:
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
    return epoch_losses


def main():
    print("=" * 60)
    print("CURRICULUM PFA TRAINING")
    print(f"Phase 1 (ep 1-2): {PHASE1_WEIGHTS}")
    print(f"Phase 2 (ep 3-5): {PHASE2_WEIGHTS}")
    print("=" * 60)

    output_dir = Path("results/curriculum_pfa")
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
        weights = get_weights(epoch)
        phase = 1 if epoch < PHASE_SWITCH_EPOCH else 2
        print(f"\n--- Epoch {epoch} (Phase {phase}) weights: {weights} ---")

        train_losses = train_one_epoch_curriculum(model, train_loader, optimizer, epoch, device)
        scheduler.step()

        metrics = evaluate_faithfulness_v2(model, val_loader, device, infer_with_point, max_batches=120)

        elapsed = time.time() - t0
        record = {
            "epoch": epoch, "phase": phase, "weights": weights,
            "time": elapsed, **train_losses, **metrics
        }
        history.append(record)

        switch_acc = metrics.get("switch_accuracy", metrics.get("switch_acc", 0))
        dice = metrics.get("mean_dice", metrics.get("dice", 0))

        print(f"Ep {epoch} | loss={train_losses['total']:.4f} "
              f"(seg={train_losses['seg']:.3f} gnd={train_losses['grounding']:.3f} "
              f"ctr={train_losses['contrast']:.3f} stb={train_losses['stability']:.3f})")
        print(f"       | Switch={switch_acc:.3f} Dice={dice:.3f} | {elapsed:.0f}s")

        state = {k: v for k, v in model.state_dict().items()
                 if any(t in k for t in ["lora", "point_embed", "not_a_point",
                                         "mask_downscaling", "output_hyper",
                                         "iou_prediction", "output_upscaling"])}
        torch.save(state, output_dir / f"curriculum_ep{epoch}.pth")

        score = switch_acc if dice >= 0.55 else switch_acc * 0.5
        if score > best_score:
            best_score = score
            torch.save(state, output_dir / "best_curriculum.pth")
            print(f"  >> New best! Score={score:.3f}")

    with (output_dir / "curriculum_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE")
    print("=" * 60)
    for r in history:
        sw = r.get("switch_accuracy", r.get("switch_acc", 0))
        d = r.get("mean_dice", r.get("dice", 0))
        print(f"  Ep {r['epoch']} (Phase {r['phase']}): Switch={sw:.3f} Dice={d:.3f}")

    final = history[-1]
    sw_final = final.get("switch_accuracy", final.get("switch_acc", 0))
    d_final = final.get("mean_dice", final.get("dice", 0))
    print(f"\nGate 2 check: Switch={sw_final:.3f} (need>=0.62), Dice={d_final:.3f} (need>=0.55)")
    if sw_final >= 0.62 and d_final >= 0.55:
        print(">>> PASS - proceed to 3 seeds")
    elif sw_final >= 0.55:
        print(">>> BORDERLINE - try fallback recipe (seg=0.5, contrast=2.5)")
    else:
        print(">>> FAIL - try parameter separation")


if __name__ == "__main__":
    main()
