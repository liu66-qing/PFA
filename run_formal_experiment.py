"""Formal 50-epoch Direct Switch Loss experiment for AAAI 2027.

Configuration (locked after 3-round adversarial review with Codex):
  L = 1.0 * L_seg + 1.0 * L_switch(m=0.2) + 0.5 * L_ground
  Epochs: 50
  LR: 1e-4, warmup 2 epochs linear + cosine decay to 1e-6
  Eval: fixed prompt bank, every epoch
  Checkpoint selection: max Dice s.t. Switch >= 0.90
  Seeds: 42, 123, 7

Usage:
  python run_formal_experiment.py --method switch_loss --seed 42
  python run_formal_experiment.py --method seg_only --seed 42
"""
import argparse
import sys
import os
import time
import json
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, "src")

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, infer_with_embedding
from pfa_loss import SegmentationLoss, PromptGroundingLoss
from eval_faithfulness_v2 import evaluate_faithfulness_v2

# ============================================================
# CONSTANTS (do not change between runs)
# ============================================================
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"

TOTAL_EPOCHS = 50
WARMUP_EPOCHS = 2
BASE_LR = 1e-4
ETA_MIN = 1e-6
ACCUM_STEPS = 4
EVAL_BATCHES = 120

# Loss weights
CONFIGS = {
    "switch_loss": {
        "lambda_seg": 1.0,
        "lambda_switch": 1.0,
        "lambda_ground": 0.5,
        "switch_margin": 0.2,
    },
    "seg_only": {
        "lambda_seg": 1.0,
        "lambda_switch": 0.0,
        "lambda_ground": 0.0,
        "switch_margin": 0.2,
    },
    # Single prompt + grounding (no paired, no switch)
    "single_ground": {
        "lambda_seg": 1.0,
        "lambda_switch": 0.0,
        "lambda_ground": 0.5,
        "switch_margin": 0.2,
        "force_single": True,
    },
    # Ablations
    "margin_01": {
        "lambda_seg": 1.0,
        "lambda_switch": 1.0,
        "lambda_ground": 0.5,
        "switch_margin": 0.1,
    },
    "margin_03": {
        "lambda_seg": 1.0,
        "lambda_switch": 1.0,
        "lambda_ground": 0.5,
        "switch_margin": 0.3,
    },
    "no_ground": {
        "lambda_seg": 1.0,
        "lambda_switch": 1.0,
        "lambda_ground": 0.0,
        "switch_margin": 0.2,
    },
    # Pareto sweep
    "switch_025": {
        "lambda_seg": 1.0,
        "lambda_switch": 0.25,
        "lambda_ground": 0.5,
        "switch_margin": 0.2,
    },
    "switch_05": {
        "lambda_seg": 1.0,
        "lambda_switch": 0.5,
        "lambda_ground": 0.5,
        "switch_margin": 0.2,
    },
    "switch_20": {
        "lambda_seg": 1.0,
        "lambda_switch": 2.0,
        "lambda_ground": 0.5,
        "switch_margin": 0.2,
    },
    # Mechanism ablation: paired training with seg loss on both prompts, NO switch loss
    "paired_seg_only": {
        "lambda_seg": 1.0,
        "lambda_switch": 0.0,
        "lambda_ground": 0.5,
        "switch_margin": 0.2,
    },
    # Mechanism ablation: paired training, NO ground, NO switch
    "paired_no_ground": {
        "lambda_seg": 1.0,
        "lambda_switch": 0.0,
        "lambda_ground": 0.0,
        "switch_margin": 0.2,
        "force_paired": True,
    },
}

SEEDS = [42, 123, 7]


# ============================================================
# LR Schedule: linear warmup + cosine decay
# ============================================================
class WarmupCosineScheduler:
    """Linear warmup for warmup_epochs, then cosine decay to eta_min."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, eta_min):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + np.cos(np.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ============================================================
# Switch Loss
# ============================================================
def soft_dice(probs, target):
    """Differentiable soft Dice score."""
    probs_flat = probs.view(-1)
    target_flat = target.view(-1).float()
    intersection = (probs_flat * target_flat).sum()
    union = probs_flat.sum() + target_flat.sum()
    return (2.0 * intersection + 1.0) / (union + 1.0)


def compute_switch_loss(pred_A, pred_B, gt_A, gt_B, margin=0.2):
    """Direct switch loss: paired prompt-contrastive."""
    probs_A = torch.sigmoid(pred_A)
    probs_B = torch.sigmoid(pred_B)

    dice_A_targetA = soft_dice(probs_A, gt_A)
    dice_A_targetB = soft_dice(probs_A, gt_B)
    dice_B_targetB = soft_dice(probs_B, gt_B)
    dice_B_targetA = soft_dice(probs_B, gt_A)

    violation_A = F.relu(margin + dice_A_targetB - dice_A_targetA)
    violation_B = F.relu(margin + dice_B_targetA - dice_B_targetB)

    return violation_A + violation_B


# ============================================================
# Training loop
# ============================================================
def train_one_epoch(model, dataloader, optimizer, config, device):
    """Train one epoch with configurable loss weights."""
    model.train()
    seg_fn = SegmentationLoss()
    ground_fn = PromptGroundingLoss()

    losses = {"total": 0.0, "seg": 0.0, "switch": 0.0, "ground": 0.0}
    n_batches = 0
    n_pairs = 0

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        if batch["primary_organ_id"] == -1:
            continue

        image = batch["image"].to(device)
        gt_A = batch["gt_primary"].to(device)
        point_A = batch["clean_point"].to(device)

        # Forward A
        pred_A, img_emb = infer_with_point(model, image, point_A, device)

        # Seg loss A
        l_seg_A = seg_fn(pred_A, gt_A)

        # Grounding A
        pt_A_g = point_A.unsqueeze(0) if point_A.dim() == 1 else point_A
        l_ground_A = ground_fn(pred_A, pt_A_g, None)

        # Switch loss (if cross-organ pair available)
        l_switch = torch.tensor(0.0, device=device)
        l_seg_B = torch.tensor(0.0, device=device)
        l_ground_B = torch.tensor(0.0, device=device)
        has_pair = False

        # Do paired forward if any paired-related config is active
        is_paired = config.get("lambda_switch", 0) > 0 or config.get("lambda_ground", 0) > 0 or config.get("force_paired", False)
        if config.get("force_single", False):
            is_paired = False
        do_paired = batch["has_cross"] and is_paired
        if do_paired:
            gt_B = batch["gt_cross"].to(device)
            point_B = batch["cross_point"].to(device)

            if gt_B.sum() > 10:
                pred_B = infer_with_embedding(
                    model, img_emb, point_B, device, image.shape[-1])
                l_seg_B = seg_fn(pred_B, gt_B)
                pt_B_g = point_B.unsqueeze(0) if point_B.dim() == 1 else point_B
                l_ground_B = ground_fn(pred_B, pt_B_g, None)
                l_switch = compute_switch_loss(
                    pred_A, pred_B, gt_A, gt_B, config["switch_margin"])
                has_pair = True
                n_pairs += 1

        # Weighted total
        if has_pair:
            l_seg = config["lambda_seg"] * (l_seg_A + l_seg_B) / 2.0
            l_ground = config["lambda_ground"] * (l_ground_A + l_ground_B) / 2.0
        else:
            l_seg = config["lambda_seg"] * l_seg_A
            l_ground = config["lambda_ground"] * l_ground_A

        total = l_seg + config["lambda_switch"] * l_switch + l_ground

        (total / ACCUM_STEPS).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

        losses["total"] += total.item()
        losses["seg"] += l_seg_A.item()
        losses["switch"] += l_switch.item()
        losses["ground"] += l_ground_A.item()
        n_batches += 1

    # Final accumulation
    if n_batches % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

    if n_batches > 0:
        for k in losses:
            losses[k] /= n_batches
    losses["pair_ratio"] = n_pairs / max(n_batches, 1)
    return losses


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(CONFIGS.keys()), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    config = CONFIGS[args.method]
    device = torch.device(args.device)

    output_dir = Path(f"results/formal/{args.method}_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    run_config = {
        "method": args.method,
        "seed": args.seed,
        "epochs": args.epochs,
        "config": config,
        "lr": BASE_LR,
        "warmup": WARMUP_EPOCHS,
        "accum_steps": ACCUM_STEPS,
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    print("=" * 60)
    print(f"FORMAL EXPERIMENT: {args.method} (seed={args.seed})")
    print(f"L = {config['lambda_seg']}*seg + {config['lambda_switch']}*switch"
          f"(m={config['switch_margin']}) + {config['lambda_ground']}*ground")
    print(f"Epochs: {args.epochs}, LR: {BASE_LR}, Warmup: {WARMUP_EPOCHS}")
    print("=" * 60)

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Model + LoRA
    model = load_model(SAM_ROOT, CKPT, "vit_b_ori", device)
    lora_info = apply_lora_to_sam_med3d(model)
    print(f"LoRA: {lora_info['trainable_params']:,} params "
          f"({lora_info['trainable_pct']:.2f}%)")

    # Data (fixed seed for reproducibility)
    data_dir = args.data_dir
    train_ds = PEADataset(data_dir, split="Tr", samples_per_volume=4, seed=args.seed)
    val_ds = PEADataset(data_dir, split="Va", samples_per_volume=2, seed=43)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=BASE_LR, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(
        optimizer, WARMUP_EPOCHS, args.epochs, BASE_LR, ETA_MIN)

    # Training loop
    history = []
    best_dice_constrained = -1
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = scheduler.step()

        train_losses = train_one_epoch(model, train_loader, optimizer, config, device)

        # Evaluate
        metrics = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_faithfulness_v2(
                model, val_loader, device, infer_with_point,
                max_batches=EVAL_BATCHES)

        elapsed = time.time() - t0
        record = {"epoch": epoch, "lr": lr, "time": elapsed,
                  **train_losses, **metrics}
        history.append(record)

        switch_acc = metrics.get("switch_accuracy", metrics.get("switch_acc", 0))
        dice = metrics.get("mean_dice", metrics.get("dice", 0))

        print(f"Ep {epoch:3d}/{args.epochs} | "
              f"loss={train_losses['total']:.4f} "
              f"(seg={train_losses['seg']:.3f} sw={train_losses['switch']:.3f} "
              f"gnd={train_losses['ground']:.3f}) "
              f"lr={lr:.2e}")
        if metrics:
            print(f"         | Switch={switch_acc:.3f} Dice={dice:.3f} | {elapsed:.0f}s")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {k: v for k, v in model.state_dict().items()
                     if any(t in k for t in ["lora", "point_embed", "not_a_point",
                                             "mask_downscaling", "output_hyper",
                                             "iou_prediction", "output_upscaling"])}
            torch.save(state, output_dir / f"epoch{epoch:03d}.pth")

        # Best checkpoint: max Dice s.t. Switch >= 0.90
        if metrics and switch_acc >= 0.90 and dice > best_dice_constrained:
            best_dice_constrained = dice
            best_epoch = epoch
            state = {k: v for k, v in model.state_dict().items()
                     if any(t in k for t in ["lora", "point_embed", "not_a_point",
                                             "mask_downscaling", "output_hyper",
                                             "iou_prediction", "output_upscaling"])}
            torch.save(state, output_dir / "best.pth")
            print(f"  >> New best! Dice={dice:.3f} Switch={switch_acc:.3f} (ep{epoch})")

    # Save history
    with (output_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print(f"COMPLETE: {args.method} seed={args.seed}")
    print("=" * 60)
    print(f"Best constrained (Switch>=0.90): Dice={best_dice_constrained:.3f} at ep{best_epoch}")
    final = history[-1]
    sw_f = final.get("switch_accuracy", final.get("switch_acc", 0))
    d_f = final.get("mean_dice", final.get("dice", 0))
    print(f"Final epoch: Switch={sw_f:.3f} Dice={d_f:.3f}")

    # Epoch curve summary (every 10)
    print("\nEpoch curve:")
    for r in history:
        if r["epoch"] % 10 == 0 or r["epoch"] == 1:
            sw = r.get("switch_accuracy", r.get("switch_acc", 0))
            d = r.get("mean_dice", r.get("dice", 0))
            if sw > 0 or d > 0:
                print(f"  Ep {r['epoch']:3d}: Switch={sw:.3f} Dice={d:.3f}")


if __name__ == "__main__":
    main()
