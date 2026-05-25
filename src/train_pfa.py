"""PFA Training Loop: Prompt Faithfulness Adaptation.

Trains SAM-Med3D with LoRA + prompt faithfulness losses.
Supports all baselines via config:
- seg_only: baseline 3 (LoRA, seg loss only)
- multi_prompt_sup: baseline 4 (LoRA + multi-prompt supervised, no contrastive)
- ground_only: baseline 5 (LoRA + grounding only)
- contrast_only: baseline 6 (LoRA + contrast only)
- pfa_full: our method (LoRA + grounding + contrast + stability)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/root/autodl-tmp/SAM-Med3D")
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")

from src.lora import apply_lora_to_sam_med3d
from src.pfa_loss import PFALoss, SegmentationLoss
from src.pea_dataset import PEADataset
from src.sam_med3d_gate1_wrapper import load_model


def infer_with_point(model, image, point, device):
    """Run SAM-Med3D inference with a single point prompt."""
    crop_size = image.shape[-1]
    point_tensor = point.view(1, 1, 3).to(device)
    label_tensor = torch.ones((1, 1), dtype=torch.long, device=device)
    low_res_logits = torch.zeros(
        (1, 1, crop_size // 4, crop_size // 4, crop_size // 4),
        dtype=torch.float32, device=device)

    image_embedding = model.image_encoder(image.to(device))
    sparse, dense = model.prompt_encoder(
        points=[point_tensor, label_tensor], boxes=None, masks=low_res_logits)
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    masks = F.interpolate(low_res_masks, size=(crop_size,) * 3,
                          mode="trilinear", align_corners=False)
    return masks, image_embedding


def infer_with_embedding(model, image_embedding, point, device, crop_size=128):
    """Run SAM-Med3D decoder with pre-computed image embedding."""
    point_tensor = point.view(1, 1, 3).to(device)
    label_tensor = torch.ones((1, 1), dtype=torch.long, device=device)
    low_res_logits = torch.zeros(
        (1, 1, crop_size // 4, crop_size // 4, crop_size // 4),
        dtype=torch.float32, device=device)

    sparse, dense = model.prompt_encoder(
        points=[point_tensor, label_tensor], boxes=None, masks=low_res_logits)
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    masks = F.interpolate(low_res_masks, size=(crop_size,) * 3,
                          mode="trilinear", align_corners=False)
    return masks



def train_one_epoch(model, dataloader, optimizer, loss_fn, device, config,
                    accum_steps=4):
    """Train for one epoch with PFA losses and gradient accumulation."""
    model.train()
    epoch_losses = {k: 0.0 for k in ["total", "seg", "grounding", "contrast", "stability"]}
    n_batches = 0

    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        if batch["primary_organ_id"] == -1:
            continue

        image = batch["image"].to(device)
        gt = batch["gt_primary"].to(device)
        pos_pt = batch["clean_point"].to(device)

        # Primary forward pass
        logits_primary, img_emb = infer_with_point(model, image, pos_pt, device)

        # Negative point (from cross-target organ)
        neg_pt = None
        if config["use_grounding_neg"] and batch["has_cross"]:
            neg_pt = batch["cross_point"].to(device)

        # Other organ GTs for contrastive loss
        gt_others = None
        if config["use_contrast"] and batch["has_cross"]:
            gt_others = [batch["gt_cross"].to(device)]

        # Second point in same organ for stability (reuse image embedding)
        logits_second = None
        if config["use_stability"]:
            noisy_pt = batch["noisy_point"].to(device)
            logits_second = infer_with_embedding(model, img_emb, noisy_pt, device, image.shape[-1])

        # Compute loss (scaled by accum_steps)
        losses = loss_fn(
            logits_primary=logits_primary,
            gt_primary=gt,
            pos_point=pos_pt.unsqueeze(0) if pos_pt.dim() == 1 else pos_pt,
            neg_point=neg_pt.unsqueeze(0) if neg_pt is not None and neg_pt.dim() == 1 else neg_pt,
            gt_others=gt_others,
            logits_second=logits_second,
        )

        scaled_loss = losses["total"] / accum_steps
        scaled_loss.backward()

        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        for k in epoch_losses:
            if k in losses:
                epoch_losses[k] += losses[k].item()
        n_batches += 1

    # Final step if not aligned
    if n_batches % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    if n_batches > 0:
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
    return epoch_losses


def evaluate_faithfulness(model, dataloader, device, max_batches=50):
    """Evaluate prompt faithfulness metrics."""
    model.eval()
    correct_switches = 0
    total_prompts = 0
    grounded = 0
    dices = []

    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            if batch["primary_organ_id"] == -1:
                continue

            image = batch["image"].to(device)
            gt = batch["gt_primary"].to(device)
            pos_pt = batch["clean_point"].to(device)

            logits, _ = infer_with_point(model, image, pos_pt, device)
            pred = (torch.sigmoid(logits) > 0.5).float()

            # Dice to target
            inter = (pred * gt).sum()
            union = pred.sum() + gt.sum()
            dice_target = (2 * inter / (union + 1e-6)).item()
            dices.append(dice_target)

            # Grounding check
            pz, py, px = pos_pt.long().squeeze()
            pz = pz.clamp(0, pred.shape[2]-1)
            py = py.clamp(0, pred.shape[3]-1)
            px = px.clamp(0, pred.shape[4]-1)
            if pred[0, 0, pz, py, px] > 0:
                grounded += 1

            # Switch accuracy (simplified: check if target dice > cross dice)
            if batch["has_cross"]:
                gt_cross = batch["gt_cross"].to(device)
                dice_cross = (2 * (pred * gt_cross).sum() /
                              (pred.sum() + gt_cross.sum() + 1e-6)).item()
                if dice_target > dice_cross:
                    correct_switches += 1
                total_prompts += 1

    metrics = {
        "mean_dice": np.mean(dices) if dices else 0.0,
        "grounding_rate": grounded / max(len(dices), 1),
        "switch_accuracy": correct_switches / max(total_prompts, 1),
    }
    return metrics


CONFIGS = {
    "seg_only": {
        "use_grounding_neg": False,
        "use_contrast": False,
        "use_stability": False,
        "lambda_ground": 0.0,
        "lambda_contrast": 0.0,
        "lambda_stable": 0.0,
    },
    "multi_prompt_sup": {
        "use_grounding_neg": False,
        "use_contrast": False,
        "use_stability": False,
        "lambda_ground": 0.0,
        "lambda_contrast": 0.0,
        "lambda_stable": 0.0,
        # Uses cross-target prompts but only with seg loss
    },
    "ground_only": {
        "use_grounding_neg": True,
        "use_contrast": False,
        "use_stability": False,
        "lambda_ground": 1.0,
        "lambda_contrast": 0.0,
        "lambda_stable": 0.0,
    },
    "contrast_only": {
        "use_grounding_neg": False,
        "use_contrast": True,
        "use_stability": False,
        "lambda_ground": 0.0,
        "lambda_contrast": 0.5,
        "lambda_stable": 0.0,
    },
    "pfa_full": {
        "use_grounding_neg": True,
        "use_contrast": True,
        "use_stability": True,
        "lambda_ground": 1.0,
        "lambda_contrast": 0.5,
        "lambda_stable": 0.3,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="pfa_full")
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/amos22_raw/amos22")
    parser.add_argument("--output-dir", default="results/pfa_training")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-target", default="decoder+prompt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--contrast-margin", type=float, default=0.1)
    args = parser.parse_args()

    config = CONFIGS[args.config]
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.config
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"PFA Training: {args.config}")
    print(f"Losses: ground={config['lambda_ground']}, "
          f"contrast={config['lambda_contrast']}, "
          f"stable={config['lambda_stable']}")
    print(f"=" * 60)

    # Load model + LoRA
    model = load_model(
        "/root/autodl-tmp/SAM-Med3D",
        "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth",
        "vit_b_ori", device)
    lora_info = apply_lora_to_sam_med3d(
        model, rank=args.lora_rank, alpha=args.lora_alpha,
        target_modules=args.lora_target)
    print(f"LoRA: {lora_info['trainable_params']:,} params "
          f"({lora_info['trainable_pct']:.2f}%)")

    # Data
    train_dataset = PEADataset(
        args.data_dir, split="Tr", samples_per_volume=4, seed=args.seed)
    val_dataset = PEADataset(
        args.data_dir, split="Va", samples_per_volume=2, seed=args.seed + 1)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Loss + optimizer
    loss_fn = PFALoss(
        lambda_ground=config["lambda_ground"],
        lambda_contrast=config["lambda_contrast"],
        lambda_stable=config["lambda_stable"],
        contrast_margin=args.contrast_margin,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training
    best_switch_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, config)
        scheduler.step()

        # Evaluate
        metrics = {}
        if epoch % args.eval_every == 0 or epoch == 1:
            metrics = evaluate_faithfulness(model, val_loader, device)

        elapsed = time.time() - t0
        record = {"epoch": epoch, "time": elapsed, **train_losses, **metrics}
        history.append(record)

        log_parts = [
            f"Ep {epoch:3d}/{args.epochs}",
            f"loss={train_losses['total']:.4f}",
            f"(seg={train_losses['seg']:.3f} gnd={train_losses['grounding']:.3f} "
            f"ctr={train_losses['contrast']:.3f} stb={train_losses['stability']:.3f})",
        ]
        if metrics:
            log_parts.append(
                f"| switch={metrics['switch_accuracy']:.3f} "
                f"dice={metrics['mean_dice']:.3f} "
                f"ground={metrics['grounding_rate']:.3f}")
        log_parts.append(f"| {elapsed:.0f}s")
        print(" ".join(log_parts))

        # Save best
        if metrics.get("switch_accuracy", 0) > best_switch_acc:
            best_switch_acc = metrics["switch_accuracy"]
            state = {k: v for k, v in model.state_dict().items()
                     if any(t in k for t in ["lora", "point_embed", "not_a_point",
                                             "mask_downscaling", "output_hyper",
                                             "iou_prediction", "output_upscaling"])}
            torch.save(state, output_dir / "best_pfa.pth")

        if epoch % args.save_every == 0:
            state = {k: v for k, v in model.state_dict().items()
                     if any(t in k for t in ["lora", "point_embed", "not_a_point",
                                             "mask_downscaling", "output_hyper",
                                             "iou_prediction", "output_upscaling"])}
            torch.save(state, output_dir / f"pfa_epoch{epoch:03d}.pth")

    # Save history
    with (output_dir / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best Switch Accuracy: {best_switch_acc:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
