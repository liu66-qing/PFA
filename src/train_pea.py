"""PEA Training Loop.

Trains SAM-Med3D with LoRA + prompt-equivariant consistency loss.
Supports all baselines via loss configuration:
- seg_only: baseline 3 (LoRA, seg loss only)
- jitter_aug: baseline 4a (LoRA + prompt jitter augmentation)
- multi_prompt_sup: baseline 4b (LoRA + multi-prompt supervised)
- cpc_style: baseline 5 (LoRA + CPC-style invariance)
- pea_full: our method (LoRA + full equivariant loss)
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
from src.pea_loss import PEALoss
from src.pea_dataset import PEADataset
from src.sam_med3d_gate1_wrapper import load_model


def infer_with_point(model, image, point, device):
    """Run SAM-Med3D inference with a single point prompt."""
    crop_size = image.shape[-1]
    point_tensor = point.unsqueeze(0).unsqueeze(0).to(device)  # 1×1×3
    label_tensor = torch.ones((1, 1), dtype=torch.long, device=device)
    low_res_logits = torch.zeros(
        (1, 1, crop_size // 4, crop_size // 4, crop_size // 4),
        dtype=torch.float32, device=device)

    image_embedding = model.image_encoder(image.to(device))
    sparse, dense = model.prompt_encoder(
        points=[point_tensor, label_tensor], boxes=None, masks=low_res_logits)
    low_res_masks, iou_pred = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    masks = F.interpolate(low_res_masks, size=(crop_size,) * 3,
                          mode="trilinear", align_corners=False)
    return masks  # 1×1×D×H×W logits


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, config):
    """Train for one epoch."""
    model.train()
    epoch_losses = {"total": 0, "seg": 0, "same_target": 0,
                    "cross_target": 0, "multi_prompt": 0}
    n_batches = 0

    for batch in dataloader:
        if batch["primary_organ_id"] == -1:
            continue

        image = batch["image"].to(device)  # B×1×D×H×W
        gt = batch["gt_primary"].to(device)
        clean_pt = batch["clean_point"]
        noisy_pt = batch["noisy_point"]

        optimizer.zero_grad()

        # Forward: clean prompt
        logits_clean = infer_with_point(model, image, clean_pt, device)

        # Forward: noisy prompt (same-target)
        logits_noisy = None
        if config["use_same_target"]:
            logits_noisy = infer_with_point(model, image, noisy_pt, device)

        # Forward: cross-target prompt
        logits_cross = None
        gt_cross = None
        if config["use_cross_target"] and batch["has_cross"]:
            cross_pt = batch["cross_point"]
            logits_cross = infer_with_point(model, image, cross_pt, device)
            gt_cross = batch["gt_cross"].to(device)

        # Compute loss
        losses = loss_fn(
            logits_clean=logits_clean,
            gt_clean=gt,
            logits_noisy=logits_noisy,
            logits_cross=logits_cross,
            gt_cross=gt_cross,
        )

        losses["total"].backward()
        optimizer.step()

        for k in epoch_losses:
            epoch_losses[k] += losses[k].item()
        n_batches += 1

    if n_batches > 0:
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
    return epoch_losses


def validate(model, dataloader, device):
    """Compute clean-prompt Dice on validation set."""
    model.eval()
    dices = []

    with torch.no_grad():
        for batch in dataloader:
            if batch["primary_organ_id"] == -1:
                continue
            image = batch["image"].to(device)
            gt = batch["gt_primary"].to(device)
            clean_pt = batch["clean_point"]

            logits = infer_with_point(model, image, clean_pt, device)
            pred = (torch.sigmoid(logits) > 0.5).float()

            # Dice
            intersection = (pred * gt).sum()
            union = pred.sum() + gt.sum()
            dice = (2 * intersection / (union + 1e-6)).item()
            dices.append(dice)

    return np.mean(dices) if dices else 0.0


CONFIGS = {
    "seg_only": {
        "use_same_target": False,
        "use_cross_target": False,
        "use_multi_prompt": False,
        "lambda_inv": 0.0,
        "lambda_sens": 0.0,
        "lambda_multi": 0.0,
    },
    "jitter_aug": {
        "use_same_target": False,
        "use_cross_target": False,
        "use_multi_prompt": False,
        "lambda_inv": 0.0,
        "lambda_sens": 0.0,
        "lambda_multi": 0.0,
        # Jitter is applied in dataset, loss is just seg
    },
    "multi_prompt_sup": {
        "use_same_target": False,
        "use_cross_target": True,  # uses cross-target but only seg loss
        "use_multi_prompt": False,
        "lambda_inv": 0.0,
        "lambda_sens": 0.0,  # no dissimilarity, just supervised on cross prompt
        "lambda_multi": 0.0,
    },
    "cpc_style": {
        "use_same_target": True,
        "use_cross_target": False,
        "use_multi_prompt": False,
        "lambda_inv": 1.0,
        "lambda_sens": 0.0,
        "lambda_multi": 0.0,
    },
    "pea_full": {
        "use_same_target": True,
        "use_cross_target": True,
        "use_multi_prompt": True,
        "lambda_inv": 1.0,
        "lambda_sens": 0.5,
        "lambda_multi": 0.3,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="pea_full")
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/amos22_raw/amos22")
    parser.add_argument("--output-dir", default="results/training")
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
    args = parser.parse_args()

    config = CONFIGS[args.config]
    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / args.config
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training config: {args.config}")
    print(f"Loss weights: inv={config['lambda_inv']}, sens={config['lambda_sens']}, "
          f"multi={config['lambda_multi']}")

    # Load model and apply LoRA
    model = load_model(
        "/root/autodl-tmp/SAM-Med3D",
        "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth",
        "vit_b_ori", device)

    lora_info = apply_lora_to_sam_med3d(
        model, rank=args.lora_rank, alpha=args.lora_alpha,
        target_modules=args.lora_target)
    print(f"LoRA applied: {lora_info['trainable_params']:,} trainable params "
          f"({lora_info['trainable_pct']:.2f}%)")

    # Dataset
    train_dataset = PEADataset(
        args.data_dir, split="Tr", samples_per_volume=4, seed=args.seed)
    val_dataset = PEADataset(
        args.data_dir, split="Va", samples_per_volume=1, seed=args.seed + 1)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # Loss and optimizer
    loss_fn = PEALoss(
        lambda_inv=config["lambda_inv"],
        lambda_sens=config["lambda_sens"],
        lambda_multi=config["lambda_multi"],
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01)

    # Training loop
    best_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device, config)
        val_dice = validate(model, val_loader, device)
        elapsed = time.time() - t0

        record = {"epoch": epoch, "val_dice": val_dice, "time": elapsed, **train_losses}
        history.append(record)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={train_losses['total']:.4f} "
              f"(seg={train_losses['seg']:.4f} same={train_losses['same_target']:.4f} "
              f"cross={train_losses['cross_target']:.4f}) | "
              f"val_dice={val_dice:.4f} | {elapsed:.1f}s")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {k: v for k, v in model.state_dict().items()
                 if "lora" in k or v.requires_grad},
                output_dir / "best_lora.pth")

        if epoch % args.save_every == 0:
            torch.save(
                {k: v for k, v in model.state_dict().items()
                 if "lora" in k or v.requires_grad},
                output_dir / f"lora_epoch{epoch:03d}.pth")

    # Save history
    with (output_dir / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val Dice: {best_dice:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
