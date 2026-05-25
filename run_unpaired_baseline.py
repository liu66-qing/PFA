#!/usr/bin/env python3
"""
Unpaired Multi-Prompt Baseline Experiment
==========================================
Key control: same total prompt-target supervision (2 per step),
but the second prompt comes from a DIFFERENT image rather than
being paired on the same image.

This isolates whether "same-image pairing" is the key mechanism
vs just "more prompt exposure per step".
"""
import sys
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg/src")
sys.path.insert(0, "/root/autodl-tmp/PEA-MedSeg")

import os
import json
import time
import torch
import numpy as np
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
from pea_dataset import PEADataset
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, infer_with_embedding
from pfa_loss import SegmentationLoss, PromptGroundingLoss
from eval_faithfulness_v2 import evaluate_faithfulness_v2

DEVICE = "cuda:0"
SEED = 42
EPOCHS = 20
LR = 1e-4
ACCUM_STEPS = 4
SAVE_DIR = "/root/autodl-tmp/PEA-MedSeg/results/formal/unpaired_multi_prompt"
SAM_ROOT = "/root/autodl-tmp/SAM-Med3D"
CKPT = "/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth"
DATA_DIR = "/root/autodl-tmp/data/amos22_cached"

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_with_lora(device):
    sys.path.insert(0, SAM_ROOT)
    model = load_model(SAM_ROOT, CKPT, "vit_b_ori", device)
    apply_lora_to_sam_med3d(model, rank=8)
    model.to(device)
    # Freeze encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    return model

def evaluate_switch(model, val_loader, device):
    model.eval()
    results = evaluate_faithfulness_v2(
        model, val_loader, device, infer_with_point,
        max_batches=60
    )
    model.train()
    return results

seg_loss_fn = None
def seg_fn(pred, gt):
    global seg_loss_fn
    if seg_loss_fn is None:
        seg_loss_fn = SegmentationLoss()
    return seg_loss_fn(pred, gt)

def train_unpaired_epoch(model, dataloader, optimizer, device):
    """
    Unpaired multi-prompt training:
    - Collect all batches, shuffle to create random cross-image pairs
    - For each pair (batch_A, batch_B from different images):
      - Forward pass 1: image_A, prompt_A -> pred_A, supervise with gt_A
      - Forward pass 2: image_B, prompt_B -> pred_B, supervise with gt_B
    - Total loss = (l_seg_A + l_seg_B) / 2
    - Same 2x supervision per step as paired training, but NO same-image contrast
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Collect all batches
    all_batches = []
    for batch in dataloader:
        all_batches.append(batch)

    # Create shuffled indices for second image
    n = len(all_batches)
    second_indices = list(range(n))
    np.random.shuffle(second_indices)
    # Ensure no self-pairing
    for i in range(n):
        if second_indices[i] == i:
            j = (i + 1) % n
            second_indices[i], second_indices[j] = second_indices[j], second_indices[i]

    optimizer.zero_grad()

    for step in range(n):
        batch_A = all_batches[step]
        batch_B = all_batches[second_indices[step]]

        # Forward pass 1: image A
        image_A = batch_A["image"].to(device)
        gt_A = batch_A["gt_primary"].to(device)
        point_A = batch_A["clean_point"].to(device)

        if gt_A.sum() < 10:
            continue

        with torch.no_grad():
            img_emb_A = model.image_encoder(image_A)
        pred_A = infer_with_embedding(model, img_emb_A, point_A, device, image_A.shape[-1])
        l_seg_A = seg_fn(pred_A, gt_A)

        # Forward pass 2: image B (DIFFERENT image)
        image_B = batch_B["image"].to(device)
        gt_B = batch_B["gt_primary"].to(device)
        point_B = batch_B["clean_point"].to(device)

        if gt_B.sum() < 10:
            loss = l_seg_A
        else:
            with torch.no_grad():
                img_emb_B = model.image_encoder(image_B)
            pred_B = infer_with_embedding(model, img_emb_B, point_B, device, image_B.shape[-1])
            l_seg_B = seg_fn(pred_B, gt_B)
            loss = (l_seg_A + l_seg_B) / 2.0

        (loss / ACCUM_STEPS).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    # Final accumulation flush
    if n_batches % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print("UNPAIRED MULTI-PROMPT BASELINE")
    print("Same 2x prompt supervision per step, from DIFFERENT images")
    print("Control for: paired training sees 2x prompts per step")
    print("=" * 60)

    # Load model
    model = load_model_with_lora(DEVICE)

    # Dataset
    train_ds = PEADataset(
        data_dir=DATA_DIR,
        split="Tr",
        crop_size=128,
        samples_per_volume=4,
        seed=SEED
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

    val_ds = PEADataset(
        data_dir=DATA_DIR,
        split="Va",
        crop_size=128,
        samples_per_volume=2,
        seed=43
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.01)

    # LR scheduler: 2-epoch warmup + cosine
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=2)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - 2, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[2])

    # Training loop
    history = []
    best_dice = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss = train_unpaired_epoch(model, train_loader, optimizer, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        # Evaluate every 5 epochs or first/last
        if epoch % 5 == 0 or epoch == EPOCHS or epoch == 1:
            metrics = evaluate_switch(model, val_loader, DEVICE)
            dice_val = metrics["mean_dice"]
            switch_val = metrics["switch_accuracy"]

            record = {
                "epoch": epoch,
                "dice": dice_val,
                "switch_acc": switch_val,
                "loss": avg_loss,
                "time": elapsed
            }
            history.append(record)

            print(f"Epoch {epoch:3d} | Dice={dice_val:.3f} | Switch={switch_val:.3f} | "
                  f"Loss={avg_loss:.4f} | Time={elapsed:.0f}s")

            if dice_val > best_dice:
                best_dice = dice_val
                torch.save(
                    {k: v for k, v in model.state_dict().items()
                     if "lora" in k.lower()},
                    os.path.join(SAVE_DIR, "best_lora.pth")
                )
        else:
            print(f"Epoch {epoch:3d} | Loss={avg_loss:.4f} | Time={elapsed:.0f}s")

    # Save history
    with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE. Best Dice={best_dice:.3f}")
    print(f"Results: {SAVE_DIR}/history.json")
    print("=" * 60)

if __name__ == "__main__":
    main()
