"""Corrected prompt faithfulness evaluation.

True Switch Accuracy: 
  For each cross-target pair (organ A, organ B):
  1. Prompt for A -> pred_A. Check: Dice(pred_A, GT_A) > Dice(pred_A, GT_B)
  2. Prompt for B -> pred_B. Check: Dice(pred_B, GT_B) > Dice(pred_B, GT_A)
  Both must be correct for a 'switch' to count as successful.

Also reports:
  - Target Margin: Dice(pred, prompted_organ) - max(Dice(pred, other_organs))
  - Grounding Rate: prompt point inside predicted mask
  - Mean Dice: segmentation quality
  - Mask Volume Ratio: pred volume / GT volume (detects tiny blobs)
"""

import numpy as np
import torch
import torch.nn.functional as F


def evaluate_faithfulness_v2(model, dataloader, device, infer_fn, max_batches=50):
    """Corrected faithfulness evaluation with true switch test."""
    model.eval()
    
    results = {
        'switch_correct': 0,
        'switch_total': 0,
        'grounded': 0,
        'grounded_total': 0,
        'dices': [],
        'target_margins': [],
        'volume_ratios': [],
    }
    
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            if batch['primary_organ_id'] == -1:
                continue
            
            image = batch['image'].to(device)
            gt_primary = batch['gt_primary'].to(device)
            pos_pt = batch['clean_point'].to(device)
            
            # Forward pass with primary prompt
            logits_A, img_emb = infer_fn(model, image, pos_pt, device)
            pred_A = (torch.sigmoid(logits_A) > 0.5).float()
            
            # Dice with target
            dice_A_target = dice_score(pred_A, gt_primary)
            results['dices'].append(dice_A_target)
            
            # Volume ratio (detect tiny blobs)
            pred_vol = pred_A.sum().item()
            gt_vol = gt_primary.sum().item()
            if gt_vol > 0:
                results['volume_ratios'].append(pred_vol / gt_vol)
            
            # Grounding check (3x3x3 ball around prompt)
            pz, py, px = pos_pt.long().squeeze().tolist()
            pz = max(0, min(pz, pred_A.shape[2]-1))
            py = max(0, min(py, pred_A.shape[3]-1))
            px = max(0, min(px, pred_A.shape[4]-1))
            # Check 3x3x3 neighborhood
            z_lo, z_hi = max(0, pz-1), min(pred_A.shape[2], pz+2)
            y_lo, y_hi = max(0, py-1), min(pred_A.shape[3], py+2)
            x_lo, x_hi = max(0, px-1), min(pred_A.shape[4], px+2)
            ball = pred_A[0, 0, z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
            if ball.sum() > 0:
                results['grounded'] += 1
            results['grounded_total'] += 1
            
            # TRUE Switch Accuracy: need cross-target
            if batch['has_cross']:
                gt_cross = batch['gt_cross'].to(device)
                cross_pt = batch['cross_point'].to(device)
                
                # Forward pass with cross prompt (different organ)
                logits_B, _ = infer_fn(model, image, cross_pt, device)
                pred_B = (torch.sigmoid(logits_B) > 0.5).float()
                
                # Check A: pred_A should match primary more than cross
                dice_A_primary = dice_score(pred_A, gt_primary)
                dice_A_cross = dice_score(pred_A, gt_cross)
                
                # Check B: pred_B should match cross more than primary
                dice_B_cross = dice_score(pred_B, gt_cross)
                dice_B_primary = dice_score(pred_B, gt_primary)
                
                # Both directions must be correct
                A_correct = dice_A_primary > dice_A_cross
                B_correct = dice_B_cross > dice_B_primary
                
                if A_correct and B_correct:
                    results['switch_correct'] += 1
                results['switch_total'] += 1
                
                # Target margin for A
                margin_A = dice_A_primary - dice_A_cross
                results['target_margins'].append(margin_A)
                
                # Target margin for B
                margin_B = dice_B_cross - dice_B_primary
                results['target_margins'].append(margin_B)
    
    metrics = {
        'switch_accuracy': results['switch_correct'] / max(results['switch_total'], 1),
        'mean_dice': np.mean(results['dices']) if results['dices'] else 0.0,
        'grounding_rate': results['grounded'] / max(results['grounded_total'], 1),
        'target_margin': np.mean(results['target_margins']) if results['target_margins'] else 0.0,
        'mean_volume_ratio': np.mean(results['volume_ratios']) if results['volume_ratios'] else 0.0,
        'n_switch_pairs': results['switch_total'],
        'n_samples': len(results['dices']),
    }
    return metrics


def dice_score(pred, gt):
    """Compute Dice score between binary tensors."""
    inter = (pred * gt).sum().item()
    union = pred.sum().item() + gt.sum().item()
    if union == 0:
        return 0.0
    return 2 * inter / union
