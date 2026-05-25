"""Experiment 1: Detailed epoch 1 checkpoint analysis.

Check if epoch 1 PFA advantage is real or blob collapse:
- Dice distribution (per-sample)
- Volume ratio distribution
- Mask size statistics
- Switch accuracy with Dice threshold (only count if Dice > 0.2)
"""
import sys, torch, json
import numpy as np
sys.path.insert(0, 'src')

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point
from eval_faithfulness_v2 import dice_score

device = 'cuda'
MAX_BATCHES = 120


def detailed_eval(ckpt_path, label, val_loader):
    model = load_model('/root/autodl-tmp/SAM-Med3D',
                       '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth',
                       'vit_b_ori', device)
    apply_lora_to_sam_med3d(model)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()

    dices = []
    vol_ratios = []
    mask_sizes = []
    gt_sizes = []
    switch_results = []
    switch_with_dice_filter = []  # only count if both preds have Dice > 0.2

    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= MAX_BATCHES:
                break
            if batch['primary_organ_id'] == -1:
                continue

            image = batch['image'].to(device)
            gt_primary = batch['gt_primary'].to(device)
            pos_pt = batch['clean_point'].to(device)

            logits, _ = infer_with_point(model, image, pos_pt, device)
            pred = (torch.sigmoid(logits) > 0.5).float()

            d = dice_score(pred, gt_primary)
            dices.append(d)

            pred_vol = pred.sum().item()
            gt_vol = gt_primary.sum().item()
            mask_sizes.append(pred_vol)
            gt_sizes.append(gt_vol)
            if gt_vol > 0:
                vol_ratios.append(pred_vol / gt_vol)

            # Switch test
            if batch['has_cross']:
                gt_cross = batch['gt_cross'].to(device)
                cross_pt = batch['cross_point'].to(device)

                logits_B, _ = infer_with_point(model, image, cross_pt, device)
                pred_B = (torch.sigmoid(logits_B) > 0.5).float()

                dice_A_primary = dice_score(pred, gt_primary)
                dice_A_cross = dice_score(pred, gt_cross)
                dice_B_cross = dice_score(pred_B, gt_cross)
                dice_B_primary = dice_score(pred_B, gt_primary)

                A_correct = dice_A_primary > dice_A_cross
                B_correct = dice_B_cross > dice_B_primary
                switch_results.append(int(A_correct and B_correct))

                # Filtered: only count if predictions are non-trivial
                if dice_A_primary > 0.2 and dice_B_cross > 0.2:
                    switch_with_dice_filter.append(int(A_correct and B_correct))

    dices = np.array(dices)
    vol_ratios = np.array(vol_ratios)
    mask_sizes = np.array(mask_sizes)
    gt_sizes = np.array(gt_sizes)

    print(f'\n=== {label} ===')
    print(f'  Dice: mean={dices.mean():.4f}, std={dices.std():.4f}, '
          f'median={np.median(dices):.4f}, min={dices.min():.4f}, max={dices.max():.4f}')
    print(f'  Dice > 0.5: {(dices > 0.5).sum()}/{len(dices)} ({(dices > 0.5).mean()*100:.1f}%)')
    print(f'  Dice > 0.3: {(dices > 0.3).sum()}/{len(dices)} ({(dices > 0.3).mean()*100:.1f}%)')
    print(f'  Dice < 0.1: {(dices < 0.1).sum()}/{len(dices)} ({(dices < 0.1).mean()*100:.1f}%)')
    print(f'  Vol ratio: mean={vol_ratios.mean():.3f}, std={vol_ratios.std():.3f}, '
          f'median={np.median(vol_ratios):.3f}')
    print(f'  Mask size: mean={mask_sizes.mean():.0f}, gt_size mean={gt_sizes.mean():.0f}')
    print(f'  Tiny masks (<100 voxels): {(mask_sizes < 100).sum()}/{len(mask_sizes)}')
    print(f'  Empty masks (0 voxels): {(mask_sizes == 0).sum()}/{len(mask_sizes)}')
    print(f'  Switch Accuracy (all): {np.mean(switch_results):.4f} ({sum(switch_results)}/{len(switch_results)})')
    if switch_with_dice_filter:
        print(f'  Switch Accuracy (Dice>0.2 filter): {np.mean(switch_with_dice_filter):.4f} '
              f'({sum(switch_with_dice_filter)}/{len(switch_with_dice_filter)})')
    else:
        print(f'  Switch Accuracy (Dice>0.2 filter): N/A (no qualifying pairs)')

    return {
        'dice_mean': float(dices.mean()),
        'dice_std': float(dices.std()),
        'dice_median': float(np.median(dices)),
        'vol_ratio_mean': float(vol_ratios.mean()),
        'vol_ratio_std': float(vol_ratios.std()),
        'mask_size_mean': float(mask_sizes.mean()),
        'gt_size_mean': float(gt_sizes.mean()),
        'tiny_masks': int((mask_sizes < 100).sum()),
        'empty_masks': int((mask_sizes == 0).sum()),
        'switch_acc_all': float(np.mean(switch_results)) if switch_results else 0,
        'switch_acc_filtered': float(np.mean(switch_with_dice_filter)) if switch_with_dice_filter else 0,
        'n_switch_pairs': len(switch_results),
        'n_filtered_pairs': len(switch_with_dice_filter),
    }


# Load data once
val_ds = PEADataset('/root/autodl-tmp/data/amos22_cached', split='Va',
                    samples_per_volume=2, seed=43)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

print('Experiment 1: Epoch 1 vs Epoch 5 detailed analysis')
print('=' * 70)

results = {}
results['pfa_ep1'] = detailed_eval(
    'results/pilot_pfa_v2/pfa_full/best_pfa.pth', 'PFA_FULL epoch 1', val_loader)
results['pfa_ep5'] = detailed_eval(
    'results/pilot_pfa_v2/pfa_full/pfa_epoch005.pth', 'PFA_FULL epoch 5', val_loader)
results['seg_ep1'] = detailed_eval(
    'results/pilot_seg_v2/seg_only/best_pfa.pth', 'SEG_ONLY epoch 1', val_loader)
results['seg_ep5'] = detailed_eval(
    'results/pilot_seg_v2/seg_only/pfa_epoch005.pth', 'SEG_ONLY epoch 5', val_loader)

# Summary
print('\n' + '=' * 70)
print('SUMMARY: Is epoch 1 PFA advantage real or blob collapse?')
print('-' * 70)
print(f'{"Config":<18} {"Dice":>6} {"VolRatio":>9} {"Switch":>7} {"Switch(filt)":>12} {"Tiny%":>6}')
print('-' * 70)
for name in ['pfa_ep1', 'seg_ep1', 'pfa_ep5', 'seg_ep5']:
    r = results[name]
    tiny_pct = r['tiny_masks'] / MAX_BATCHES * 100
    print(f'{name:<18} {r["dice_mean"]:>6.3f} {r["vol_ratio_mean"]:>9.3f} '
          f'{r["switch_acc_all"]:>7.3f} {r["switch_acc_filtered"]:>12.3f} {tiny_pct:>6.1f}%')

with open('results/exp1_epoch_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to results/exp1_epoch_analysis.json')
