"""Final evaluation on epoch 5 checkpoints with corrected metrics."""
import sys, torch, json
import numpy as np
sys.path.insert(0, 'src')

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point
from eval_faithfulness_v2 import evaluate_faithfulness_v2, dice_score
from scipy.stats import chi2 as chi2_dist

device = 'cuda'
MAX_BATCHES = 120


def load_and_eval(ckpt_path, label):
    model = load_model('/root/autodl-tmp/SAM-Med3D',
                       '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth',
                       'vit_b_ori', device)
    apply_lora_to_sam_med3d(model)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)

    val_ds = PEADataset('/root/autodl-tmp/data/amos22_cached', split='Va',
                        samples_per_volume=2, seed=43)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    metrics = evaluate_faithfulness_v2(model, val_loader, device, infer_with_point,
                                       max_batches=MAX_BATCHES)
    print(f'\n=== {label} ===')
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    return metrics


def get_per_pair_switches(ckpt_path, val_loader, max_batches=120):
    model = load_model('/root/autodl-tmp/SAM-Med3D',
                       '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth',
                       'vit_b_ori', device)
    apply_lora_to_sam_med3d(model)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()

    results = []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            if batch['primary_organ_id'] == -1:
                continue
            if not batch['has_cross']:
                continue

            image = batch['image'].to(device)
            gt_primary = batch['gt_primary'].to(device)
            gt_cross = batch['gt_cross'].to(device)
            pos_pt = batch['clean_point'].to(device)
            cross_pt = batch['cross_point'].to(device)

            logits_A, _ = infer_with_point(model, image, pos_pt, device)
            pred_A = (torch.sigmoid(logits_A) > 0.5).float()
            logits_B, _ = infer_with_point(model, image, cross_pt, device)
            pred_B = (torch.sigmoid(logits_B) > 0.5).float()

            A_correct = dice_score(pred_A, gt_primary) > dice_score(pred_A, gt_cross)
            B_correct = dice_score(pred_B, gt_cross) > dice_score(pred_B, gt_primary)
            results.append(int(A_correct and B_correct))
    return results


print(f'Evaluating with max_batches = {MAX_BATCHES}')
print('=' * 70)

# 1. Aggregate metrics
results = {}
results['baseline'] = load_and_eval(None, 'BASELINE (random LoRA init)')
results['seg_only_ep5'] = load_and_eval(
    'results/pilot_seg_v2/seg_only/pfa_epoch005.pth', 'SEG_ONLY epoch 5')
results['pfa_full_ep5'] = load_and_eval(
    'results/pilot_pfa_v2/pfa_full/pfa_epoch005.pth', 'PFA_FULL epoch 5')

# 2. Summary table
print('\n' + '=' * 70)
header = f"{'Metric':<20} {'Baseline':>10} {'seg_only':>10} {'pfa_full':>10} {'pfa-seg':>10}"
print(header)
print('-' * 70)
for k in ['switch_accuracy', 'mean_dice', 'target_margin', 'mean_volume_ratio', 'grounding_rate']:
    b = results['baseline'].get(k, 0)
    s = results['seg_only_ep5'].get(k, 0)
    p = results['pfa_full_ep5'].get(k, 0)
    d = p - s
    print(f'{k:<20} {b:>10.4f} {s:>10.4f} {p:>10.4f} {d:>+10.4f}')
n_b = results['baseline']['n_switch_pairs']
n_s = results['seg_only_ep5']['n_switch_pairs']
n_p = results['pfa_full_ep5']['n_switch_pairs']
print(f"{'n_switch_pairs':<20} {n_b:>10} {n_s:>10} {n_p:>10}")

# 3. Paired McNemar test
print('\n' + '=' * 70)
print('PAIRED McNEMAR TEST (epoch 5 checkpoints)')
print('-' * 70)

val_ds = PEADataset('/root/autodl-tmp/data/amos22_cached', split='Va',
                    samples_per_volume=2, seed=43)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

seg_pairs = get_per_pair_switches(
    'results/pilot_seg_v2/seg_only/pfa_epoch005.pth', val_loader)
pfa_pairs = get_per_pair_switches(
    'results/pilot_pfa_v2/pfa_full/pfa_epoch005.pth', val_loader)

n = min(len(seg_pairs), len(pfa_pairs))
seg_pairs = seg_pairs[:n]
pfa_pairs = pfa_pairs[:n]

print(f'Paired samples: {n}')
print(f'seg_only correct: {sum(seg_pairs)}/{n} = {sum(seg_pairs)/n:.4f}')
print(f'pfa_full correct: {sum(pfa_pairs)}/{n} = {sum(pfa_pairs)/n:.4f}')

both_correct = sum(s == 1 and p == 1 for s, p in zip(seg_pairs, pfa_pairs))
pfa_only = sum(s == 0 and p == 1 for s, p in zip(seg_pairs, pfa_pairs))
seg_only_v = sum(s == 1 and p == 0 for s, p in zip(seg_pairs, pfa_pairs))
both_wrong = sum(s == 0 and p == 0 for s, p in zip(seg_pairs, pfa_pairs))

print(f'\n2x2 table:')
print(f'  Both correct:       {both_correct}')
print(f'  PFA only correct:   {pfa_only}')
print(f'  seg only correct:   {seg_only_v}')
print(f'  Both wrong:         {both_wrong}')

discordant = pfa_only + seg_only_v
if discordant > 0:
    chi2_val = (abs(pfa_only - seg_only_v) - 1)**2 / discordant
    p_value = 1 - chi2_dist.cdf(chi2_val, df=1)
    print(f'\nMcNemar chi2 = {chi2_val:.4f}, p = {p_value:.4f}')
    if p_value < 0.05:
        print('*** SIGNIFICANT at p < 0.05 ***')
    elif p_value < 0.10:
        print('* Marginally significant (p < 0.10)')
    else:
        print('NOT significant at p < 0.05')

# Save
with open('results/pilot_eval_v2_epoch5.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to results/pilot_eval_v2_epoch5.json')
