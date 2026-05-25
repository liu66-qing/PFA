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

def get_per_pair_results(ckpt_path, val_loader, max_batches=120):
    model = load_model('/root/autodl-tmp/SAM-Med3D',
                       '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth',
                       'vit_b_ori', device)
    apply_lora_to_sam_med3d(model)
    if ckpt_path:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    
    pair_results = []
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
            
            dice_A_primary = dice_score(pred_A, gt_primary)
            dice_A_cross = dice_score(pred_A, gt_cross)
            dice_B_cross = dice_score(pred_B, gt_cross)
            dice_B_primary = dice_score(pred_B, gt_primary)
            
            A_correct = dice_A_primary > dice_A_cross
            B_correct = dice_B_cross > dice_B_primary
            switch_correct = A_correct and B_correct
            pair_results.append(int(switch_correct))
    
    return pair_results

val_ds = PEADataset('/root/autodl-tmp/data/amos22_cached', split='Va',
                    samples_per_volume=2, seed=43)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

print('Computing per-pair switch results...')
seg_pairs = get_per_pair_results('results/pilot_seg_only/seg_only/best_pfa.pth', val_loader)
pfa_pairs = get_per_pair_results('results/pilot_pfa_full/pfa_full/best_pfa.pth', val_loader)

n = min(len(seg_pairs), len(pfa_pairs))
seg_pairs = seg_pairs[:n]
pfa_pairs = pfa_pairs[:n]

print(f'Paired samples: {n}')
print(f'seg_only correct: {sum(seg_pairs)}/{n} = {sum(seg_pairs)/n:.4f}')
print(f'pfa_full correct: {sum(pfa_pairs)}/{n} = {sum(pfa_pairs)/n:.4f}')

# McNemar 2x2 table
both_correct = sum(s == 1 and p == 1 for s, p in zip(seg_pairs, pfa_pairs))
pfa_only = sum(s == 0 and p == 1 for s, p in zip(seg_pairs, pfa_pairs))
seg_only_val = sum(s == 1 and p == 0 for s, p in zip(seg_pairs, pfa_pairs))
both_wrong = sum(s == 0 and p == 0 for s, p in zip(seg_pairs, pfa_pairs))

print(f'\nMcNemar 2x2 table:')
print(f'  Both correct: {both_correct}')
print(f'  PFA only correct: {pfa_only}')
print(f'  seg_only only correct: {seg_only_val}')
print(f'  Both wrong: {both_wrong}')

# McNemar test (with continuity correction)
discordant = pfa_only + seg_only_val
if discordant > 0:
    chi2 = (abs(pfa_only - seg_only_val) - 1)**2 / discordant
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    print(f'\nMcNemar chi2 = {chi2:.4f}, p = {p_value:.4f}')
    if p_value < 0.05:
        print('SIGNIFICANT at p < 0.05')
    else:
        print('NOT significant at p < 0.05')
else:
    print('No discordant pairs - cannot compute McNemar')
