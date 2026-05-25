"""Final corrected evaluation on 5-epoch checkpoints with larger sample size."""
import sys, torch, json
import numpy as np
sys.path.insert(0, 'src')

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point
from eval_faithfulness_v2 import evaluate_faithfulness_v2

device = 'cuda'
MAX_BATCHES = 120  # Use more samples for reliability


def eval_config(ckpt_path, label):
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


print('Evaluating with max_batches =', MAX_BATCHES)
results = {}
results['baseline'] = eval_config(None, 'BASELINE (LoRA random init)')
results['seg_only'] = eval_config('results/pilot_seg_only/seg_only/best_pfa.pth', 'SEG_ONLY (5 epochs)')
results['pfa_full'] = eval_config('results/pilot_pfa_full/pfa_full/best_pfa.pth', 'PFA_FULL (5 epochs)')

# Summary comparison
print('\n' + '=' * 70)
header = f"{'Metric':<20} {'Baseline':>10} {'seg_only':>10} {'pfa_full':>10} {'pfa-seg':>10}"
print(header)
print('-' * 70)
for k in ['switch_accuracy', 'mean_dice', 'target_margin', 'mean_volume_ratio', 'grounding_rate']:
    b = results['baseline'].get(k, 0)
    s = results['seg_only'].get(k, 0)
    p = results['pfa_full'].get(k, 0)
    d = p - s
    row = f'{k:<20} {b:>10.4f} {s:>10.4f} {p:>10.4f} {d:>+10.4f}'
    print(row)

pairs_row = f"{'n_switch_pairs':<20} {results['baseline']['n_switch_pairs']:>10} {results['seg_only']['n_switch_pairs']:>10} {results['pfa_full']['n_switch_pairs']:>10}"
print(pairs_row)

# Save results
with open('results/pilot_eval_v2.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved to results/pilot_eval_v2.json')
