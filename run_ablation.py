"""Quick ablation: test lower PFA lambda weights and warmup schedule.

Configs to test (3 epochs each for speed):
1. pfa_low: lambda_g=0.3, lambda_c=0.1, lambda_s=0.1 (reduced from 1.0/0.5/0.3)
2. pfa_warmup: seg_only for epoch 1, then add PFA losses for epochs 2-5
3. ground_only: just grounding loss (lambda_g=1.0, no contrast/stability)
"""
import sys, os, time, json, torch
import numpy as np
sys.path.insert(0, 'src')

from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point, train_one_epoch
from pfa_loss import PFALoss
from eval_faithfulness_v2 import evaluate_faithfulness_v2
from pathlib import Path

device = 'cuda'
DATA_DIR = '/root/autodl-tmp/data/amos22_cached'
SAM_ROOT = '/root/autodl-tmp/SAM-Med3D'
CKPT = '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth'


def run_experiment(config_name, lambdas, warmup_epochs=0, total_epochs=5):
    """Run a training experiment and evaluate every epoch."""
    print(f'\n{"="*60}')
    print(f'Config: {config_name}')
    print(f'Lambdas: ground={lambdas[0]}, contrast={lambdas[1]}, stable={lambdas[2]}')
    print(f'Warmup: {warmup_epochs} epochs seg_only first')
    print(f'{"="*60}')

    # Model
    model = load_model(SAM_ROOT, CKPT, 'vit_b_ori', device)
    apply_lora_to_sam_med3d(model)

    # Data
    train_ds = PEADataset(DATA_DIR, split='Tr', samples_per_volume=4, seed=42)
    val_ds = PEADataset(DATA_DIR, split='Va', samples_per_volume=2, seed=43)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-6)

    history = []
    for epoch in range(1, total_epochs + 1):
        # Determine current lambdas (warmup = seg_only first)
        if epoch <= warmup_epochs:
            cur_lambdas = (0.0, 0.0, 0.0)
            cur_config = {
                'use_grounding_neg': False, 'use_contrast': False,
                'use_stability': False, 'lambda_ground': 0.0,
                'lambda_contrast': 0.0, 'lambda_stable': 0.0,
            }
        else:
            cur_lambdas = lambdas
            cur_config = {
                'use_grounding_neg': lambdas[0] > 0,
                'use_contrast': lambdas[1] > 0,
                'use_stability': lambdas[2] > 0,
                'lambda_ground': lambdas[0],
                'lambda_contrast': lambdas[1],
                'lambda_stable': lambdas[2],
            }

        loss_fn = PFALoss(
            lambda_ground=cur_lambdas[0],
            lambda_contrast=cur_lambdas[1],
            lambda_stable=cur_lambdas[2],
        )

        t0 = time.time()
        train_losses = train_one_epoch(model, train_loader, optimizer, loss_fn, device, cur_config)
        scheduler.step()

        # Evaluate every epoch
        metrics = evaluate_faithfulness_v2(model, val_loader, device, infer_with_point, max_batches=60)
        elapsed = time.time() - t0

        record = {'epoch': epoch, 'time': elapsed, **train_losses, **metrics}
        history.append(record)

        print(f'  Ep {epoch}/{total_epochs} loss={train_losses["total"]:.4f} '
              f'| switch={metrics["switch_accuracy"]:.3f} dice={metrics["mean_dice"]:.3f} '
              f'margin={metrics["target_margin"]:.3f} vol_ratio={metrics["mean_volume_ratio"]:.2f} '
              f'| {elapsed:.0f}s')

    # Save
    out_dir = Path(f'results/ablation_{config_name}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save final checkpoint
    state = {k: v for k, v in model.state_dict().items()
             if any(t in k for t in ['lora', 'point_embed', 'not_a_point',
                                     'mask_downscaling', 'output_hyper',
                                     'iou_prediction', 'output_upscaling'])}
    torch.save(state, out_dir / 'final.pth')
    return history


# Run ablations
all_results = {}

# 1. Ground only (isolate grounding contribution)
all_results['ground_only'] = run_experiment(
    'ground_only', lambdas=(1.0, 0.0, 0.0), total_epochs=5)

# 2. PFA low lambdas
all_results['pfa_low'] = run_experiment(
    'pfa_low', lambdas=(0.3, 0.1, 0.1), total_epochs=5)

# 3. PFA with warmup (1 epoch seg_only, then PFA)
all_results['pfa_warmup'] = run_experiment(
    'pfa_warmup', lambdas=(1.0, 0.5, 0.3), warmup_epochs=1, total_epochs=5)

# Summary
print('\n' + '=' * 70)
print('ABLATION SUMMARY (epoch 5 metrics)')
print('-' * 70)
print(f'{"Config":<15} {"Switch":>8} {"Dice":>8} {"Margin":>8} {"VolRatio":>10}')
print('-' * 70)
for name, hist in all_results.items():
    ep5 = hist[-1]
    print(f'{name:<15} {ep5["switch_accuracy"]:>8.3f} {ep5["mean_dice"]:>8.3f} '
          f'{ep5["target_margin"]:>8.3f} {ep5["mean_volume_ratio"]:>10.3f}')

# Save all
with open('results/ablation_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print('\nSaved to results/ablation_summary.json')
