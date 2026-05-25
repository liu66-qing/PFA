"""Run corrected faithfulness eval on saved checkpoints."""
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

# Load model
model = load_model('/root/autodl-tmp/SAM-Med3D', 
                   '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth',
                   'vit_b_ori', device)
lora_info = apply_lora_to_sam_med3d(model)
print(f'LoRA injected: {lora_info["trainable_params"]:,} params')

# Validation data
val_ds = PEADataset('/root/autodl-tmp/data/amos22_cached', split='Va', 
                    samples_per_volume=2, seed=43)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
print(f'Val dataset: {len(val_ds)} items')

# 1. Evaluate BASELINE (no checkpoint loaded = random LoRA init)
print('\n=== BASELINE (LoRA random init, no training) ===')
metrics_base = evaluate_faithfulness_v2(model, val_loader, device, infer_with_point, max_batches=30)
for k, v in metrics_base.items():
    print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

# 2. Load seg_only checkpoint
ckpt_path = 'results/pilot_seg_only/seg_only/best_pfa.pth'
try:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f'\n=== SEG_ONLY (loaded {ckpt_path}) ===')
    metrics_seg = evaluate_faithfulness_v2(model, val_loader, device, infer_with_point, max_batches=30)
    for k, v in metrics_seg.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
except Exception as e:
    print(f'Could not load seg_only checkpoint: {e}')

# 3. Load pfa_full checkpoint if available
ckpt_path2 = 'results/pilot_pfa_full/pfa_full/best_pfa.pth'
try:
    # Reload fresh model
    model2 = load_model('/root/autodl-tmp/SAM-Med3D',
                        '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth',
                        'vit_b_ori', device)
    apply_lora_to_sam_med3d(model2)
    state2 = torch.load(ckpt_path2, map_location=device)
    model2.load_state_dict(state2, strict=False)
    print(f'\n=== PFA_FULL (loaded {ckpt_path2}) ===')
    metrics_pfa = evaluate_faithfulness_v2(model2, val_loader, device, infer_with_point, max_batches=30)
    for k, v in metrics_pfa.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
except Exception as e:
    print(f'Could not load pfa_full checkpoint: {e}')
