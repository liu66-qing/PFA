import sys, time, torch
sys.path.insert(0, 'src')
from pea_dataset import PEADataset
from torch.utils.data import DataLoader
from sam_med3d_gate1_wrapper import load_model
from lora import apply_lora_to_sam_med3d
from train_pfa import infer_with_point

# Data
ds = PEADataset('/root/autodl-tmp/data/amos22_cached', split='Tr', samples_per_volume=4, seed=42)
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
print(f'Dataset: {len(ds)} items')

# Model
model = load_model('/root/autodl-tmp/SAM-Med3D', '/root/autodl-tmp/SAM-Med3D/ckpt/sam_med3d_turbo.pth', 'vit_b_ori', 'cuda')
lora_info = apply_lora_to_sam_med3d(model)
print(f'LoRA: {lora_info["trainable_params"]:,} params')

# Test 3 forward passes
for i, batch in enumerate(loader):
    if batch['primary_organ_id'].item() == -1:
        continue
    image = batch['image'].cuda()
    point = batch['clean_point'].cuda()
    t0 = time.time()
    logits, emb = infer_with_point(model, image, point, 'cuda')
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch['gt_primary'].cuda())
    loss.backward()
    elapsed = time.time() - t0
    print(f'Step {i}: fwd+bwd={elapsed:.2f}s, loss={loss.item():.4f}, logits={logits.shape}')
    if i >= 2:
        break

print(f'GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
print('Pipeline OK!')
