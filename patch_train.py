"""Patch train_pfa.py to reuse image embedding for stability loss."""
import sys

filepath = sys.argv[1] if len(sys.argv) > 1 else 'src/train_pfa.py'
with open(filepath, 'r') as f:
    content = f.read()

# 1. Add infer_with_embedding function after infer_with_point
new_func = '''

def infer_with_embedding(model, image_embedding, point, device, crop_size=128):
    """Run SAM-Med3D decoder with pre-computed image embedding."""
    point_tensor = point.view(1, 1, 3).to(device)
    label_tensor = torch.ones((1, 1), dtype=torch.long, device=device)
    low_res_logits = torch.zeros(
        (1, 1, crop_size // 4, crop_size // 4, crop_size // 4),
        dtype=torch.float32, device=device)

    sparse, dense = model.prompt_encoder(
        points=[point_tensor, label_tensor], boxes=None, masks=low_res_logits)
    low_res_masks, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    masks = F.interpolate(low_res_masks, size=(crop_size,) * 3,
                          mode="trilinear", align_corners=False)
    return masks

'''

# Find end of infer_with_point
marker = "    return masks, image_embedding\n"
idx = content.find(marker)
if idx == -1:
    print("ERROR: Could not find marker")
    sys.exit(1)
insert_pos = idx + len(marker)
content = content[:insert_pos] + new_func + content[insert_pos:]

# 2. Replace stability block to reuse img_emb
old_block = '        # Second point in same organ for stability\n        logits_second = None\n        if config["use_stability"]:\n            noisy_pt = batch["noisy_point"].to(device)\n            logits_second, _ = infer_with_point(model, image, noisy_pt, device)'

new_block = '        # Second point in same organ for stability (reuse image embedding)\n        logits_second = None\n        if config["use_stability"]:\n            noisy_pt = batch["noisy_point"].to(device)\n            logits_second = infer_with_embedding(model, img_emb, noisy_pt, device, image.shape[-1])'

if old_block not in content:
    print("ERROR: Could not find stability block to replace")
    print("Looking for:", repr(old_block[:80]))
    sys.exit(1)

content = content.replace(old_block, new_block)

with open(filepath, 'w') as f:
    f.write(content)

print("OK: Patched train_pfa.py - reuse encoder embedding for stability loss")
