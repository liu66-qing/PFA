"""LoRA module for SAM-Med3D.

Injects low-rank adapters into the mask decoder's TwoWayTransformer3D attention
layers and optionally into the prompt encoder's mask downscaling path.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear."""

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features
        device = linear.weight.device

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_lora_into_attention(attn_module: nn.Module, rank: int = 8, alpha: float = 16.0):
    """Replace q/k/v/out projections in an Attention module with LoRA versions."""
    attn_module.q_proj = LoRALinear(attn_module.q_proj, rank, alpha)
    attn_module.k_proj = LoRALinear(attn_module.k_proj, rank, alpha)
    attn_module.v_proj = LoRALinear(attn_module.v_proj, rank, alpha)
    attn_module.out_proj = LoRALinear(attn_module.out_proj, rank, alpha)


def apply_lora_to_sam_med3d(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: str = "decoder+prompt",
) -> dict:
    """Apply LoRA to SAM-Med3D mask decoder and optionally prompt encoder.

    Args:
        model: SAM-Med3D model instance
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: "decoder", "prompt", or "decoder+prompt"

    Returns:
        dict with counts of injected LoRA modules
    """
    count = {"attention_layers": 0, "total_lora_params": 0}

    if "decoder" in target_modules:
        decoder = model.mask_decoder
        transformer = decoder.transformer

        # Inject into each TwoWayAttentionBlock3D
        for layer in transformer.layers:
            inject_lora_into_attention(layer.self_attn, rank, alpha)
            inject_lora_into_attention(layer.cross_attn_token_to_image, rank, alpha)
            inject_lora_into_attention(layer.cross_attn_image_to_token, rank, alpha)
            count["attention_layers"] += 3

        # Final attention layer
        inject_lora_into_attention(transformer.final_attn_token_to_image, rank, alpha)
        count["attention_layers"] += 1

    if "prompt" in target_modules:
        # LoRA on the mask downscaling conv layers (treat as linear on flattened)
        # For prompt encoder, we add trainable bias to point embeddings instead
        # since the main linear layers are in the PE random matrix (frozen buffer)
        pass  # Point embeddings are already small; we'll just unfreeze them

    # Freeze everything, then unfreeze LoRA params + specific prompt encoder params
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            count["total_lora_params"] += param.numel()

    # Unfreeze prompt encoder point embeddings and mask downscaling
    if "prompt" in target_modules:
        for name, param in model.prompt_encoder.named_parameters():
            if "point_embeddings" in name or "not_a_point_embed" in name:
                param.requires_grad = True
                count["total_lora_params"] += param.numel()
            if "mask_downscaling" in name:
                param.requires_grad = True
                count["total_lora_params"] += param.numel()

    # Unfreeze decoder output heads (small, needed for task adaptation)
    for name, param in model.mask_decoder.named_parameters():
        if "output_hypernetworks_mlps" in name:
            param.requires_grad = True
            count["total_lora_params"] += param.numel()
        if "iou_prediction_head" in name:
            param.requires_grad = True
            count["total_lora_params"] += param.numel()
        if "output_upscaling" in name:
            param.requires_grad = True
            count["total_lora_params"] += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    count["total_model_params"] = total_params
    count["trainable_params"] = trainable
    count["trainable_pct"] = 100.0 * trainable / total_params

    return count
