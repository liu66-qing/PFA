"""PEA Loss Module: Prompt-Equivariant Adaptation losses.

Implements the four-term loss:
  L_total = L_seg + λ_inv * L_same_target + λ_sens * L_cross_target + λ_multi * L_multi_prompt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    """Combined Dice + BCE loss."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        return self.dice_weight * dice + self.bce_weight * bce


class SameTargetConsistencyLoss(nn.Module):
    """L_same_target: predictions from same-target prompts should agree.

    Uses symmetric KL divergence between sigmoid outputs of clean and noisy prompts.
    """

    def forward(
        self,
        logits_clean: torch.Tensor,
        logits_noisy: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits_clean).clamp(1e-6, 1 - 1e-6)
        q = torch.sigmoid(logits_noisy).clamp(1e-6, 1 - 1e-6)
        kl_pq = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
        kl_qp = q * (q.log() - p.log()) + (1 - q) * ((1 - q).log() - (1 - p).log())
        return 0.5 * (kl_pq.mean() + kl_qp.mean())


class CrossTargetDissimilarityLoss(nn.Module):
    """L_cross_target: predictions for different targets should NOT overlap.

    Two components:
    1. Dissimilarity: penalize IoU between pred_A and pred_B exceeding threshold τ
    2. Correctness: pred_B should match GT_B (standard seg loss)
    """

    def __init__(self, tau: float = 0.1, correctness_weight: float = 1.0):
        super().__init__()
        self.tau = tau
        self.correctness_weight = correctness_weight
        self.seg_loss = SegmentationLoss()

    def forward(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        gt_b: torch.Tensor,
    ) -> torch.Tensor:
        # Dissimilarity: soft IoU between pred_A and pred_B should be low
        prob_a = torch.sigmoid(logits_a)
        prob_b = torch.sigmoid(logits_b)
        intersection = (prob_a * prob_b).sum(dim=(1, 2, 3, 4))
        union = (prob_a + prob_b - prob_a * prob_b).sum(dim=(1, 2, 3, 4))
        iou = intersection / (union + 1e-6)
        dissimilarity = F.relu(iou - self.tau).mean()

        # Correctness: pred_B should segment GT_B
        correctness = self.seg_loss(logits_b, gt_b)

        return dissimilarity + self.correctness_weight * correctness


class MultiPromptConsistencyLoss(nn.Module):
    """L_multi_prompt: point and box prompts for same target should agree."""

    def forward(
        self,
        logits_point: torch.Tensor,
        logits_box: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits_point).clamp(1e-6, 1 - 1e-6)
        q = torch.sigmoid(logits_box).clamp(1e-6, 1 - 1e-6)
        kl_pq = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
        return kl_pq.mean()


class PEALoss(nn.Module):
    """Full PEA loss combining all four terms."""

    def __init__(
        self,
        lambda_inv: float = 1.0,
        lambda_sens: float = 0.5,
        lambda_multi: float = 0.3,
        cross_target_tau: float = 0.1,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_sens = lambda_sens
        self.lambda_multi = lambda_multi

        self.seg_loss = SegmentationLoss()
        self.same_target_loss = SameTargetConsistencyLoss()
        self.cross_target_loss = CrossTargetDissimilarityLoss(tau=cross_target_tau)
        self.multi_prompt_loss = MultiPromptConsistencyLoss()

    def forward(
        self,
        logits_clean: torch.Tensor,
        gt_clean: torch.Tensor,
        logits_noisy: torch.Tensor | None = None,
        logits_cross: torch.Tensor | None = None,
        gt_cross: torch.Tensor | None = None,
        logits_point: torch.Tensor | None = None,
        logits_box: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all loss terms.

        Returns dict with individual losses and total.
        """
        losses = {}

        # L_seg: standard segmentation loss on clean prompt
        losses["seg"] = self.seg_loss(logits_clean, gt_clean)

        # L_same_target: consistency between clean and noisy predictions
        if logits_noisy is not None:
            losses["same_target"] = self.same_target_loss(logits_clean, logits_noisy)
        else:
            losses["same_target"] = torch.tensor(0.0, device=logits_clean.device)

        # L_cross_target: dissimilarity + correctness for cross-target
        if logits_cross is not None and gt_cross is not None:
            losses["cross_target"] = self.cross_target_loss(
                logits_clean, logits_cross, gt_cross)
        else:
            losses["cross_target"] = torch.tensor(0.0, device=logits_clean.device)

        # L_multi_prompt: point-box agreement
        if logits_point is not None and logits_box is not None:
            losses["multi_prompt"] = self.multi_prompt_loss(logits_point, logits_box)
        else:
            losses["multi_prompt"] = torch.tensor(0.0, device=logits_clean.device)

        # Total
        losses["total"] = (
            losses["seg"]
            + self.lambda_inv * losses["same_target"]
            + self.lambda_sens * losses["cross_target"]
            + self.lambda_multi * losses["multi_prompt"]
        )

        return losses
