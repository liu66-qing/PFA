"""PFA Loss Module: Prompt Faithfulness Adaptation losses.

Revised from PEA after Gate 1B pivot. Core insight: the problem is not
prompt noise sensitivity, but prompt unfaithfulness (model ignores prompts).

Loss:
  L_total = L_seg + λ_g * L_grounding + λ_c * L_contrast + λ_s * L_stability
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
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="mean")
        return self.dice_weight * dice + self.bce_weight * bce


class PromptGroundingLoss(nn.Module):
    """L_grounding: prompt point region must be inside/outside predicted mask.

    Uses a 3x3x3 ball around the prompt point with distance weighting.
    Positive grounding: logits in ball around pos_point should be high.
    Negative grounding: logits in ball around neg_point should be low.
    """

    def __init__(self, ball_radius: int = 1):
        super().__init__()
        self.ball_radius = ball_radius
        # Pre-compute offsets for 3x3x3 ball
        r = ball_radius
        offsets = []
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    dist = (dz**2 + dy**2 + dx**2) ** 0.5
                    if dist <= r * 1.8:  # include all 27 for radius=1
                        weight = 1.0 / (1.0 + dist)  # distance weighting
                        offsets.append((dz, dy, dx, weight))
        self.offsets = offsets  # list of (dz, dy, dx, weight)

    def _ball_loss(self, logits: torch.Tensor, center: torch.Tensor,
                   positive: bool) -> torch.Tensor:
        """Compute grounding loss over ball around center point."""
        B = logits.shape[0]
        loss = torch.tensor(0.0, device=logits.device)
        total_weight = 0.0

        for i in range(B):
            cz, cy, cx = center[i].long()
            for dz, dy, dx, w in self.offsets:
                z = (cz + dz).clamp(0, logits.shape[2] - 1)
                y = (cy + dy).clamp(0, logits.shape[3] - 1)
                x = (cx + dx).clamp(0, logits.shape[4] - 1)
                logit_val = logits[i, 0, z, y, x]
                if positive:
                    loss = loss - w * F.logsigmoid(logit_val)
                else:
                    loss = loss - w * F.logsigmoid(-logit_val)
                total_weight += w

        return loss / max(total_weight, 1e-6)

    def forward(
        self,
        logits: torch.Tensor,
        pos_points: torch.Tensor,
        neg_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: B×1×D×H×W prediction logits
            pos_points: B×3 (z,y,x) positive prompt coordinates
            neg_points: B×3 (z,y,x) negative prompt coordinates (optional)
        """
        loss = self._ball_loss(logits, pos_points, positive=True)
        if neg_points is not None:
            loss = loss + self._ball_loss(logits, neg_points, positive=False)
            loss = loss / 2.0  # average pos and neg
        return loss


class SemanticContrastLoss(nn.Module):
    """L_contrast: prediction should match prompted organ MORE than any other.

    Margin-based: Dice(pred, GT_target) - max(Dice(pred, GT_other)) >= margin
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def _soft_dice(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice score (differentiable)."""
        probs_flat = probs.view(-1)
        target_flat = target.view(-1).float()
        intersection = (probs_flat * target_flat).sum()
        union = probs_flat.sum() + target_flat.sum()
        return (2.0 * intersection + 1.0) / (union + 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        gt_target: torch.Tensor,
        gt_others: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            logits: B×1×D×H×W prediction logits (from prompt targeting gt_target)
            gt_target: B×1×D×H×W ground truth of prompted organ
            gt_others: list of B×1×D×H×W ground truths of other organs in ROI
                       (up to 3 organs, prioritizing adjacent/confusable ones)
        """
        if not gt_others:
            return torch.tensor(0.0, device=logits.device)

        probs = torch.sigmoid(logits)
        B = logits.shape[0]
        loss = torch.tensor(0.0, device=logits.device)

        for i in range(B):
            dice_target = self._soft_dice(probs[i], gt_target[i])

            # Find max Dice to any other organ (hard negative mining)
            max_dice_other = torch.tensor(0.0, device=logits.device)
            for gt_o in gt_others:
                if gt_o[i].sum() < 10:  # skip empty masks in ROI
                    continue
                dice_other = self._soft_dice(probs[i], gt_o[i])
                max_dice_other = torch.max(max_dice_other, dice_other)

            # Margin loss: target should dominate by margin (max violation)
            violation = self.margin + max_dice_other - dice_target
            loss = loss + F.relu(violation)

        return loss / max(B, 1)


class IntraOrganStabilityLoss(nn.Module):
    """L_stability: different prompts within same organ → same prediction.

    Symmetric KL between predictions from two points inside the same organ.
    """

    def forward(
        self,
        logits_p1: torch.Tensor,
        logits_p2: torch.Tensor,
    ) -> torch.Tensor:
        p = torch.sigmoid(logits_p1).clamp(1e-6, 1 - 1e-6)
        q = torch.sigmoid(logits_p2).clamp(1e-6, 1 - 1e-6)
        kl_pq = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
        kl_qp = q * (q.log() - p.log()) + (1 - q) * ((1 - q).log() - (1 - p).log())
        return 0.5 * (kl_pq.mean() + kl_qp.mean())


class PFALoss(nn.Module):
    """Full PFA loss: Prompt Faithfulness Adaptation."""

    def __init__(
        self,
        lambda_ground: float = 1.0,
        lambda_contrast: float = 0.5,
        lambda_stable: float = 0.3,
        contrast_margin: float = 0.1,
    ):
        super().__init__()
        self.lambda_ground = lambda_ground
        self.lambda_contrast = lambda_contrast
        self.lambda_stable = lambda_stable

        self.seg_loss = SegmentationLoss()
        self.grounding_loss = PromptGroundingLoss()
        self.contrast_loss = SemanticContrastLoss(margin=contrast_margin)
        self.stability_loss = IntraOrganStabilityLoss()

    def forward(
        self,
        logits_primary: torch.Tensor,
        gt_primary: torch.Tensor,
        pos_point: torch.Tensor,
        neg_point: torch.Tensor | None = None,
        gt_others: list[torch.Tensor] | None = None,
        logits_second: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits_primary: prediction from primary prompt (B×1×D×H×W)
            gt_primary: GT mask of prompted organ (B×1×D×H×W)
            pos_point: positive prompt coordinates (B×3)
            neg_point: negative prompt in other organ (B×3), optional
            gt_others: list of GT masks for other organs in ROI, optional
            logits_second: prediction from second point in same organ, optional
        """
        losses = {}

        # L_seg
        losses["seg"] = self.seg_loss(logits_primary, gt_primary)

        # L_grounding
        losses["grounding"] = self.grounding_loss(logits_primary, pos_point, neg_point)

        # L_contrast
        if gt_others:
            losses["contrast"] = self.contrast_loss(
                logits_primary, gt_primary, gt_others)
        else:
            losses["contrast"] = torch.tensor(0.0, device=logits_primary.device)

        # L_stability
        if logits_second is not None:
            losses["stability"] = self.stability_loss(logits_primary, logits_second)
        else:
            losses["stability"] = torch.tensor(0.0, device=logits_primary.device)

        # Total
        losses["total"] = (
            losses["seg"]
            + self.lambda_ground * losses["grounding"]
            + self.lambda_contrast * losses["contrast"]
            + self.lambda_stable * losses["stability"]
        )

        return losses
