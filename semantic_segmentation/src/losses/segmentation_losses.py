from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropySegmentationLoss(nn.Module):
    """
    Standard semantic segmentation cross-entropy.

    Expected shapes:
        logits:  [B, C, H, W]
        targets: [B, H, W]

    For S5Mars:
        logits:  [B, 9, 512, 512]
        targets: [B, 512, 512], values 0..8
        ignore_index = 0

    Pixels where targets == 0 are ignored.
    """

    def __init__(self, ignore_index: int = 0) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class GeneralizedDiceLoss(nn.Module):
    """
    Generalized Dice Loss for semantic segmentation.

    All target labels are treated as valid semantic classes.
    Class 0 participates in the Dice computation exactly like every other class.
    Classes absent from the current target batch receive zero weight.
    """

    def __init__(
        self,
        num_classes: int,
        weight_type: str = "square",
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()

        if weight_type not in {"uniform", "simple", "square"}:
            raise ValueError(
                f"Invalid weight_type='{weight_type}'. "
                "Expected one of: uniform, simple, square."
            )

        self.num_classes = num_classes
        self.weight_type = weight_type
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:
                [B, C, H, W], raw model outputs.
                For S5Mars: [B, 9, 512, 512]

            targets:
                [B, H, W], integer class IDs.
                For S5Mars: [B, 512, 512], values 0..8

        Returns:
            Scalar generalized dice loss.
        """
        # probs: [B, C, H, W]
        probs = torch.softmax(logits, dim=1)

        # target_one_hot: [B, H, W, C] -> [B, C, H, W]
        target_one_hot = F.one_hot(
            targets.long(),
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).float()

        # Flatten batch and spatial dimensions:
        # probs:          [B, C, H, W] -> [C, B*H*W]
        # target_one_hot: [B, C, H, W] -> [C, B*H*W]
        probs = probs.permute(1, 0, 2, 3).reshape(self.num_classes, -1)
        target_one_hot = target_one_hot.permute(1, 0, 2, 3).reshape(self.num_classes, -1)

        class_volume = target_one_hot.sum(dim=1)
        present = class_volume > 0

        if self.weight_type == "uniform":
            weights = present.to(class_volume.dtype)
        elif self.weight_type == "simple":
            weights = torch.zeros_like(class_volume)
            weights[present] = 1.0 / class_volume[present]
        elif self.weight_type == "square":
            weights = torch.zeros_like(class_volume)
            weights[present] = 1.0 / class_volume[present].pow(2)
        else:
            raise RuntimeError("Invalid weight_type should have been caught in __init__.")

        if not present.any():
            return logits.sum() * 0.0

        intersection = (probs * target_one_hot).sum(dim=1)
        denominator = probs.sum(dim=1) + target_one_hot.sum(dim=1)

        numerator = 2.0 * (weights * intersection).sum() + self.smooth
        denominator = (weights * denominator).sum() + self.smooth

        dice_score = numerator / denominator
        return 1.0 - dice_score


class CombinedSegmentationLoss(nn.Module):
    """
    Combined segmentation loss:

        loss = alpha * cross_entropy + (1 - alpha) * generalized_dice

    For S5Mars:
        logits:  [B, 9, 512, 512]
        targets: [B, 512, 512], values 0..8

    Cross-entropy ignores target value 0. Generalized Dice uses all classes,
    including class 0.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 0,
        alpha: float = 0.5,
        weight_type: str = "square",
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha = alpha

        self.cross_entropy = CrossEntropySegmentationLoss(
            ignore_index=ignore_index,
        )

        self.generalized_dice = GeneralizedDiceLoss(
            num_classes=num_classes,
            weight_type=weight_type,
            smooth=smooth,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(logits, targets)
        gd = self.generalized_dice(logits, targets)

        return self.alpha * ce + (1.0 - self.alpha) * gd
