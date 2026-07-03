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

    This implementation follows the S5Mars/Mars-Bench convention:
        - targets are class IDs 0..8
        - class 0 is ignore_index
        - Dice is computed only on valid target pixels
        - Dice is averaged over semantic classes 1..8

    Important:
        Do not remove predictions equal to 0.
        If the model predicts class 0 on a valid target pixel, this should still
        reduce the Dice score for the true class.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 0,
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
        self.ignore_index = ignore_index
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
        if logits.ndim != 4:
            raise ValueError(f"Expected logits [B, C, H, W], got {logits.shape}")

        if targets.ndim != 3:
            raise ValueError(f"Expected targets [B, H, W], got {targets.shape}")

        batch_size, channels, height, width = logits.shape

        if channels != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} channels, got {channels}.")

        if targets.shape != (batch_size, height, width):
            raise ValueError(
                f"Target shape {targets.shape} does not match logits spatial shape "
                f"{(batch_size, height, width)}."
            )

        # probs: [B, C, H, W]
        probs = torch.softmax(logits, dim=1)

        # valid_mask: [B, H, W]
        # Pixels with target == ignore_index are excluded from Dice computation.
        valid_mask = targets != self.ignore_index

        if not valid_mask.any():
            return logits.sum() * 0.0

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        # target_one_hot: [B, H, W, C] -> [B, C, H, W]
        target_one_hot = F.one_hot(
            safe_targets.long(),
            num_classes=self.num_classes,
        ).permute(0, 3, 1, 2).float()

        # valid_mask_bc: [B, 1, H, W]
        valid_mask_bc = valid_mask.unsqueeze(1).float()

        probs = probs * valid_mask_bc
        target_one_hot = target_one_hot * valid_mask_bc

        class_ids = [
            class_id
            for class_id in range(self.num_classes)
            if class_id != self.ignore_index
        ]

        probs = probs[:, class_ids, :, :]
        target_one_hot = target_one_hot[:, class_ids, :, :]

        probs = probs.permute(1, 0, 2, 3).reshape(len(class_ids), -1)
        target_one_hot = target_one_hot.permute(1, 0, 2, 3).reshape(len(class_ids), -1)

        class_volume = target_one_hot.sum(dim=1)

        if self.weight_type == "uniform":
            weights = torch.ones_like(class_volume)
        elif self.weight_type == "simple":
            weights = 1.0 / torch.clamp(class_volume, min=self.smooth)
        elif self.weight_type == "square":
            weights = 1.0 / torch.clamp(class_volume**2, min=self.smooth)
        else:
            raise RuntimeError("Invalid weight_type should have been caught in __init__.")

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
        ignore_index = 0
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
            ignore_index=ignore_index,
            weight_type=weight_type,
            smooth=smooth,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(logits, targets)
        gd = self.generalized_dice(logits, targets)

        return self.alpha * ce + (1.0 - self.alpha) * gd
