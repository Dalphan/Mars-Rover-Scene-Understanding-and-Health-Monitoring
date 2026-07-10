from __future__ import annotations

from src.losses.segmentation_losses import (
    CombinedSegmentationLoss,
    CrossEntropySegmentationLoss,
    GeneralizedDiceLoss,
)


def build_criterion(
    name: str,
    num_classes: int,
    ignore_index: int,
    alpha: float = 0.5,
    weight_type: str = "square",
    smooth: float = 1e-5,
):
    """
    Build segmentation loss.

    Supported losses:
        - cross_entropy
        - generalized_dice
        - combined

    Input shape convention:
        logits: [B, C, H, W] = [B, 9, 512, 512]
        targets: [B, H, W]  = [B, 512, 512]

    S5Mars:
        C = 9
        target values = 0..8
        cross_entropy ignores target value 0
        generalized_dice uses all classes, including class 0
    """
    name = name.lower()

    if name == "cross_entropy":
        return CrossEntropySegmentationLoss(ignore_index=ignore_index)

    if name == "generalized_dice":
        return GeneralizedDiceLoss(
            num_classes=num_classes,
            weight_type=weight_type,
            smooth=smooth,
        )

    if name == "combined":
        return CombinedSegmentationLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
            alpha=alpha,
            weight_type=weight_type,
            smooth=smooth,
        )

    raise ValueError(
        f"Unknown segmentation criterion '{name}'. "
        "Expected one of: cross_entropy, generalized_dice, combined."
    )
