import torch

from src.losses.factory import build_criterion
from src.losses.segmentation_losses import (
    CombinedSegmentationLoss,
    CrossEntropySegmentationLoss,
    GeneralizedDiceLoss,
)


def test_build_cross_entropy_loss():
    criterion = build_criterion(
        name="cross_entropy",
        num_classes=9,
        ignore_index=0,
    )

    assert isinstance(criterion, CrossEntropySegmentationLoss)


def test_build_generalized_dice_loss():
    criterion = build_criterion(
        name="generalized_dice",
        num_classes=9,
        ignore_index=0,
        weight_type="square",
        smooth=1e-5,
    )

    assert isinstance(criterion, GeneralizedDiceLoss)


def test_build_combined_loss():
    criterion = build_criterion(
        name="combined",
        num_classes=9,
        ignore_index=0,
        alpha=0.5,
        weight_type="square",
        smooth=1e-5,
    )

    assert isinstance(criterion, CombinedSegmentationLoss)


def test_losses_return_scalar():
    logits = torch.randn(2, 9, 512, 512, requires_grad=True)

    targets = torch.randint(
        low=0,
        high=9,
        size=(2, 512, 512),
        dtype=torch.long,
    )

    for name in ["cross_entropy", "generalized_dice", "combined"]:
        criterion = build_criterion(
            name=name,
            num_classes=9,
            ignore_index=0,
            alpha=0.5,
            weight_type="square",
            smooth=1e-5,
        )

        loss = criterion(logits, targets)

        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_generalized_dice_includes_class_zero_and_ignores_absent_classes():
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    logits = torch.full((1, 9, 4, 4), -10.0)
    logits[:, 0, :, :] = 10.0
    logits.requires_grad_()

    for weight_type in ["uniform", "simple", "square"]:
        criterion = GeneralizedDiceLoss(
            num_classes=9,
            ignore_index=0,
            weight_type=weight_type,
            smooth=1e-5,
        )

        loss = criterion(logits, targets)

        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() < 1e-4
