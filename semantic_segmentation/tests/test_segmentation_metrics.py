import torch

from src.metrics.segmentation_metrics import (
    compute_segmentation_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)


def test_confusion_matrix_ignores_target_zero():
    num_classes = 9
    ignore_index = 0

    confmat = create_confusion_matrix(num_classes=num_classes)

    preds = torch.tensor(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
            ]
        ]
    )

    targets = torch.tensor(
        [
            [
                [0, 1, 2],
                [0, 4, 5],
            ]
        ]
    )

    confmat = update_confusion_matrix(
        confmat=confmat,
        preds=preds,
        targets=targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    # target == 0 pixels must not contribute to the confusion matrix.
    assert confmat.sum().item() == 4

    metrics = compute_segmentation_metrics(
        confmat=confmat,
        ignore_index=ignore_index,
    )

    assert "pixel_accuracy" in metrics
    assert "miou" in metrics
    assert "per_class_iou" in metrics


def test_out_of_range_ignore_index_keeps_background_in_miou():
    confmat = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    metrics = compute_segmentation_metrics(confmat, ignore_index=-100)

    # Both supported semantic classes, including Background (0), participate.
    assert metrics["miou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0
