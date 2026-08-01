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


def test_miou_excludes_classes_without_ground_truth_support():
    confmat = create_confusion_matrix(num_classes=9)
    for class_id in range(1, 9):
        confmat[class_id, class_id] = 1

    metrics = compute_segmentation_metrics(
        confmat=confmat,
        ignore_index=-100,
    )

    assert metrics["per_class_iou"][0] == 0.0
    assert metrics["miou"] == 1.0


def test_negative_ignore_index_does_not_exclude_a_valid_class():
    confmat = create_confusion_matrix(num_classes=2)
    confmat[0, 0] = 1
    confmat[1, 1] = 1

    metrics = compute_segmentation_metrics(
        confmat=confmat,
        ignore_index=-100,
    )

    assert metrics["per_class_iou"] == [1.0, 1.0]
    assert metrics["miou"] == 1.0
