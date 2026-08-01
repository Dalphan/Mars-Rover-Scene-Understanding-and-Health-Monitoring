from __future__ import annotations

import torch


def create_confusion_matrix(num_classes: int, device=None):
    """
    Create a confusion matrix for semantic segmentation.

    Shape:
        confmat: [num_classes, num_classes]

    Rows:
        ground-truth classes

    Columns:
        predicted classes
    """
    return torch.zeros(
        (num_classes, num_classes),
        dtype=torch.float64,
        device=device,
    )


@torch.no_grad()
def update_confusion_matrix(
    confmat: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
):
    """
    Update confusion matrix.

    preds:
        [B, H, W], predicted class IDs, values 0..8

    targets:
        [B, H, W], ground-truth class IDs or ignore_index.

    Pixels whose target equals ignore_index do not contribute to the confusion
    matrix. Predictions of any class on the remaining pixels are kept and
    counted normally.
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    valid = targets != ignore_index

    preds = preds[valid]
    targets = targets[valid]

    valid_range = (targets >= 0) & (targets < num_classes)
    preds = preds[valid_range]
    targets = targets[valid_range]

    indices = targets * num_classes + preds

    batch_confmat = torch.bincount(
        indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    confmat += batch_confmat.to(confmat.dtype)

    return confmat


def compute_segmentation_metrics(
    confmat: torch.Tensor,
    ignore_index: int = 0,
):
    """
    Compute pixel accuracy, per-class IoU, and mIoU.

    mIoU is computed over classes that occur in the ground truth. If
    ignore_index is a valid class ID, that class is excluded as well. Sentinel
    values outside the class range (for example -100) do not exclude a class.
    """
    tp = torch.diag(confmat)

    support = confmat.sum(dim=1)
    predicted = confmat.sum(dim=0)

    union = support + predicted - tp

    iou = tp / torch.clamp(union, min=1.0)

    valid_classes = support > 0
    if 0 <= ignore_index < iou.numel():
        valid_classes[ignore_index] = False

    miou = iou[valid_classes].mean() if valid_classes.any() else confmat.new_tensor(0.0)

    total_correct = tp.sum()
    total_labeled = confmat.sum()

    pixel_accuracy = total_correct / torch.clamp(total_labeled, min=1.0)

    return {
        "pixel_accuracy": pixel_accuracy.item(),
        "miou": miou.item(),
        "per_class_iou": iou.detach().cpu().tolist(),
    }
