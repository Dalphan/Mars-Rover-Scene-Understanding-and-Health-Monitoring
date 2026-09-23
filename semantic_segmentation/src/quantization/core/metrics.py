from __future__ import annotations

import torch

from src.metrics.segmentation_metrics import (
    create_confusion_matrix,
    update_confusion_matrix,
)

def evaluate_torch(
    model,
    loader,
    device,
    num_classes: int,
    ignore_index: int,
    split_name: str,
    class_names=None,
):
    model.eval()
    confmat = create_confusion_matrix(num_classes)
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            update_confusion_matrix(
                confmat,
                predictions.cpu(),
                targets.cpu(),
                num_classes,
                ignore_index,
            )
    metrics = quantization_metrics(
        confmat, num_classes, ignore_index, class_names
    )
    metrics["split"] = split_name
    metrics["samples"] = len(loader.dataset)
    return metrics


def quantization_metrics(confmat, num_classes, ignore_index, class_names=None):
    """Notebook-format metrics, including named IoU and null absent classes."""

    confmat = confmat.double()
    true_positive = torch.diag(confmat)
    support = confmat.sum(dim=1)
    predicted = confmat.sum(dim=0)
    union = support + predicted - true_positive
    iou = torch.full((int(num_classes),), float("nan"), dtype=torch.float64)
    present = union > 0
    iou[present] = true_positive[present] / union[present]
    valid_classes = support > 0
    if 0 <= int(ignore_index) < int(num_classes):
        valid_classes[int(ignore_index)] = False
    names = (
        {int(key): str(value) for key, value in class_names.items()}
        if class_names is not None
        else {index: str(index) for index in range(int(num_classes))}
    )
    return {
        "pixel_accuracy": (
            true_positive.sum() / confmat.sum().clamp(min=1.0)
        ).item(),
        "miou": iou[valid_classes].mean().item(),
        "per_class_iou": {
            names[index]: (iou[index].item() if present[index] else None)
            for index in range(int(num_classes))
        },
    }
