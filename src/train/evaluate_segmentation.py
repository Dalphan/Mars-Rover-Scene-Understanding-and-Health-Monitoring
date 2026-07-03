from __future__ import annotations

import torch
from tqdm import tqdm

from src.metrics.segmentation_metrics import (
    compute_segmentation_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    ignore_index: int,
):
    model.eval()

    total_loss = 0.0
    confmat = create_confusion_matrix(num_classes=num_classes, device=device)

    for batch in tqdm(loader, desc="Validation"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        # images: [B, 3, 512, 512]
        # masks:  [B, 512, 512], values 0..8

        logits = model(images)
        # logits: [B, 9, 512, 512]

        loss = criterion(logits, masks)
        total_loss += loss.item()

        # logits: [B, 9, 512, 512]
        # preds = argmax(logits, dim=1)
        # preds: [B, 512, 512], values 0..8
        preds = torch.argmax(logits, dim=1)

        confmat = update_confusion_matrix(
            confmat=confmat,
            preds=preds,
            targets=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    metrics = compute_segmentation_metrics(
        confmat=confmat,
        ignore_index=ignore_index,
    )

    metrics["loss"] = total_loss / max(len(loader), 1)

    return metrics
