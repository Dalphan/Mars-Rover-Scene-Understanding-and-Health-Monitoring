from __future__ import annotations

import torch
from tqdm import tqdm

from src.data.transforms import prepare_batch
from src.metrics.segmentation_metrics import (
    compute_segmentation_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.utils.memory import log_memory_usage


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
    ignore_index: int,
    image_mean,
    image_std,
    desc: str = "Validation",
    limit_batches: int | None = None,
    log_memory: bool = False,
    logger=None,
):
    model.eval()

    total_loss = 0.0
    num_batches = 0
    confmat = create_confusion_matrix(num_classes=num_classes, device=device)
    log_memory_usage(f"{desc} start", enabled=log_memory, logger=logger)

    for batch_idx, batch in enumerate(tqdm(loader, desc=desc)):
        if limit_batches is not None and batch_idx >= int(limit_batches):
            break
        images, masks = prepare_batch(batch, device, image_mean, image_std)
        del batch

        logits = model(images)
        # logits: [B, 9, 512, 512]

        loss = criterion(logits, masks)
        total_loss += loss.item()
        num_batches += 1

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
        del preds, logits, loss, images, masks

    metrics = compute_segmentation_metrics(
        confmat=confmat,
        ignore_index=ignore_index,
    )

    metrics["loss"] = total_loss / max(num_batches, 1)
    log_memory_usage(f"{desc} end", enabled=log_memory, logger=logger)

    return metrics
