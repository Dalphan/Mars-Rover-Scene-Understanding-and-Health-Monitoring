from __future__ import annotations

import logging
import math

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.transforms import prepare_batch
from src.metrics.segmentation_metrics import (
    compute_segmentation_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.train.run_context import RunContext
from src.utils.memory import log_memory_usage

def build_scheduler(cfg: DictConfig, optimizer, steps_per_epoch: int):
    total_steps = max(int(cfg.epochs) * max(int(steps_per_epoch), 1), 1)
    warmup_steps = max(
        int(cfg.scheduler.warmup_epochs) * max(int(steps_per_epoch), 1), 0
    )

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epoch: int,
    cfg: DictConfig,
    context: RunContext,
    logger: logging.Logger,
) -> dict:
    model.train()
    total_loss = 0.0
    num_batches = 0
    confmat = create_confusion_matrix(context.num_classes, device=device)
    log_memory_usage(
        f"train epoch {epoch} start", cfg.memory.log_usage, logger
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train epoch {epoch}")):
        if cfg.limits.train_batches is not None and batch_idx >= int(
            cfg.limits.train_batches
        ):
            break
        images, masks = prepare_batch(
            batch, device, list(cfg.image_mean), list(cfg.image_std)
        )
        del batch
        logits = model(images)
        loss = criterion(logits, masks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=1)
            update_confusion_matrix(
                confmat,
                preds,
                masks,
                context.num_classes,
                int(cfg.ignore_index),
            )
        del preds, logits, loss, images, masks
        interval = int(cfg.memory.log_interval_batches or 0)
        if interval and (batch_idx + 1) % interval == 0:
            log_memory_usage(
                f"train epoch {epoch} batch {batch_idx + 1}",
                cfg.memory.log_usage,
                logger,
            )

    metrics = compute_segmentation_metrics(confmat, int(cfg.ignore_index))
    metrics["loss"] = total_loss / max(num_batches, 1)
    log_memory_usage(f"train epoch {epoch} end", cfg.memory.log_usage, logger)
    return {
        "train_loss": metrics["loss"],
        "train_pixel_accuracy": metrics["pixel_accuracy"],
        "train_miou": metrics["miou"],
        "train_per_class_iou": metrics["per_class_iou"],
    }
