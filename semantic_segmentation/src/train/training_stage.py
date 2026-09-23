from __future__ import annotations

from dataclasses import dataclass

import torch

from src.train.evaluate_segmentation import evaluate
from src.train.reporting import checkpoint_metadata, save_json
from src.train.training_loop import build_scheduler, train_one_epoch
from src.utils.checkpoints import save_checkpoint
from src.utils.memory import release_host_memory


@dataclass
class TrainingState:
    optimizer: object | None
    scheduler: object | None
    history: list
    best_miou: float | None


def run_training_stage(experiment, cfg) -> TrainingState:
    """Fit on the source train split and select checkpoints on validation mIoU."""

    if cfg.execution.skip_train:
        return TrainingState(None, None, [], None)

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in experiment.model.parameters()
            if parameter.requires_grad
        ),
        lr=float(cfg.optimizer.lr),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    scheduler = build_scheduler(cfg, optimizer, len(experiment.train_loader))
    metadata = checkpoint_metadata(cfg, experiment.context)
    history = []
    best_miou = -1.0

    for epoch in range(int(cfg.epochs)):
        train_stats = train_one_epoch(
            experiment.model,
            experiment.train_loader,
            experiment.criterion,
            optimizer,
            scheduler,
            experiment.device,
            epoch,
            cfg,
            experiment.context,
            experiment.logger,
        )
        release_host_memory()
        val_stats = evaluate(
            experiment.model,
            experiment.val_loader,
            experiment.criterion,
            experiment.device,
            experiment.context.num_classes,
            int(cfg.ignore_index),
            list(cfg.image_mean),
            list(cfg.image_std),
            desc="Validation",
            limit_batches=cfg.limits.val_batches,
            log_memory=bool(cfg.memory.log_usage),
            logger=experiment.logger,
        )
        release_host_memory()
        is_best = val_stats["miou"] > best_miou
        if is_best:
            best_miou = val_stats["miou"]
        checkpoint_path = save_checkpoint(
            experiment.context.output_dir,
            epoch,
            experiment.model,
            optimizer,
            scheduler,
            best_miou,
            cfg,
            bool(cfg.checkpoint.save_best and is_best),
            metadata=metadata,
        )
        row = {
            "epoch": epoch,
            **train_stats,
            "val_loss": val_stats["loss"],
            "val_pixel_accuracy": val_stats["pixel_accuracy"],
            "val_miou": val_stats["miou"],
            "val_per_class_iou": val_stats["per_class_iou"],
            "best_miou": best_miou,
        }
        history.append(row)
        experiment.logger.info(
            "epoch=%d metrics=%s checkpoint=%s", epoch, row, checkpoint_path
        )
    save_json(history, experiment.context.output_dir / "history.json")
    return TrainingState(optimizer, scheduler, history, best_miou)
