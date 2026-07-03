from __future__ import annotations

import logging
import math

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.dataloaders import build_dataloader, build_dataset
from src.losses.factory import build_criterion
from src.metrics.segmentation_metrics import (
    compute_segmentation_metrics,
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.models.factory import build_model
from src.train.evaluate_segmentation import evaluate
from src.utils.checkpoints import save_checkpoint
from src.utils.freeze import apply_freeze_mode, count_trainable_parameters
from src.utils.logging_utils import configure_logging
from src.utils.seed import set_seed
from src.utils.shape_debug import log_batch_shapes


def build_dataloaders(cfg: DictConfig, logger: logging.Logger | None = None):
    """
    Build train, validation, and test dataloaders using the existing S5Mars loader.
    """
    logger = logger or logging.getLogger(__name__)
    train_dataset = build_dataset(cfg, split=cfg.splits.train, logger=logger)
    val_dataset = build_dataset(cfg, split=cfg.splits.val, logger=logger)
    test_dataset = build_dataset(cfg, split=cfg.splits.test, logger=logger)

    train_loader = build_dataloader(train_dataset, cfg, split="train", logger=logger)
    val_loader = build_dataloader(val_dataset, cfg, split="val", logger=logger)
    test_loader = build_dataloader(test_dataset, cfg, split="test", logger=logger)

    logger.info(
        "Dataset ready: train=%d val=%d test=%d",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    logger.info(
        "Dataloaders ready: train_batches=%d val_batches=%d test_batches=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )
    return train_loader, val_loader, test_loader


def build_scheduler(cfg: DictConfig, optimizer, steps_per_epoch: int):
    """
    Build a per-step warmup + cosine scheduler.
    """
    if cfg.scheduler.name != "cosine":
        return None

    total_steps = max(int(cfg.epochs) * max(steps_per_epoch, 1), 1)
    warmup_steps = max(int(cfg.scheduler.warmup_epochs) * max(steps_per_epoch, 1), 0)

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
    epoch,
    cfg,
):
    model.train()

    total_loss = 0.0
    confmat = create_confusion_matrix(
        num_classes=cfg.model.num_classes,
        device=device,
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train epoch {epoch}")):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        # images: [B, 3, 512, 512]
        # masks:  [B, 512, 512], values 0..8, ignore_index=0
        if batch_idx == 0 and cfg.model.log_shapes:
            log_batch_shapes(batch, prefix="train/")
            print(f"train/images device shape: {tuple(images.shape)}")
            print(f"train/masks device shape: {tuple(masks.shape)}")

        # logits: [B, 9, 512, 512]
        logits = model(images)

        # criterion can be:
        #   cross_entropy:
        #       CE over valid pixels only, ignoring target == 0.
        #
        #   generalized_dice:
        #       Generalized Dice over valid pixels only, ignoring target == 0
        #       and computing Dice over classes 1..8.
        #
        #   combined:
        #       alpha * cross_entropy + (1 - alpha) * generalized_dice.
        # CrossEntropyLoss expects:
        #   logits: [B, C, H, W] = [B, 9, 512, 512]
        #   target: [B, H, W]    = [B, 512, 512]
        # Target pixels equal to 0 are ignored.
        loss = criterion(logits, masks)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=1)
            # preds: [B, 512, 512], values 0..8
            confmat = update_confusion_matrix(
                confmat=confmat,
                preds=preds,
                targets=masks,
                num_classes=cfg.model.num_classes,
                ignore_index=cfg.model.ignore_index,
            )

    metrics = compute_segmentation_metrics(
        confmat=confmat,
        ignore_index=cfg.model.ignore_index,
    )
    metrics["loss"] = total_loss / max(len(loader), 1)

    return {
        "train_loss": metrics["loss"],
        "train_pixel_accuracy": metrics["pixel_accuracy"],
        "train_miou": metrics["miou"],
        "train_per_class_iou": metrics["per_class_iou"],
    }


@hydra.main(config_path="../../configs", config_name="train/segformer_s5mars", version_base=None)
def main(cfg: DictConfig):
    """
    Train SegFormer-B0 on S5Mars using PyTorch only.
    """
    logger = configure_logging(cfg)
    logger.info("Training config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, logger=logger)
    _ = test_loader

    model = build_model(cfg).to(device)
    apply_freeze_mode(model, cfg.freeze)
    param_stats = count_trainable_parameters(model)
    print(f"Freeze mode: {cfg.freeze}")
    print(f"Total parameters: {param_stats['total']:,}")
    print(f"Trainable parameters: {param_stats['trainable']:,}")
    print(f"Frozen parameters: {param_stats['frozen']:,}")
    logger.info(
        "Freeze mode=%s total_parameters=%d trainable_parameters=%d frozen_parameters=%d",
        cfg.freeze,
        param_stats["total"],
        param_stats["trainable"],
        param_stats["frozen"],
    )

    criterion = build_criterion(
        name=cfg.criterion.name,
        num_classes=cfg.model.num_classes,
        ignore_index=cfg.model.ignore_index,
        alpha=cfg.criterion.alpha,
        weight_type=cfg.criterion.weight_type,
        smooth=cfg.criterion.smooth,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    scheduler = build_scheduler(cfg, optimizer, len(train_loader))

    best_miou = -1.0

    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            cfg=cfg,
        )

        val_stats = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=cfg.model.num_classes,
            ignore_index=cfg.model.ignore_index,
        )
        val_stats = {
            "val_loss": val_stats["loss"],
            "val_pixel_accuracy": val_stats["pixel_accuracy"],
            "val_miou": val_stats["miou"],
            "val_per_class_iou": val_stats["per_class_iou"],
        }

        current_miou = val_stats["val_miou"]

        is_best = current_miou > best_miou
        if is_best:
            best_miou = current_miou

        miou_gap = train_stats["train_miou"] - val_stats["val_miou"]
        loss_gap = val_stats["val_loss"] - train_stats["train_loss"]
        print(
            f"Epoch {epoch} | "
            f"train_loss={train_stats['train_loss']:.4f} | "
            f"train_miou={train_stats['train_miou']:.4f} | "
            f"val_loss={val_stats['val_loss']:.4f} | "
            f"val_miou={val_stats['val_miou']:.4f}"
        )
        print(
            f"Overfitting check | "
            f"miou_gap={miou_gap:.4f} | "
            f"loss_gap={loss_gap:.4f}"
        )
        logger.info(
            "epoch=%d train_loss=%.6f train_miou=%.6f train_pixel_accuracy=%.6f val_loss=%.6f val_miou=%.6f val_pixel_accuracy=%.6f miou_gap=%.6f loss_gap=%.6f best_miou=%.6f",
            epoch,
            train_stats["train_loss"],
            train_stats["train_miou"],
            train_stats["train_pixel_accuracy"],
            val_stats["val_loss"],
            val_stats["val_miou"],
            val_stats["val_pixel_accuracy"],
            miou_gap,
            loss_gap,
            best_miou,
        )

        save_checkpoint(
            output_dir=cfg.checkpoint.output_dir,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_miou=best_miou,
            cfg=cfg,
            is_best=bool(cfg.checkpoint.save_best and is_best),
        )


if __name__ == "__main__":
    main()
