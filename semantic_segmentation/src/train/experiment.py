from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.dataloaders import build_loader
from src.losses.factory import build_criterion
from src.models.factory import build_model
from src.utils.model_stats import analyze_model, count_trainable_parameters


@dataclass
class TrainingExperiment:
    """Models and loaders shared by the training pipeline stages."""

    context: Any
    logger: Any
    device: Any
    model: Any
    criterion: Any
    train_dataset: Any
    train_loader: Any
    val_dataset: Any
    val_loader: Any
    test_dataset: Any
    test_loader: Any


def _build_loaders(cfg, logger):
    if cfg.execution.skip_train:
        test_dataset, test_loader = build_loader(
            cfg, str(cfg.splits.test), shuffle=False, logger=logger
        )
        logger.info("Training skipped; only source test data were loaded")
        return None, None, None, None, test_dataset, test_loader

    train_dataset, train_loader = build_loader(
        cfg, str(cfg.splits.train), shuffle=True, logger=logger
    )
    val_dataset, val_loader = build_loader(
        cfg, str(cfg.splits.val), shuffle=False, logger=logger
    )
    test_dataset, test_loader = build_loader(
        cfg, str(cfg.splits.test), shuffle=False, logger=logger
    )
    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )


def _build_model(cfg, context, device, logger):
    model = build_model(
        cfg,
        num_classes=context.num_classes,
        ignore_index=int(cfg.ignore_index),
    ).to(device)
    model.apply_freeze(str(cfg.freeze))
    stats = count_trainable_parameters(model)
    logger.info(
        "Model=%s run=%s freeze=%s total=%d trainable=%d frozen=%d",
        cfg.model.name,
        context.model_run_name,
        cfg.freeze,
        stats["total"],
        stats["trainable"],
        stats["frozen"],
    )
    if cfg.model_analysis.enabled:
        analyze_model(
            model,
            input_shape=(1, 3, int(cfg.image_size[0]), int(cfg.image_size[1])),
            warmup_iterations=int(cfg.model_analysis.warmup_iterations),
            measurement_iterations=int(cfg.model_analysis.measurement_iterations),
            logger=logger,
        )
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_ids = list(range(min(int(cfg.runtime.data_parallel_max_gpus), gpu_count)))
    if len(device_ids) == 2:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info("Using DataParallel on GPUs %s", device_ids)
    return model


def build_training_experiment(cfg, context, device, logger):
    loaders = _build_loaders(cfg, logger)
    model = _build_model(cfg, context, device, logger)
    criterion = build_criterion(
        name=str(cfg.criterion.name),
        num_classes=context.num_classes,
        ignore_index=int(cfg.ignore_index),
        alpha=float(cfg.criterion.alpha),
        weight_type=str(cfg.criterion.weight_type),
        smooth=float(cfg.criterion.smooth),
    )
    return TrainingExperiment(
        context=context,
        logger=logger,
        device=device,
        model=model,
        criterion=criterion,
        train_dataset=loaders[0],
        train_loader=loaders[1],
        val_dataset=loaders[2],
        val_loader=loaders[3],
        test_dataset=loaders[4],
        test_loader=loaders[5],
    )
