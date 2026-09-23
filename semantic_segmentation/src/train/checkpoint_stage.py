from __future__ import annotations

import torch

from src.train.reporting import checkpoint_expectations
from src.utils.checkpoints import (
    download_checkpoint_from_drive_folder,
    unwrap_model,
    validate_checkpoint_compatibility,
)


def prepare_evaluation_checkpoint(experiment, state, cfg):
    """Optionally download and load the checkpoint used for evaluation."""

    context = experiment.context
    if cfg.execution.skip_train and cfg.checkpoint.download_from_drive:
        if cfg.checkpoint.force_download or not context.checkpoint_path.exists():
            download_checkpoint_from_drive_folder(
                context.drive_folder_id,
                context.checkpoint_name,
                context.checkpoint_path,
            )
    elif cfg.checkpoint.download_from_drive:
        experiment.logger.info("Drive download skipped because training is enabled")

    if not cfg.execution.load_checkpoint:
        return None
    if not context.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {context.checkpoint_path}")

    raw_checkpoint = torch.load(
        context.checkpoint_path, map_location=experiment.device
    )
    expected = checkpoint_expectations(context)
    expected["criterion_weight_type"] = str(cfg.criterion.weight_type)
    validate_checkpoint_compatibility(raw_checkpoint, expected)
    unwrap_model(experiment.model).load_state_dict(raw_checkpoint["model_state_dict"])
    state.best_miou = raw_checkpoint.get("best_miou")
    experiment.logger.info(
        "Loaded checkpoint=%s epoch=%s best_miou=%s",
        context.checkpoint_path,
        raw_checkpoint.get("epoch"),
        state.best_miou,
    )
    return context.checkpoint_path
