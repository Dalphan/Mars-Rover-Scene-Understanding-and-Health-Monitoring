from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.utils.checkpoints import (
    checkpoint_config,
    download_checkpoint_from_drive_folder,
)

LOGGER = logging.getLogger(__name__)

def build_inference_model(model_cfg, dataset_cfg):
    name = str(model_cfg.name)
    if name == "segformer_b0":
        from src.models.segformer_b0 import SegFormerB0ForMars

        return SegFormerB0ForMars(
            pretrained_name=str(model_cfg.pretrained_name),
            num_classes=int(dataset_cfg.num_classes),
            ignore_index=int(dataset_cfg.ignore_index),
            log_shapes=False,
        )
    if name == "smp":
        from src.models.smp_model import SMPModelForMars

        return SMPModelForMars(
            architecture=str(model_cfg.smp_architecture),
            encoder_name=str(model_cfg.smp_encoder_name),
            encoder_weights=model_cfg.smp_encoder_weights,
            in_channels=3,
            num_classes=int(dataset_cfg.num_classes),
            log_shapes=False,
        )
    raise ValueError("Quantization supports model.name=segformer_b0 or smp")


def validate_model_checkpoint(checkpoint, model_cfg) -> None:
    config = checkpoint_config(checkpoint)
    if not config:
        LOGGER.warning("Checkpoint has no config; architecture cannot be verified")
        return
    expected = {"model_name": str(model_cfg.name)}
    if str(model_cfg.name) == "smp":
        expected.update(
            smp_architecture=str(model_cfg.smp_architecture),
            smp_encoder_name=str(model_cfg.smp_encoder_name),
        )
    mismatches = {
        key: {"expected": value, "checkpoint": config.get(key)}
        for key, value in expected.items()
        if config.get(key) not in (None, value)
    }
    if mismatches:
        raise ValueError(f"Checkpoint incompatible with config: {mismatches}")


def load_trained_model(checkpoint_path, model_cfg, dataset_cfg, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = build_inference_model(model_cfg, dataset_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        validate_model_checkpoint(checkpoint, model_cfg)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def ensure_checkpoint(checkpoint_cfg, experiment: str | None = None) -> Path:
    filename = str(checkpoint_cfg.filename)
    path = Path(str(checkpoint_cfg.local_dir)) / filename
    folder_id = getattr(checkpoint_cfg, "folder_id", None)
    if experiment is not None:
        folder_id = checkpoint_cfg.folder_id_override or checkpoint_cfg.drive_folders.get(
            experiment
        )
    if checkpoint_cfg.get("force_download", False) or not path.is_file():
        download_checkpoint_from_drive_folder(folder_id, filename, path)
    return path
