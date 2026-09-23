from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

def require_module(name: str, install_hint: str = "requirements-quantization.txt"):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional dependency {name!r} is missing; install {install_hint}"
        ) from exc


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def save_json(data, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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
