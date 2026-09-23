from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from src.metrics.segmentation_metrics import (
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.quantization.core import quantization_metrics
from src.utils.checkpoints import checkpoint_config

class MarsSegmentationModel(nn.Module):
    """SegFormer config-only wrapper used by the authoritative QAT notebook."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError("transformers is required for QAT") from exc
        class_names = {
            int(key): str(value) for key, value in cfg.dataset.class_names.items()
        }
        config = SegformerConfig.from_pretrained(
            str(cfg.model.pretrained_name),
            num_labels=int(cfg.dataset.num_classes),
            semantic_loss_ignore_index=int(cfg.dataset.ignore_index),
            id2label=class_names,
            label2id={label: index for index, label in class_names.items()},
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images):
        logits = self.model(pixel_values=images, return_dict=False)[0]
        return F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )


def load_fp32_model(path: Path, cfg: DictConfig):
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint_config(checkpoint) if isinstance(checkpoint, dict) else {}
    expected = {
        "model_name": str(cfg.model.name),
        "pretrained_name": str(cfg.model.pretrained_name),
    }
    mismatches = {
        key: {"expected": value, "checkpoint": config.get(key)}
        for key, value in expected.items()
        if config.get(key) not in (None, value)
    }
    checkpoint_image_size = config.get("image_size")
    if checkpoint_image_size is not None and tuple(checkpoint_image_size) != tuple(
        cfg.dataset.image_size
    ):
        mismatches["image_size"] = {
            "expected": list(cfg.dataset.image_size),
            "checkpoint": checkpoint_image_size,
        }
    if mismatches:
        raise ValueError(f"Checkpoint incompatible with QAT config: {mismatches}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    classifier_key = "model.decode_head.classifier.weight"
    found = state_dict.get(classifier_key)
    if found is None or found.shape[0] != int(cfg.dataset.num_classes):
        raise ValueError("Checkpoint classifier has an incompatible class count")
    model = MarsSegmentationModel(cfg)
    model.load_state_dict(state_dict, strict=True)
    return model


class CombinedLoss(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.num_classes = int(cfg.dataset.num_classes)
        self.ignore_index = int(cfg.dataset.ignore_index)
        self.alpha = float(cfg.training.combined_loss_alpha)
        self.smooth = float(cfg.training.dice_smooth)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        probabilities = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
        one_hot = F.one_hot(safe_targets, self.num_classes).permute(0, 3, 1, 2).float()
        one_hot = one_hot * valid.unsqueeze(1)
        volumes = one_hot.sum((0, 2, 3))
        weights = torch.where(volumes > 0, volumes.clamp_min(1).pow(-2), 0.0)
        intersection = (probabilities * one_hot).sum((0, 2, 3))
        denominator = (probabilities + one_hot).sum((0, 2, 3))
        dice = 1.0 - (2.0 * (weights * intersection).sum() + self.smooth) / (
            (weights * denominator).sum() + self.smooth
        )
        return self.alpha * self.cross_entropy(logits, targets) + (1 - self.alpha) * dice


def freeze_batch_norm(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            module.requires_grad_(False)


@torch.inference_mode()
def evaluate_torch_model(model, loader, device, cfg, split):
    model.eval()
    confmat = create_confusion_matrix(int(cfg.dataset.num_classes))
    for batch in tqdm(loader, desc=f"evaluate {split}"):
        predictions = model(batch["image"].to(device, non_blocking=True)).argmax(1)
        update_confusion_matrix(
            confmat,
            predictions.cpu(),
            batch["mask"],
            int(cfg.dataset.num_classes),
            int(cfg.dataset.ignore_index),
        )
    metrics = quantization_metrics(
        confmat,
        cfg.dataset.num_classes,
        cfg.dataset.ignore_index,
        cfg.dataset.class_names,
    )
    metrics.update(split=str(split), samples=len(loader.dataset))
    return metrics


def distillation_loss(student_logits, teacher_logits, temperature: float):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return (
        F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(1).mean()
        * temperature**2
    )


