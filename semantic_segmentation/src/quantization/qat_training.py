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

import torch
import torch.nn as nn
from tqdm import tqdm


def segformer_linear_group(name: str) -> str | None:
    if name.endswith((".mlp.dense1", ".mlp.dense2", ".mlp.fc1", ".mlp.fc2")):
        return "encoder_mlp"
    decoder = (
        ".decode_head." in name
        and name.endswith(".proj")
        and (".linear_c." in name or ".linear_projections." in name)
    )
    return "decoder_projection" if decoder else None


def prepare_selective_qat(student, calibration_loader, cfg, device, restore_path):
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.nn import TensorQuantizer
    except ImportError as exc:
        raise RuntimeError("NVIDIA ModelOpt 0.46.1 is required for QAT") from exc

    source_linear = {
        name: module
        for name, module in student.named_modules()
        if isinstance(module, nn.Linear)
    }
    selected_groups = [str(value) for value in cfg.model.int8_linear_groups]
    discovered = {
        group: sorted(
            name for name in source_linear if segformer_linear_group(name) == group
        )
        for group in selected_groups
    }
    expected_counts = {"encoder_mlp": 16, "decoder_projection": 4}
    for group in selected_groups:
        if group not in expected_counts or len(discovered[group]) != expected_counts[group]:
            raise RuntimeError(
                f"Unexpected SegFormer Linear group {group}: {len(discovered.get(group, []))}"
            )
    fallback_prefixes = [str(value) for value in cfg.model.fp16_fallback_module_prefixes]
    target_linear_names = sorted(
        name
        for names in discovered.values()
        for name in names
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in fallback_prefixes)
    )
    conv_parent = mtq.QuantModuleRegistry.get_key(nn.Conv2d)
    linear_parent = mtq.QuantModuleRegistry.get_key(nn.Linear)
    if not conv_parent or not linear_parent:
        raise RuntimeError("ModelOpt has no Conv2d/Linear quantization registry entries")
    linear_cfg = []
    for name in target_linear_names:
        linear_cfg.extend(
            [
                {
                    "quantizer_name": f"{name}.weight_quantizer",
                    "parent_class": linear_parent,
                    "enable": True,
                    "cfg": {"num_bits": 8, "axis": 0, "trt_high_precision_dtype": "Half"},
                },
                {
                    "quantizer_name": f"{name}.input_quantizer",
                    "parent_class": linear_parent,
                    "enable": True,
                    "cfg": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Half"},
                },
            ]
        )
    selective_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "parent_class": conv_parent,
                "enable": True,
                "cfg": {"num_bits": 8, "axis": 0, "trt_high_precision_dtype": "Half"},
            },
            {
                "quantizer_name": "*input_quantizer",
                "parent_class": conv_parent,
                "enable": True,
                "cfg": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Half"},
            },
        ]
        + linear_cfg
        + [
            {"quantizer_name": f"*{prefix}.*", "enable": False}
            for prefix in fallback_prefixes
        ],
        "algorithm": "max",
    }

    @torch.inference_mode()
    def calibration_loop(model):
        model.eval()
        for batch in tqdm(calibration_loader, desc="ModelOpt calibration"):
            model(batch["image"].to(device, non_blocking=True))

    if cfg.steps.restore_saved_qat:
        student = mto.restore(student, str(restore_path), map_location=device).to(device)
    else:
        student = mtq.quantize(student, selective_cfg, forward_loop=calibration_loop)
    freeze_batch_norm(student)

    modules = dict(student.named_modules())
    conv_modules = {
        name: module for name, module in modules.items() if isinstance(module, nn.Conv2d)
    }
    linear_modules = {
        name: module for name, module in modules.items() if isinstance(module, nn.Linear)
    }
    layer_norms = {
        name: module for name, module in modules.items() if isinstance(module, nn.LayerNorm)
    }
    expected_fp16_conv = sorted(
        name
        for name in conv_modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in fallback_prefixes)
    )
    expected_int8_conv = sorted(set(conv_modules) - set(expected_fp16_conv))

    def quantized_names(modules_by_name):
        int8_names, fp16_names, partial = [], [], []
        for name, module in modules_by_name.items():
            input_q = getattr(module, "input_quantizer", None)
            weight_q = getattr(module, "weight_quantizer", None)
            input_enabled = input_q is not None and input_q.is_enabled
            weight_enabled = weight_q is not None and weight_q.is_enabled
            if input_enabled and weight_enabled:
                int8_names.append(name)
            elif not input_enabled and not weight_enabled:
                fp16_names.append(name)
            else:
                partial.append(name)
        return sorted(int8_names), sorted(fp16_names), sorted(partial)

    int8_conv, fp16_conv, partial_conv = quantized_names(conv_modules)
    int8_linear, fp16_linear, partial_linear = quantized_names(linear_modules)
    enabled = sorted(
        name
        for name, module in modules.items()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    )
    expected_enabled = sorted(
        f"{name}.{quantizer}"
        for name in expected_int8_conv + target_linear_names
        for quantizer in ("input_quantizer", "weight_quantizer")
    )
    if (
        partial_conv
        or partial_linear
        or int8_conv != expected_int8_conv
        or fp16_conv != expected_fp16_conv
        or int8_linear != target_linear_names
        or set(enabled) != set(expected_enabled)
    ):
        raise RuntimeError(
            {
                "partial_conv": partial_conv,
                "partial_linear": partial_linear,
                "expected_int8_conv": expected_int8_conv,
                "actual_int8_conv": int8_conv,
                "expected_int8_linear": target_linear_names,
                "actual_int8_linear": int8_linear,
                "missing_quantizers": sorted(set(expected_enabled) - set(enabled)),
                "unexpected_quantizers": sorted(set(enabled) - set(expected_enabled)),
            }
        )
    audit = {
        "conv2d_total": len(conv_modules),
        "conv2d_int8": len(int8_conv),
        "conv2d_fp16_fallback": len(fp16_conv),
        "fp16_fallback_conv_names": fp16_conv,
        "linear_total": len(linear_modules),
        "linear_int8": len(int8_linear),
        "linear_int8_names": int8_linear,
        "linear_fp16": len(fp16_linear),
        "linear_fp16_names": fp16_linear,
        "linear_groups_int8": discovered,
        "layer_norm_total_fp16": len(layer_norms),
        "enabled_quantizer_names": enabled,
    }
    return student, audit, TensorQuantizer, mto

import logging
import math

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.quantization.core import save_json
from src.quantization.qat_config import QATPaths
from src.quantization.qat_storage import upload_file

LOGGER = logging.getLogger("s5mars_modelopt_qat")

def train_qat(
    teacher,
    student,
    train_loader,
    val_loader,
    criterion,
    mto,
    paths: QATPaths,
    cfg: DictConfig,
    device,
):
    if cfg.steps.restore_saved_qat:
        restored = evaluate_torch_model(
            student, val_loader, device, cfg, cfg.dataset.val_split
        )
        best_weights = {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()
        }
        return student, [], restored["miou"], best_weights

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("QAT student has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    accumulation = int(cfg.training.gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = max(int(cfg.training.epochs) * steps_per_epoch, 1)
    warmup_steps = int(cfg.training.warmup_epochs) * steps_per_epoch

    def lr_multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    amp_enabled = bool(cfg.training.use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_miou = -float("inf")
    best_weights = None
    epochs_without_improvement = 0
    temperature = float(cfg.training.distillation_temperature)

    for epoch in range(int(cfg.training.epochs)):
        student.train()
        freeze_batch_norm(student)
        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "supervised": 0.0, "distillation": 0.0}
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"QAT {epoch + 1}/{cfg.training.epochs}")
        ):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                student_logits = student(images)
            supervised = criterion(student_logits.float(), masks)
            distillation = distillation_loss(
                student_logits.float(), teacher_logits.float(), temperature
            )
            loss = (
                float(cfg.training.supervised_weight) * supervised
                + float(cfg.training.distillation_weight) * distillation
            )
            scaler.scale(loss / accumulation).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running["loss"] += loss.item()
            running["supervised"] += supervised.item()
            running["distillation"] += distillation.item()

        metrics = evaluate_torch_model(
            student, val_loader, device, cfg, cfg.dataset.val_split
        )
        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": running["loss"] / max(len(train_loader), 1),
            "train_supervised_loss": running["supervised"] / max(len(train_loader), 1),
            "train_distillation_loss": running["distillation"] / max(len(train_loader), 1),
            "val_pixel_accuracy": metrics["pixel_accuracy"],
            "val_miou": metrics["miou"],
        }
        history.append(row)
        mto.save(student, str(paths.last_qat))
        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            epochs_without_improvement = 0
            best_weights = {
                name: value.detach().cpu().clone()
                for name, value in student.state_dict().items()
            }
            mto.save(student, str(paths.best_qat))
            if cfg.steps.upload_best_each_epoch:
                try:
                    upload_file(paths.best_qat, cfg)
                except Exception as error:
                    LOGGER.warning("Best-checkpoint upload deferred: %s", error)
        else:
            epochs_without_improvement += 1
        save_json(history, paths.history)
        LOGGER.info("QAT epoch metrics: %s", row)
        if epochs_without_improvement >= int(cfg.training.early_stopping_patience):
            LOGGER.info("QAT early stopping")
            break

    if best_weights is None:
        raise RuntimeError("QAT produced no best checkpoint")
    student.load_state_dict(best_weights, strict=True)
    student.eval()
    freeze_batch_norm(student)
    return student, history, best_miou, best_weights
