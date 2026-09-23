from __future__ import annotations

import logging
import math

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.quantization.core import save_json
from src.quantization.qat_config import QATPaths
from src.quantization.qat_model import (
    distillation_loss,
    evaluate_torch_model,
    freeze_batch_norm,
)
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


