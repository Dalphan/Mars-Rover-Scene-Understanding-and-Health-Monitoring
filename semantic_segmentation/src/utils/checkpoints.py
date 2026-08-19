from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    output_dir,
    epoch,
    model,
    optimizer,
    scheduler,
    best_miou,
    cfg,
    is_best: bool,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_miou": best_miou,
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    last_path = output_dir / "last.ckpt"
    torch.save(checkpoint, last_path)

    if is_best:
        best_path = output_dir / "best.ckpt"
        torch.save(checkpoint, best_path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device)

    unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
