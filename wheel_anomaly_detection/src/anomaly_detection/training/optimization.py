from __future__ import annotations

from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def build_optimizer(model: nn.Module, cfg: DictConfig) -> Optimizer | None:
    """Return no optimizer for non-parametric PatchCore fitting."""
    if cfg.optimizer.name == "none":
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("optimizer=none requires a fully frozen model")
        return None
    raise ValueError(
        f"Unsupported optimizer {cfg.optimizer.name!r} for {cfg.model.name!r}; "
        "PatchCore requires optimizer=none."
    )


def build_scheduler(
    optimizer: Optimizer | None,
    cfg: DictConfig,
) -> LRScheduler | None:
    """Return no scheduler when the selected model has no optimizer."""
    if cfg.scheduler.name == "none":
        return None
    if optimizer is None:
        raise ValueError("A scheduler requires an optimizer")
    raise ValueError(
        f"Unsupported scheduler {cfg.scheduler.name!r} for {cfg.model.name!r}; "
        "PatchCore requires scheduler=none."
    )
