from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.models import build_model
from src.anomaly_detection.training import build_optimizer, build_scheduler


@hydra.main(
    config_path="../../configs/anomaly_detection",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(
        f"Model: {cfg.model.name} | backbone={cfg.model.backbone} | "
        f"parameters={total_parameters:,} | trainable={trainable_parameters:,} | "
        f"optimizer={cfg.optimizer.name} | scheduler={cfg.scheduler.name}"
    )


if __name__ == "__main__":
    main()
