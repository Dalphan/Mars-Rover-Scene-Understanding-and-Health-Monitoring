from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.train.pipeline import run_training


@hydra.main(
    config_path="../../configs",
    config_name="train/segmentation",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the notebook-synchronized training pipeline."""

    run_training(cfg)


if __name__ == "__main__":
    main()
