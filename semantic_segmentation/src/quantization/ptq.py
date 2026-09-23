from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.quantization.ptq_pipeline import run_ptq


@hydra.main(
    config_path="../../configs",
    config_name="quantization/ptq",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the notebook-synchronized PTQ pipeline."""

    run_ptq(cfg)


if __name__ == "__main__":
    main()
