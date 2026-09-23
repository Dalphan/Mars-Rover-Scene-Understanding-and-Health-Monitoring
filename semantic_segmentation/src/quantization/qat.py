from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.quantization.qat_pipeline import run_qat


@hydra.main(
    config_path="../../configs",
    config_name="quantization/qat",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the notebook-synchronized QAT pipeline."""

    run_qat(cfg)


if __name__ == "__main__":
    main()
