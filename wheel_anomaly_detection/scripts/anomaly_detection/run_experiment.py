from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.training import run_anomaly_experiment


@hydra.main(
    config_path="../../configs/anomaly_detection",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_anomaly_experiment(cfg)


if __name__ == "__main__":
    main()
