from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection.data import build_dataloaders


@hydra.main(
    config_path="../../configs/anomaly_detection",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Resolve relative dataset paths before Hydra can affect the working directory.
    cfg.dataset.root = str(Path(to_absolute_path(str(cfg.dataset.root))).resolve())
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    print(f"[1/3] Loading samples.csv from {cfg.dataset.root}")
    train_loader, validation_loader, test_loader = build_dataloaders(cfg)
    loaders = (
        ("train", train_loader),
        ("validation", validation_loader),
        ("test", test_loader),
    )

    print("[2/3] DataLoaders created")
    for split, loader in loaders:
        print(f"  {split}: {len(loader.dataset)} samples, {len(loader)} batches")

    print("[3/3] Reading one batch per split")
    for split, loader in loaders:
        batch = next(iter(loader))
        if batch["image"].dtype != torch.float32 or not torch.isfinite(batch["image"]).all():
            raise TypeError(f"{split} images must be finite float32 tensors")
        if batch["target_mask"].dtype != torch.uint8 or batch["anomaly_mask"].dtype != torch.uint8:
            raise TypeError(f"{split} masks must remain uint8 tensors")
        print(
            f"  {split}: image={tuple(batch['image'].shape)} {batch['image'].dtype}, "
            f"range=[{batch['image'].min().item():.3f}, {batch['image'].max().item():.3f}], "
            f"target_mask={tuple(batch['target_mask'].shape)}, "
            f"anomaly_mask={tuple(batch['anomaly_mask'].shape)}, "
            f"labels={batch['label'].tolist()}"
        )


if __name__ == "__main__":
    main()
