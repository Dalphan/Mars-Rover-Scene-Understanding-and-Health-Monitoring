from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .curiosity_wheel_dataset import CuriosityWheelDataset
from .preprocessing import PreprocessingConfig, WheelPreprocessor


def _optional_tuple(value):
    """Convert an optional OmegaConf list to an immutable tuple."""
    return None if value is None else tuple(value)


def build_preprocessing(config: DictConfig) -> WheelPreprocessor:
    """Build preprocessing from one Hydra train or evaluation section."""
    return WheelPreprocessor(
        PreprocessingConfig(
            resize=_optional_tuple(config.resize),
            normalize_mean=_optional_tuple(config.normalize_mean),
            normalize_std=_optional_tuple(config.normalize_std),
            augmentations_enabled=bool(config.augmentations_enabled),
            brightness=config.brightness,
            contrast=config.contrast,
            gamma=config.gamma,
            saturation=config.saturation,
            sensor_noise=config.sensor_noise,
            gaussian_noise=config.gaussian_noise,
            gaussian_blur=_optional_tuple(config.gaussian_blur),
        )
    )


def build_dataloader(
    root: str | Path,
    split: str,
    *,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    preprocessing: WheelPreprocessor | None = None,
) -> DataLoader:
    """Build one loader; only the training split is shuffled."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = CuriosityWheelDataset(root, split, preprocessing=preprocessing)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def build_dataloaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test loaders from the Hydra config."""
    train_preprocessing = build_preprocessing(cfg.preprocessing.train)
    evaluation_preprocessing = build_preprocessing(cfg.preprocessing.evaluation)

    def make_loader(split: str, *, is_train: bool = False) -> DataLoader:
        return build_dataloader(
            cfg.dataset.root,
            split,
            batch_size=int(cfg.dataloader.batch_size),
            num_workers=int(cfg.dataloader.num_workers),
            pin_memory=bool(cfg.dataloader.pin_memory),
            seed=int(cfg.seed),
            preprocessing=train_preprocessing if is_train else evaluation_preprocessing,
        )

    return (
        make_loader(str(cfg.dataset.splits.train), is_train=True),
        make_loader(str(cfg.dataset.splits.validation)),
        make_loader(str(cfg.dataset.splits.test)),
    )
