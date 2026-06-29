from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import DataLoader

from src.data.s5mars_dataset import S5MarsHFDataset
from src.data.transforms import SegmentationTransform


def build_dataset(cfg, split: str, logger: logging.Logger | None = None) -> S5MarsHFDataset:
    logger = logger or logging.getLogger(__name__)
    env_token = os.environ.get("HF_TOKEN")
    token = env_token or cfg.huggingface.token
    if token in ("", "HF_TOKEN_PLACEHOLDER", None):
        logger.warning("HF token is empty or placeholder; attempting dataset load without authentication")
        token = None
    else:
        logger.info("Using Hugging Face token from %s", "environment" if env_token else "config")

    transform = None
    if cfg.transforms.enabled:
        transform = SegmentationTransform(
            resize_enabled=cfg.transforms.resize.enabled,
            size=(cfg.transforms.resize.height, cfg.transforms.resize.width),
            normalize_enabled=cfg.transforms.normalize.enabled,
            mean=cfg.transforms.normalize.mean,
            std=cfg.transforms.normalize.std,
        )
        logger.info("Segmentation transform enabled")

    return S5MarsHFDataset(
        repo_id=cfg.huggingface.repo_id,
        split=split,
        token=token if cfg.huggingface.use_auth_token else None,
        cache_dir=cfg.huggingface.cache_dir,
        image_column=cfg.dataset.image_column,
        mask_column=cfg.dataset.mask_column,
        class_labels_column=cfg.dataset.class_labels_column,
        image_mode=cfg.dataset.image_mode,
        transform=transform,
        logger=logger,
    )


def build_dataloader(dataset, cfg, split: str, logger: logging.Logger | None = None) -> DataLoader:
    logger = logger or logging.getLogger(__name__)
    shuffle = bool(cfg.dataloader.shuffle and split == "train")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last,
        collate_fn=segmentation_collate_fn,
    )
    logger.info(
        "DataLoader ready: samples=%d batches=%d batch_size=%d shuffle=%s num_workers=%d pin_memory=%s",
        len(dataset),
        len(dataloader),
        cfg.dataloader.batch_size,
        shuffle,
        cfg.dataloader.num_workers,
        cfg.dataloader.pin_memory,
    )
    return dataloader


def segmentation_collate_fn(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "mask": torch.stack([sample["mask"] for sample in batch]),
        "class_labels": [sample["class_labels"] for sample in batch],
        "width": torch.tensor([sample["width"] for sample in batch], dtype=torch.long),
        "height": torch.tensor([sample["height"] for sample in batch], dtype=torch.long),
        "index": torch.tensor([sample["index"] for sample in batch], dtype=torch.long),
    }
