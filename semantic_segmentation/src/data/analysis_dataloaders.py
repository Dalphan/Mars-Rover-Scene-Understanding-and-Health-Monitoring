from __future__ import annotations

import logging
import os

import torch
from torch.utils.data import DataLoader

from src.data.s5mars_dataset import S5MarsHFDataset
from src.data.transforms import SegmentationTransform


def _num_workers(configured_workers=None) -> int:
    if configured_workers is not None:
        return int(configured_workers)
    cpu_count = os.cpu_count() or 2
    return max(1, min(4, cpu_count // 2))


def build_analysis_dataset(cfg, split: str, logger=None) -> S5MarsHFDataset:
    """Legacy tensor dataset kept specifically for dataset analysis."""

    logger = logger or logging.getLogger(__name__)
    env_token = os.environ.get("HF_TOKEN")
    token = env_token or cfg.huggingface.token
    if token in ("", "HF_TOKEN_PLACEHOLDER", None):
        logger.warning("No Hugging Face token configured; trying anonymous access")
        token = None
    transform = None
    if cfg.transforms.enabled:
        transform = SegmentationTransform(
            resize_enabled=cfg.transforms.resize.enabled,
            size=(cfg.transforms.resize.height, cfg.transforms.resize.width),
            normalize_enabled=cfg.transforms.normalize.enabled,
            mean=cfg.transforms.normalize.mean,
            std=cfg.transforms.normalize.std,
        )
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


def _analysis_collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "mask": torch.stack([sample["mask"] for sample in batch]),
        "class_labels": [sample["class_labels"] for sample in batch],
        "width": torch.tensor([sample["width"] for sample in batch], dtype=torch.long),
        "height": torch.tensor([sample["height"] for sample in batch], dtype=torch.long),
        "index": torch.tensor([sample["index"] for sample in batch], dtype=torch.long),
    }


def build_analysis_dataloader(dataset, cfg, split: str, logger=None) -> DataLoader:
    logger = logger or logging.getLogger(__name__)
    shuffle = bool(cfg.dataloader.shuffle and split == "train")
    workers = _num_workers(cfg.dataloader.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(cfg.dataloader.pin_memory),
        drop_last=bool(cfg.dataloader.drop_last),
        collate_fn=_analysis_collate,
    )
    logger.info(
        "Analysis DataLoader ready: samples=%d batches=%d workers=%d",
        len(dataset),
        len(loader),
        workers,
    )
    return loader
