from __future__ import annotations

import gc
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.mars_bench_dataset import MarsBenchHFDataset
from src.data.sampling import build_image_level_sampler
from src.data.transforms import NumpySegmentationTransform


def resolve_num_workers(configured_workers: int | None) -> int:
    if configured_workers is not None:
        workers = int(configured_workers)
        if workers < 0:
            raise ValueError("num_workers must be >= 0")
        return workers
    return max(1, os.cpu_count() or 2)


def initialize_dataloader_worker(_worker_id: int) -> None:
    """Keep worker thread pools small and seed Python/NumPy deterministically."""

    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    gc.collect()


def segmentation_collate_fn(batch: list[dict]) -> dict[str, np.ndarray]:
    return {
        "image": np.stack([sample["image"] for sample in batch]),
        "mask": np.stack([sample["mask"] for sample in batch]),
    }


def _plain_mapping(value) -> dict:
    return {key: item for key, item in value.items()}


def resolve_dataset_config(cfg, dataset_name: str | None = None):
    selected_name = str(dataset_name or cfg.dataset_name)
    if selected_name not in cfg.datasets:
        raise ValueError(
            f"Unknown dataset_name={selected_name!r}; choose one of "
            f"{sorted(cfg.datasets.keys())}"
        )
    return selected_name, cfg.datasets[selected_name]


def validate_cross_dataset_configuration(cfg) -> tuple[str, object] | None:
    """Validate MSL<->MER zero-shot evaluation before any costly loading."""

    if type(cfg.run_cross_dataset_evaluation) is not bool:
        raise TypeError("run_cross_dataset_evaluation must be true or false")
    if not cfg.run_cross_dataset_evaluation:
        return None

    source_name, source = resolve_dataset_config(cfg)
    target_name = source.cross_dataset_name
    if str(source.family) != "marsseg" or target_name is None:
        raise ValueError(
            "Cross-dataset evaluation is valid only for marsseg_msl or marsseg_mer"
        )
    target_name, target = resolve_dataset_config(cfg, str(target_name))
    if str(target.family) != "marsseg":
        raise ValueError("Cross-dataset target must belong to the MarsSeg family")
    source_names = {int(key): str(value) for key, value in source.class_names.items()}
    target_names = {int(key): str(value) for key, value in target.class_names.items()}
    if source_names != target_names:
        raise ValueError("Source and target taxonomies differ; zero-shot evaluation aborted")
    if int(source.num_classes) != int(target.num_classes):
        raise ValueError("Source and target class counts differ")
    if target_name == source_name:
        raise ValueError("Cross-dataset target must differ from the source")
    return target_name, target


def resolve_hf_token(cfg, logger: logging.Logger | None = None) -> str | None:
    logger = logger or logging.getLogger(__name__)
    if not bool(cfg.huggingface.use_auth_token):
        return None
    env_name = str(cfg.huggingface.token_env)
    env_token = os.environ.get(env_name)
    configured = cfg.huggingface.token
    token = env_token or configured
    if token in (None, "", "HF_TOKEN_PLACEHOLDER"):
        logger.warning("No Hugging Face token configured; trying anonymous access")
        return None
    logger.info("Using Hugging Face token from %s", env_name if env_token else "config")
    return str(token)


def build_dataset(
    cfg,
    split: str,
    dataset_name: str | None = None,
    include_class_labels: bool = False,
    augment: bool = False,
    logger: logging.Logger | None = None,
) -> MarsBenchHFDataset:
    logger = logger or logging.getLogger(__name__)
    selected_name, dataset_cfg = resolve_dataset_config(cfg, dataset_name)
    augmentation = cfg.augmentation
    transform = NumpySegmentationTransform(
        size=tuple(cfg.image_size),
        horizontal_flip_prob=(
            float(augmentation.horizontal_flip_prob) if augment else 0.0
        ),
        vertical_flip_prob=(
            float(augmentation.vertical_flip_prob) if augment else 0.0
        ),
        random_rotate90_prob=(
            float(augmentation.random_rotate90_prob) if augment else 0.0
        ),
    )
    columns = cfg.dataset_columns
    return MarsBenchHFDataset(
        repo_id=str(dataset_cfg.repo_id),
        split=str(split),
        num_classes=int(dataset_cfg.num_classes),
        token=resolve_hf_token(cfg, logger),
        cache_dir=cfg.huggingface.cache_dir,
        transform=transform,
        include_class_labels=include_class_labels,
        dataset_name=selected_name,
        expected_split_sizes=_plain_mapping(dataset_cfg.expected_split_sizes),
        image_column=str(columns.image),
        mask_column=str(columns.mask),
        width_column=str(columns.width),
        height_column=str(columns.height),
        class_labels_column=str(columns.class_labels),
        logger=logger,
    )


def build_dataloader(
    dataset: MarsBenchHFDataset,
    cfg,
    split: str,
    shuffle: bool,
    sampler=None,
    logger: logging.Logger | None = None,
) -> DataLoader:
    logger = logger or logging.getLogger(__name__)
    is_train = str(split) == str(cfg.splits.train)
    configured_workers = (
        cfg.dataloader.train_num_workers
        if is_train
        else cfg.dataloader.eval_num_workers
    )
    num_workers = resolve_num_workers(configured_workers)
    kwargs = {
        "dataset": dataset,
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataloader.pin_memory),
        "collate_fn": segmentation_collate_fn,
        "worker_init_fn": initialize_dataloader_worker if num_workers > 0 else None,
        "generator": torch.Generator().manual_seed(int(cfg.seed)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(cfg.dataloader.prefetch_factor)
        kwargs["persistent_workers"] = bool(
            cfg.dataloader.persistent_workers and is_train
        )
    loader = DataLoader(**kwargs)
    logger.info(
        "DataLoader ready: dataset=%s repo=%s split=%s samples=%d batches=%d "
        "batch_size=%d shuffle=%s sampler=%s augmentation=%s workers=%d "
        "pin_memory=%s",
        dataset.dataset_name,
        dataset.repo_id,
        split,
        len(dataset),
        len(loader),
        int(cfg.batch_size),
        bool(shuffle and sampler is None),
        type(sampler).__name__ if sampler is not None else "none",
        bool(cfg.augmentation.enabled and is_train),
        num_workers,
        bool(cfg.dataloader.pin_memory),
    )
    return loader


def build_loader(
    cfg,
    split: str,
    shuffle: bool,
    dataset_name: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[MarsBenchHFDataset, DataLoader]:
    logger = logger or logging.getLogger(__name__)
    is_train = str(split) == str(cfg.splits.train)
    use_oversampling = bool(cfg.oversampling.enabled and is_train)
    use_augmentation = bool(cfg.augmentation.enabled and is_train)
    dataset = build_dataset(
        cfg,
        split=split,
        dataset_name=dataset_name,
        include_class_labels=use_oversampling,
        augment=use_augmentation,
        logger=logger,
    )
    selected_name, dataset_cfg = resolve_dataset_config(cfg, dataset_name)
    sampler = None
    if use_oversampling:
        sampler = build_image_level_sampler(
            dataset=dataset,
            class_names=_plain_mapping(dataset_cfg.class_names),
            num_classes=int(dataset_cfg.num_classes),
            power=float(cfg.oversampling.power),
            max_weight=float(cfg.oversampling.max_weight),
            num_samples_multiplier=float(cfg.oversampling.num_samples_multiplier),
            replacement=cfg.oversampling.replacement,
            excluded_class_ids=list(cfg.oversampling.excluded_class_ids),
            seed=int(cfg.seed),
            logger=logger,
        )
    dataset.release_class_labels()
    loader = build_dataloader(
        dataset,
        cfg,
        split=split,
        shuffle=shuffle,
        sampler=sampler,
        logger=logger,
    )
    logger.debug("Selected dataset %s", selected_name)
    return dataset, loader
