from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .curiosity_wheel_dataset import CuriosityWheelDataset
from .preprocessing import PreprocessingConfig, WheelPreprocessor


def select_imagenet_penalty_shards(
    *, total_shards: int, num_shards: int, seed: int,
) -> tuple[str, ...]:
    """Select a stable, sorted subset of ImageNet train Parquet shards."""
    if total_shards < 1:
        raise ValueError("total_shards must be at least 1")
    if not 1 <= num_shards <= total_shards:
        raise ValueError("num_shards must be between 1 and total_shards")
    indices = sorted(random.Random(seed).sample(range(total_shards), num_shards))
    return tuple(
        f"data/train-{index:05d}-of-{total_shards:05d}.parquet"
        for index in indices
    )


def prepare_imagenet_penalty_cache(
    *,
    cache_dir: str | Path,
    dataset_id: str,
    revision: str | None,
    token: str,
    total_shards: int,
    num_shards: int,
    seed: int,
    min_samples: int,
) -> tuple[Path, ...]:
    """Download a deterministic ImageNet subset and validate it locally."""
    if not token:
        raise ValueError(
            "ImageNet access requires a Hugging Face token accepted for "
            f"{dataset_id!r}"
        )
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "EfficientAD local caching requires huggingface_hub and pyarrow"
        ) from error

    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filenames = select_imagenet_penalty_shards(
        total_shards=total_shards, num_shards=num_shards, seed=seed
    )
    print(
        f"[ImageNet cache] Preparing {len(filenames)} deterministic shards in "
        f"{cache_dir}. Existing complete files are reused.",
        flush=True,
    )

    local_files: list[Path] = []
    for position, filename in enumerate(filenames, start=1):
        local_path = cache_dir / filename
        if local_path.is_file() and local_path.stat().st_size > 0:
            print(
                f"[ImageNet cache] [{position}/{len(filenames)}] Reusing "
                f"{local_path.name} ({local_path.stat().st_size / 2**20:.1f} MiB).",
                flush=True,
            )
        else:
            print(
                f"[ImageNet cache] [{position}/{len(filenames)}] Downloading "
                f"{filename}. A rerun resumes/reuses completed downloads.",
                flush=True,
            )
            try:
                downloaded = hf_hub_download(
                    repo_id=dataset_id,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    local_dir=cache_dir,
                )
            except Exception as error:
                raise RuntimeError(
                    f"Failed to cache {filename}. Rerun this cell to resume; "
                    "already completed shards will be reused."
                ) from error
            local_path = Path(downloaded).resolve()
        local_files.append(local_path)

    total_rows = 0
    total_bytes = 0
    for local_path in local_files:
        try:
            total_rows += parquet.ParquetFile(local_path).metadata.num_rows
        except Exception as error:
            raise RuntimeError(
                f"Cached shard is not a readable Parquet file: {local_path}. "
                "Delete only this file and rerun the cache cell."
            ) from error
        total_bytes += local_path.stat().st_size
    if total_rows < min_samples:
        raise RuntimeError(
            f"The {len(local_files)} cached shards contain {total_rows:,} samples, "
            f"but EfficientAD needs at least {min_samples:,}. Increase "
            "penalty_cache_num_shards and rerun."
        )
    print(
        f"[ImageNet cache] Ready: {len(local_files)} shards, {total_rows:,} "
        f"samples, {total_bytes / 2**30:.2f} GiB. Training reads local files only.",
        flush=True,
    )
    return tuple(local_files)


class ImageNetPenaltyDataset(IterableDataset):
    """Read cached ImageNet-1k train shards with EfficientAD's penalty transform."""

    def __init__(
        self, *, local_files: Sequence[str | Path], seed: int = 42,
        shuffle_buffer_size: int = 10_000,
    ) -> None:
        super().__init__()
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1")
        local_files = tuple(str(Path(path).resolve()) for path in local_files)
        if not local_files:
            raise ValueError("local_files cannot be empty")
        missing = [path for path in local_files if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing cached ImageNet shard: {missing[0]}")
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError(
                "EfficientAD requires the 'datasets' package for local ImageNet reads"
            ) from error
        print(
            f"[ImageNet penalty] Opening {len(local_files)} local Parquet shards; "
            f"shuffle_buffer={shuffle_buffer_size:,}.",
            flush=True,
        )
        dataset = load_dataset(
            "parquet",
            data_files={"train": list(local_files)},
            split="train",
            streaming=True,
        )
        self._stream = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
        print(
            "[ImageNet penalty] Local stream configured; fit() will not read "
            "ImageNet from the network.",
            flush=True,
        )

    @staticmethod
    def _transform(image) -> torch.Tensor:
        image = image.convert("RGB")
        image = TF.resize(
            image, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        if random.random() < 0.3:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        image = TF.center_crop(image, [256, 256])
        return TF.pil_to_tensor(image).float().div_(255.0)

    def __iter__(self):
        print(
            "[ImageNet penalty] Iterator started; requesting the first local sample...",
            flush=True,
        )
        for index, sample in enumerate(self._stream, start=1):
            if index == 1:
                print(
                    "[ImageNet penalty] First local sample received; applying the "
                    "EfficientAD penalty transform.",
                    flush=True,
                )
            yield {"image": self._transform(sample["image"])}
            if index % 1_000 == 0:
                print(
                    f"[ImageNet penalty] Local stream yielded {index:,} samples.",
                    flush=True,
                )

    def state_dict(self) -> dict:
        return self._stream.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._stream.load_state_dict(state_dict)


def build_imagenet_penalty_loader(cfg: DictConfig) -> DataLoader:
    """Build the resumable local ImageNet penalty stream required by EfficientAD."""
    token_env = str(cfg.model.penalty_token_env)
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Set {token_env} to a Hugging Face token with ImageNet-1k access"
        )
    local_files = prepare_imagenet_penalty_cache(
        cache_dir=str(cfg.model.penalty_cache_dir),
        dataset_id=str(cfg.model.penalty_dataset_id),
        revision=(
            None if cfg.model.penalty_dataset_revision is None
            else str(cfg.model.penalty_dataset_revision)
        ),
        token=token,
        total_shards=int(cfg.model.penalty_total_shards),
        num_shards=int(cfg.model.penalty_cache_num_shards),
        seed=int(cfg.seed),
        min_samples=int(cfg.model.max_steps),
    )
    print(
        f"[ImageNet penalty] Building local DataLoader with num_workers=0 and "
        f"pin_memory={bool(cfg.dataloader.pin_memory)}.",
        flush=True,
    )
    dataset = ImageNetPenaltyDataset(
        local_files=local_files,
        seed=int(cfg.seed),
        shuffle_buffer_size=int(cfg.model.penalty_shuffle_buffer_size),
    )
    return DataLoader(
        dataset, batch_size=1, num_workers=0,
        pin_memory=bool(cfg.dataloader.pin_memory),
    )


def _optional_tuple(value):
    """Convert an optional OmegaConf list to an immutable tuple."""
    return None if value is None else tuple(value)


def build_preprocessing(config: DictConfig) -> WheelPreprocessor:
    """Build preprocessing from one Hydra train or evaluation section."""
    return WheelPreprocessor(
        PreprocessingConfig(
            resize=_optional_tuple(config.resize),
            resize_shorter_side=config.resize_shorter_side,
            center_crop=_optional_tuple(config.center_crop),
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
        batch_size = cfg.model.get(
            "train_batch_size" if is_train else "evaluation_batch_size",
            cfg.dataloader.batch_size,
        )
        return build_dataloader(
            cfg.dataset.root,
            split,
            batch_size=int(batch_size),
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
