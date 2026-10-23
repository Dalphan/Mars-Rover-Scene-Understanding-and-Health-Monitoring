from __future__ import annotations

import hashlib
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
        input_size: tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1")
        local_files = tuple(str(Path(path).resolve()) for path in local_files)
        if not local_files:
            raise ValueError("local_files cannot be empty")
        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2 or self.input_size[0] != self.input_size[1]:
            raise ValueError("ImageNet penalty input_size must be square")
        self._local_file_names = tuple(Path(path).name for path in local_files)
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

    def _transform(self, image) -> torch.Tensor:
        image = image.convert("RGB")
        crop_size = self.input_size[0]
        image = TF.resize(
            image, [2 * crop_size, 2 * crop_size],
            interpolation=InterpolationMode.BILINEAR, antialias=True
        )
        if random.random() < 0.3:
            image = TF.rgb_to_grayscale(image, num_output_channels=3)
        image = TF.center_crop(image, list(self.input_size))
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
        return {
            "format": "imagenet_penalty_local_v1",
            "local_file_names": list(self._local_file_names),
            "input_size": list(self.input_size),
            "stream_state": self._stream.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict.get("format") != "imagenet_penalty_local_v1":
            print(
                "[ImageNet penalty] Restoring an unversioned stream state. "
                "This assumes the checkpoint uses the same cached shards and seed.",
                flush=True,
            )
            self._stream.load_state_dict(state_dict)
            return
        saved_files = tuple(state_dict.get("local_file_names", ()))
        if saved_files != self._local_file_names:
            raise RuntimeError(
                "The ImageNet penalty cache does not match the checkpoint. "
                "Use the same seed/shard configuration used by the run."
            )
        saved_input_size = tuple(state_dict.get("input_size", (256, 256)))
        if saved_input_size != self.input_size:
            raise RuntimeError(
                "The ImageNet penalty transform size does not match the checkpoint"
            )
        self._stream.load_state_dict(state_dict["stream_state"])


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
        input_size=tuple(int(value) for value in cfg.model.input_size),
    )
    return DataLoader(
        dataset, batch_size=1, num_workers=0,
        pin_memory=bool(cfg.dataloader.pin_memory),
    )


def _optional_tuple(value):
    """Convert an optional OmegaConf list to an immutable tuple."""
    return None if value is None else tuple(value)


def _pose_crop_mapping(value) -> dict[str, tuple[int, int, int, int] | None]:
    """Materialize pose crop boxes from OmegaConf without hiding unknown poses."""
    if value is None:
        return {}
    return {
        str(pose): (
            None if crop is None else tuple(int(coordinate) for coordinate in crop)
        )
        for pose, crop in value.items()
    }


def build_preprocessing(config: DictConfig) -> WheelPreprocessor:
    """Build preprocessing from one Hydra train or evaluation section."""
    return WheelPreprocessor(
        PreprocessingConfig(
            resize=_optional_tuple(config.resize),
            resize_shorter_side=config.resize_shorter_side,
            center_crop=_optional_tuple(config.center_crop),
            pose_crop_enabled=bool(config.get("pose_crop_enabled", False)),
            pose_crops=_pose_crop_mapping(config.get("pose_crops")),
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
    camera_poses: tuple[str, ...] | None = None,
) -> DataLoader:
    """Build one loader; only the training split is shuffled."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = CuriosityWheelDataset(
        root,
        split,
        preprocessing=preprocessing,
        camera_poses=camera_poses,
    )
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
    configured_camera_poses = cfg.dataset.get("camera_poses")
    if isinstance(configured_camera_poses, str):
        raise TypeError(
            "dataset.camera_poses must be a YAML list of pose names or null"
        )
    camera_poses = (
        None
        if configured_camera_poses is None
        else tuple(str(pose) for pose in configured_camera_poses)
    )

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
            camera_poses=camera_poses,
        )

    return (
        make_loader(str(cfg.dataset.splits.train), is_train=True),
        make_loader(str(cfg.dataset.splits.validation)),
        make_loader(str(cfg.dataset.splits.test)),
    )


def build_efficientad_train_calibration_loaders(
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader]:
    """Split clean training images into disjoint optimization/calibration sets.

    The split is stable by image id and seed. The calibration view always uses
    evaluation preprocessing so stochastic train transforms cannot contaminate
    the spatial clean baseline.
    """
    fraction = float(cfg.model.spatial_calibration_fraction)
    minimum = int(cfg.model.spatial_calibration_min_images)
    if not 0 < fraction < 1:
        raise ValueError("spatial_calibration_fraction must be in (0, 1)")
    if minimum < 1:
        raise ValueError("spatial_calibration_min_images must be at least 1")

    configured_camera_poses = cfg.dataset.get("camera_poses")
    camera_poses = (
        None
        if configured_camera_poses is None
        else tuple(str(pose) for pose in configured_camera_poses)
    )
    if camera_poses is None or len(camera_poses) != 1:
        raise ValueError(
            "spatial EfficientAD calibration requires exactly one camera pose"
        )
    split = str(cfg.dataset.splits.train)
    optimization_dataset = CuriosityWheelDataset(
        cfg.dataset.root,
        split,
        preprocessing=build_preprocessing(cfg.preprocessing.train),
        camera_poses=camera_poses,
    )
    calibration_dataset = CuriosityWheelDataset(
        cfg.dataset.root,
        split,
        preprocessing=build_preprocessing(cfg.preprocessing.evaluation),
        camera_poses=camera_poses,
    )
    if any(row["condition"] != "clean" for row in optimization_dataset.rows):
        raise ValueError("EfficientAD optimization/calibration split must be clean-only")

    sample_count = len(optimization_dataset.rows)
    calibration_count = max(minimum, int(round(sample_count * fraction)))
    if calibration_count >= sample_count:
        raise ValueError(
            "The EfficientAD calibration split leaves no optimization images; "
            "reduce spatial_calibration_fraction or spatial_calibration_min_images"
        )
    seed = int(cfg.seed)
    ordered_indices = sorted(
        range(sample_count),
        key=lambda index: hashlib.sha256(
            f"{seed}:{optimization_dataset.rows[index]['image_id']}".encode("utf-8")
        ).digest(),
    )
    calibration_indices = set(ordered_indices[:calibration_count])
    optimization_dataset.rows = [
        row for index, row in enumerate(optimization_dataset.rows)
        if index not in calibration_indices
    ]
    calibration_dataset.rows = [
        row for index, row in enumerate(calibration_dataset.rows)
        if index in calibration_indices
    ]

    common = {
        "num_workers": int(cfg.dataloader.num_workers),
        "pin_memory": bool(cfg.dataloader.pin_memory),
        "persistent_workers": int(cfg.dataloader.num_workers) > 0,
    }
    optimization_loader = DataLoader(
        optimization_dataset,
        batch_size=int(cfg.model.train_batch_size),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=int(cfg.model.evaluation_batch_size),
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    print(
        "[EfficientAD] Disjoint clean train split: "
        f"optimization={len(optimization_dataset):,}, "
        f"spatial_calibration={len(calibration_dataset):,}.",
        flush=True,
    )
    return optimization_loader, calibration_loader
