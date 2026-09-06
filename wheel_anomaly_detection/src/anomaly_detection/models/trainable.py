from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from .base import AnomalyDetector, AnomalyPrediction


def _capture_training_rng_state(train_loader) -> dict[str, Any]:
    generator = getattr(train_loader, "generator", None)
    return {
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_loader_generator_state": (
            generator.get_state() if generator is not None else None
        ),
    }


def _restore_training_rng_state(checkpoint: dict[str, Any], train_loader) -> None:
    generator_state = checkpoint.get("train_loader_generator_state")
    generator = getattr(train_loader, "generator", None)
    if generator_state is not None:
        if generator is None:
            raise TypeError("The train loader cannot restore its generator state")
        generator.set_state(
            torch.as_tensor(generator_state, dtype=torch.uint8, device="cpu")
        )
    torch.set_rng_state(
        torch.as_tensor(checkpoint["torch_rng_state"], dtype=torch.uint8, device="cpu")
    )
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    cuda_rng_state = checkpoint.get("cuda_rng_state")
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state_all([
            torch.as_tensor(state, dtype=torch.uint8, device="cpu")
            for state in cuda_rng_state
        ])


def _save_checkpoint_atomic(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _clean_images(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    if "image" not in batch:
        raise KeyError("Every training batch must contain 'image'")
    if torch.any(torch.as_tensor(batch.get("label", 0)) != 0):
        raise ValueError("fit accepts only clean training images")
    return batch["image"].to(device, non_blocking=True)


def _image_auc(scores: list[float], labels: list[int]) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum(p > n for p in positives for n in negatives)
    ties = sum(p == n for p in positives for n in negatives)
    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


@torch.no_grad()
def _validation_auc(
    model, loader, device: torch.device, max_batches: int | None,
) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        prediction = model.predict(batch["image"].to(device, non_blocking=True))
        scores.extend(prediction.anomaly_score.cpu().tolist())
        labels.extend(torch.as_tensor(batch["label"]).int().tolist())
    return _image_auc(scores, labels)


class PDNSmall(nn.Module):
    """EfficientAD PDN-S, including the paper's internal ImageNet normalization."""

    def __init__(self, output_channels: int = 384) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, padding=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, output_channels, kernel_size=4)

    @staticmethod
    def normalize(images: torch.Tensor) -> torch.Tensor:
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (images - mean) / std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = self.normalize(images)
        value = F.relu(self.conv1(value))
        value = self.pool1(value)
        value = F.relu(self.conv2(value))
        value = self.pool2(value)
        value = F.relu(self.conv3(value))
        return self.conv4(value)


class EfficientADAutoencoder(nn.Module):
    """EfficientAD autoencoder with decoder geometry scaled from 256 px."""

    def __init__(self, output_channels: int = 384, output_size: int = 64) -> None:
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be positive")
        self.output_size = int(output_size)
        scale = self.output_size / 64
        self.decoder_sizes = tuple(
            max(1, round((size + 1) * scale) - 1)
            for size in (3, 8, 15, 32, 63, 127)
        )
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 8),
        )
        self.decoder_convs = nn.ModuleList([
            nn.Conv2d(64, 64, 4, padding=2),
            nn.Conv2d(64, 64, 4, padding=2),
            nn.Conv2d(64, 64, 4, padding=2),
            nn.Conv2d(64, 64, 4, padding=2),
            nn.Conv2d(64, 64, 4, padding=2),
            nn.Conv2d(64, 64, 4, padding=2),
        ])
        self.final_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.output = nn.Conv2d(64, output_channels, 3, padding=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        value = self.encoder(PDNSmall.normalize(images))
        for size, convolution in zip(self.decoder_sizes, self.decoder_convs):
            value = F.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)
            value = self.dropout(F.relu(convolution(value)))
        value = F.interpolate(
            value,
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.output(F.relu(self.final_conv(value)))


class EfficientAD(AnomalyDetector):
    """EfficientAD-S; 256 is official geometry, larger inputs scale the AE."""

    _NELSON_TO_LOCAL = {
        "0.weight": "conv1.weight", "0.bias": "conv1.bias",
        "3.weight": "conv2.weight", "3.bias": "conv2.bias",
        "6.weight": "conv3.weight", "6.bias": "conv3.bias",
        "8.weight": "conv4.weight", "8.bias": "conv4.bias",
    }

    def __init__(
        self, *, input_size: tuple[int, int] = (256, 256),
        teacher_weights_path: str | None = None,
        require_teacher_weights: bool = True, channels: int = 384,
        max_steps: int = 70_000, learning_rate: float = 1e-4,
        weight_decay: float = 1e-5, hard_quantile: float = 0.995,
        checkpoint_interval: int = 5_000, validation_interval: int = 1_000,
        early_stopping_patience: int = 8, early_stopping_min_steps: int = 0,
        early_stopping_min_relative_improvement: float = 0.001,
        lr_scheduler_patience: int = 3,
        fixed_training_duration: bool = False, mixed_precision: bool = True,
        spatial_calibration_enabled: bool = False,
        spatial_calibration_q_low: float = 0.90,
        spatial_calibration_q_high: float = 0.995,
        spatial_calibration_smoothing_sigma: float = 1.0,
        spatial_scale_floor_fraction: float = 0.25,
        static_roi_min_coverage: float = 0.50,
        image_score_topk_candidates: tuple[float, ...] = (
            0.0, 0.0005, 0.001, 0.0025, 0.005,
        ),
    ) -> None:
        super().__init__()
        input_size = tuple(int(value) for value in input_size)
        if (
            len(input_size) != 2
            or input_size[0] != input_size[1]
            or input_size[0] < 256
            or input_size[0] % 128 != 0
        ):
            raise ValueError(
                "EfficientAD input_size must be square, at least 256, "
                "and divisible by 128"
            )
        if channels < 1:
            raise ValueError("channels must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        if validation_interval < 1:
            raise ValueError("validation_interval must be at least 1")
        if early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1")
        if not 0 <= early_stopping_min_steps <= max_steps:
            raise ValueError("early_stopping_min_steps must be in [0, max_steps]")
        if not 0 <= early_stopping_min_relative_improvement < 1:
            raise ValueError(
                "early_stopping_min_relative_improvement must be in [0, 1)"
            )
        if lr_scheduler_patience < 0:
            raise ValueError("lr_scheduler_patience cannot be negative")
        if not 0 < hard_quantile <= 1:
            raise ValueError("hard_quantile must be in (0, 1]")
        if not 0 <= spatial_calibration_q_low < spatial_calibration_q_high <= 1:
            raise ValueError(
                "spatial calibration quantiles must satisfy 0 <= low < high <= 1"
            )
        if spatial_calibration_smoothing_sigma < 0:
            raise ValueError("spatial_calibration_smoothing_sigma cannot be negative")
        if spatial_scale_floor_fraction <= 0:
            raise ValueError("spatial_scale_floor_fraction must be positive")
        if not 0 < static_roi_min_coverage <= 1:
            raise ValueError("static_roi_min_coverage must be in (0, 1]")
        image_score_topk_candidates = tuple(
            float(value) for value in image_score_topk_candidates
        )
        if (
            not image_score_topk_candidates
            or len(set(image_score_topk_candidates))
            != len(image_score_topk_candidates)
            or any(value < 0 or value > 1 for value in image_score_topk_candidates)
        ):
            raise ValueError(
                "image_score_topk_candidates must be unique fractions in [0, 1]"
            )
        native_map_size = input_size[0] // 4
        self.teacher = PDNSmall(channels)
        self.student = PDNSmall(2 * channels)
        self.autoencoder = EfficientADAutoencoder(channels, native_map_size)
        self.teacher.requires_grad_(False)
        self.input_size = input_size
        self.native_map_size = native_map_size
        self.channels = channels
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hard_quantile = hard_quantile
        self.checkpoint_interval = checkpoint_interval
        self.validation_interval = validation_interval
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_steps = early_stopping_min_steps
        self.early_stopping_min_relative_improvement = (
            early_stopping_min_relative_improvement
        )
        self.lr_scheduler_patience = lr_scheduler_patience
        self.fixed_training_duration = bool(fixed_training_duration)
        self.mixed_precision = bool(mixed_precision)
        self.spatial_calibration_enabled = bool(spatial_calibration_enabled)
        self.spatial_calibration_q_low = float(spatial_calibration_q_low)
        self.spatial_calibration_q_high = float(spatial_calibration_q_high)
        self.spatial_calibration_smoothing_sigma = float(
            spatial_calibration_smoothing_sigma
        )
        self.spatial_scale_floor_fraction = float(spatial_scale_floor_fraction)
        self.static_roi_min_coverage = float(static_roi_min_coverage)
        self.image_score_topk_candidates = image_score_topk_candidates
        self.teacher_source = None
        if teacher_weights_path:
            self._load_teacher(Path(teacher_weights_path))
        elif require_teacher_weights:
            raise FileNotFoundError(
                "EfficientAD-S requires the nelson1425 teacher_small.pth checkpoint"
            )
        self.register_buffer("teacher_mean", torch.zeros(1, channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, channels, 1, 1))
        self.register_buffer("map_st_q90", torch.tensor(0.0))
        self.register_buffer("map_st_q995", torch.tensor(1.0))
        self.register_buffer("map_ae_q90", torch.tensor(0.0))
        self.register_buffer("map_ae_q995", torch.tensor(1.0))
        native_shape = (1, 1, native_map_size, native_map_size)
        self.register_buffer("map_st_spatial_q_low", torch.zeros(native_shape))
        self.register_buffer("map_st_spatial_q_high", torch.ones(native_shape))
        self.register_buffer("map_ae_spatial_q_low", torch.zeros(native_shape))
        self.register_buffer("map_ae_spatial_q_high", torch.ones(native_shape))
        self.register_buffer("static_roi", torch.ones(native_shape, dtype=torch.bool))
        self.register_buffer("selected_spatial_calibration", torch.tensor(False))
        self.register_buffer("selected_topk_fraction", torch.tensor(0.0))
        self.register_buffer("fitted", torch.tensor(False))
        self.fit_summary: dict[str, Any] = {}
        # Transient post-hoc overrides are deliberately excluded from state_dict:
        # diagnostic ablations must not alter or invalidate fitted checkpoints.
        self._diagnostic_spatial_override: bool | None = None
        self._diagnostic_static_roi_override: bool | None = None
        self._diagnostic_topk_fraction_override: float | None = None
        self._diagnostic_global_topk_pixels_override: int | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _load_teacher(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"EfficientAD teacher weights are missing: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("The nelson1425 teacher checkpoint is not a state dict")
        if set(state) == set(self._NELSON_TO_LOCAL):
            state = {self._NELSON_TO_LOCAL[key]: value for key, value in state.items()}
        expected = set(self.teacher.state_dict())
        if set(state) != expected:
            raise ValueError(
                "Teacher checkpoint is incompatible with nelson1425 PDN-S: "
                f"expected {sorted(expected)}, got {sorted(state)}"
            )
        self.teacher.load_state_dict(state, strict=True)
        self.teacher_source = str(path)

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted)

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "channels": self.channels,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hard_quantile": self.hard_quantile,
            "checkpoint_interval": self.checkpoint_interval,
            "validation_interval": self.validation_interval,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_steps": self.early_stopping_min_steps,
            "early_stopping_min_relative_improvement": (
                self.early_stopping_min_relative_improvement
            ),
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "fixed_training_duration": self.fixed_training_duration,
            "mixed_precision": self.mixed_precision,
            "spatial_calibration_enabled": self.spatial_calibration_enabled,
            "spatial_calibration_q_low": self.spatial_calibration_q_low,
            "spatial_calibration_q_high": self.spatial_calibration_q_high,
            "spatial_calibration_smoothing_sigma": (
                self.spatial_calibration_smoothing_sigma
            ),
            "spatial_scale_floor_fraction": self.spatial_scale_floor_fraction,
            "static_roi_min_coverage": self.static_roi_min_coverage,
            "image_score_topk_candidates": self.image_score_topk_candidates,
            "teacher_source": self.teacher_source,
        }

    def _validate_input_size(self, images: torch.Tensor, source: str) -> None:
        actual = tuple(images.shape[-2:])
        if actual != self.input_size:
            raise ValueError(
                f"EfficientAD expected {self.input_size} {source} images, got {actual}"
            )

    def _teacher(self, images: torch.Tensor) -> torch.Tensor:
        self._validate_input_size(images, "in-domain")
        value = self.teacher(images)
        return (value - self.teacher_mean) / self.teacher_std.clamp_min(1e-6)

    @staticmethod
    def _augment_autoencoder_input(images: torch.Tensor) -> torch.Tensor:
        factor = float(torch.empty(()).uniform_(0.8, 1.2))
        operation = int(torch.randint(3, ()).item())
        from torchvision.transforms import functional as transform_functional
        if operation == 0:
            return transform_functional.adjust_brightness(images, factor)
        if operation == 1:
            return transform_functional.adjust_contrast(images, factor)
        return transform_functional.adjust_saturation(images, factor)

    @torch.no_grad()
    def _raw_maps(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        teacher = self._teacher(images)
        student = self.student(images)
        autoencoder = self.autoencoder(images)
        map_st = (student[:, : self.channels] - teacher).square().mean(1, keepdim=True)
        map_ae = (
            student[:, self.channels :] - autoencoder
        ).square().mean(1, keepdim=True)
        return map_st, map_ae

    @torch.no_grad()
    def _clean_validation_loss(self, loader, device: torch.device) -> tuple[float, int]:
        """Evaluate only normal validation images with the in-domain train losses."""
        self.eval()
        total_loss = 0.0
        total_images = 0
        for batch in loader:
            labels = torch.as_tensor(batch.get("label", 0))
            clean = labels == 0
            if not clean.any():
                continue
            images = batch["image"][clean].to(device, non_blocking=True)
            teacher = self._teacher(images)
            student = self.student(images)
            distance = (student[:, : self.channels] - teacher).square()
            flattened = distance.flatten(1)
            thresholds = torch.quantile(
                flattened, self.hard_quantile, dim=1, keepdim=True
            )
            hard_mask = flattened >= thresholds
            hard_loss = (
                (flattened * hard_mask).sum(1)
                / hard_mask.sum(1).clamp_min(1)
            )
            autoencoder = self.autoencoder(images)
            student_autoencoder = student[:, self.channels :]
            autoencoder_loss = (
                (autoencoder - teacher).square().flatten(1).mean(1)
            )
            student_autoencoder_loss = (
                (student_autoencoder - autoencoder).square().flatten(1).mean(1)
            )
            per_image = hard_loss + autoencoder_loss + student_autoencoder_loss
            total_loss += float(per_image.sum())
            total_images += images.shape[0]
        if total_images == 0:
            raise ValueError(
                "early stopping requires at least one clean validation image"
            )
        return total_loss / total_images, total_images

    def _training_checkpoint(
        self, *, optimizer, scheduler, scaler, step: int, penalty_loader,
        train_epoch_generator_state=None, train_batches_into_epoch: int = 0,
        early_stopping_state: dict[str, Any] | None = None,
        optimization_complete: bool = False,
    ) -> dict[str, Any]:
        penalty_state = None
        penalty_dataset = getattr(penalty_loader, "dataset", None)
        if hasattr(penalty_dataset, "state_dict"):
            penalty_state = penalty_dataset.state_dict()
        return {
            "format": "efficientad_training_v4_spatial_calibration",
            "input_size": list(self.input_size),
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "step": step,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "train_epoch_generator_state": train_epoch_generator_state,
            "train_batches_into_epoch": train_batches_into_epoch,
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "penalty_dataset_state": penalty_state,
            "early_stopping_state": early_stopping_state,
            "fixed_training_duration": self.fixed_training_duration,
            "mixed_precision": self.mixed_precision,
            "optimization_complete": optimization_complete,
        }

    @staticmethod
    def _save_training_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
        _save_checkpoint_atomic(checkpoint, path)

    def fit(
        self, train_loader, *, device, validation_loader=None,
        calibration_loader=None, penalty_loader=None, work_dir=None, resume=False,
    ) -> EfficientAD:
        if penalty_loader is None:
            raise ValueError("EfficientAD training requires an ImageNet penalty_loader")
        if validation_loader is None:
            raise ValueError("EfficientAD calibration requires a normal validation_loader")
        if self.spatial_calibration_enabled and calibration_loader is None:
            raise ValueError(
                "spatial EfficientAD calibration requires a disjoint "
                "clean calibration_loader"
            )
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        device = torch.device(device)
        print(
            f"[EfficientAD] Starting fit: device={device}, "
            f"max_steps={self.max_steps:,}, resume={resume}, "
            f"training_mode={'fixed_duration' if self.fixed_training_duration else 'early_stopping'}, "
            f"validation_during_optimization={not self.fixed_training_duration}.",
            flush=True,
        )
        self.to(device)
        self.train(True)
        optimizer = torch.optim.Adam(
            [*self.student.parameters(), *self.autoencoder.parameters()],
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, int(0.95 * self.max_steps)),
                gamma=0.1,
            )
            if self.fixed_training_duration
            else torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1,
                patience=self.lr_scheduler_patience, min_lr=1e-6,
                threshold=self.early_stopping_min_relative_improvement,
                threshold_mode="rel",
            )
        )
        amp_enabled = self.mixed_precision and device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        print(
            f"[EfficientAD] Mixed precision: "
            f"{'FP16 enabled' if amp_enabled else 'disabled'}.",
            flush=True,
        )
        checkpoint_path = (
            None if work_dir is None else Path(work_dir) / "efficientad_training.ckpt"
        )
        step = 0
        resume_train_epoch_state = None
        resume_train_batches = 0
        checkpoint_rng_state = None
        checkpoint_python_rng_state = None
        checkpoint_cuda_rng_state = None
        best_state = None
        best_validation_loss = float("inf")
        best_step = 0
        checks_without_improvement = 0
        validation_history: list[dict[str, float | int]] = []
        optimization_complete = False
        stopped_early = False
        print(
            f"[EfficientAD] Training checkpoint: "
            f"{checkpoint_path if checkpoint_path is not None else 'disabled'}",
            flush=True,
        )
        if resume:
            if checkpoint_path is None or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "resume=true requires work_dir/efficientad_training.ckpt"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("format") != "efficientad_training_v4_spatial_calibration":
                raise RuntimeError(
                    "This run uses disjoint spatial calibration and pose crops; "
                    "start it with RESUME=False instead of loading a legacy checkpoint."
                )
            if bool(checkpoint.get("fixed_training_duration", False)) != (
                self.fixed_training_duration
            ):
                raise RuntimeError(
                    "The EfficientAD checkpoint training-duration mode differs "
                    "from the current configuration; start a new run with resume=false."
                )
            if bool(checkpoint.get("mixed_precision", False)) != self.mixed_precision:
                raise RuntimeError(
                    "The EfficientAD checkpoint mixed-precision mode differs "
                    "from the current configuration; start a new run with resume=false."
                )
            saved_input_size = tuple(checkpoint.get("input_size", (256, 256)))
            if saved_input_size != self.input_size:
                raise RuntimeError(
                    "The EfficientAD checkpoint input size differs from the "
                    "current configuration; start a new run with resume=false."
                )
            self.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            grad_scaler_state = checkpoint.get("grad_scaler")
            if grad_scaler_state:
                scaler.load_state_dict(grad_scaler_state)
            step = int(checkpoint["step"])
            early_state = checkpoint.get("early_stopping_state") or {}
            best_state = early_state.get("best_state")
            best_validation_loss = float(
                early_state.get("best_validation_loss", float("inf"))
            )
            best_step = int(early_state.get("best_step", 0))
            checks_without_improvement = int(
                early_state.get("checks_without_improvement", 0)
            )
            validation_history = list(early_state.get("validation_history", []))
            optimization_complete = bool(checkpoint.get("optimization_complete", False))
            stopped_early = bool(early_state.get("stopped_early", False))
            resume_train_epoch_state = checkpoint.get("train_epoch_generator_state")
            if resume_train_epoch_state is not None:
                resume_train_epoch_state = torch.as_tensor(
                    resume_train_epoch_state, dtype=torch.uint8, device="cpu"
                )
            resume_train_batches = int(checkpoint.get("train_batches_into_epoch", 0))
            if checkpoint.get("penalty_dataset_state") is not None:
                penalty_dataset = getattr(penalty_loader, "dataset", None)
                if not hasattr(penalty_dataset, "load_state_dict"):
                    raise TypeError("The penalty dataset cannot restore its stream state")
                penalty_dataset.load_state_dict(checkpoint["penalty_dataset_state"])
            checkpoint_rng_state = torch.as_tensor(
                checkpoint["torch_rng_state"], dtype=torch.uint8, device="cpu"
            )
            checkpoint_python_rng_state = checkpoint.get("python_rng_state")
            checkpoint_cuda_rng_state = checkpoint.get("cuda_rng_state")
            if checkpoint_cuda_rng_state is not None:
                checkpoint_cuda_rng_state = [
                    torch.as_tensor(state, dtype=torch.uint8, device="cpu")
                    for state in checkpoint_cuda_rng_state
                ]
            print(
                f"[EfficientAD] Resume state restored at step {step:,}; "
                f"train batches into epoch={resume_train_batches:,}; "
                f"optimization_complete={optimization_complete}.",
                flush=True,
            )

        if step == 0:
            print(
                "[EfficientAD] Computing teacher feature mean/std on the clean "
                "training set...",
                flush=True,
            )
            sums = torch.zeros(self.channels, device=device)
            squares = torch.zeros_like(sums)
            count = 0
            with torch.no_grad():
                for batch in tqdm(
                    train_loader, desc="EfficientAD teacher statistics",
                    unit="batch", leave=False,
                ):
                    value = self.teacher(_clean_images(batch, device))
                    sums += value.sum((0, 2, 3))
                    squares += value.square().sum((0, 2, 3))
                    count += value.shape[0] * value.shape[2] * value.shape[3]
            if count == 0:
                raise ValueError("train_loader produced no images")
            mean = sums / count
            variance = (squares / count - mean.square()).clamp_min(1e-6)
            self.teacher_mean.copy_(mean.view(1, -1, 1, 1))
            self.teacher_std.copy_(variance.sqrt().view(1, -1, 1, 1))
            print(
                f"[EfficientAD] Teacher statistics ready from {count:,} feature "
                "vectors.",
                flush=True,
            )
        else:
            print(
                "[EfficientAD] Reusing teacher statistics from the resume checkpoint.",
                flush=True,
            )

        train_generator = getattr(train_loader, "generator", None)
        if resume_train_epoch_state is not None and train_generator is not None:
            train_generator.set_state(resume_train_epoch_state)
        train_epoch_generator_state = (
            train_generator.get_state() if train_generator is not None else None
        )
        train_iterator = iter(train_loader)
        train_batches_into_epoch = 0
        if (
            resume_train_epoch_state is not None
            and step < self.max_steps
            and not optimization_complete
        ):
            for _ in range(resume_train_batches):
                try:
                    next(train_iterator)
                except StopIteration as error:
                    raise RuntimeError("Saved train-loader position is invalid") from error
                train_batches_into_epoch += 1
        elif resume_train_epoch_state is not None:
            train_batches_into_epoch = resume_train_batches
            print(
                "[EfficientAD] Optimization is already complete; skipping train-loader "
                "position replay and proceeding to calibration.",
                flush=True,
            )
        penalty_iterator = iter(penalty_loader)
        penalty_batch_seen = False
        print(
            "[EfficientAD] Local ImageNet penalty iterator created. The first "
            "next() call will fill its shuffle buffer from cached shards.",
            flush=True,
        )
        if checkpoint_rng_state is not None:
            torch.set_rng_state(checkpoint_rng_state)
            if checkpoint_python_rng_state is not None:
                random.setstate(checkpoint_python_rng_state)
            if torch.cuda.is_available() and checkpoint_cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(checkpoint_cuda_rng_state)
        progress = tqdm(
            total=self.max_steps, initial=step, desc="EfficientAD-S training",
            unit="step", dynamic_ncols=True,
        )
        while step < self.max_steps and not optimization_complete:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_epoch_generator_state = (
                    train_generator.get_state() if train_generator is not None else None
                )
                train_batches_into_epoch = 0
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as error:
                    raise ValueError("train_loader produced no images") from error
            if not penalty_batch_seen:
                print(
                    "[EfficientAD] Waiting for the first ImageNet penalty batch...",
                    flush=True,
                )
            try:
                penalty_batch = next(penalty_iterator)
            except StopIteration:
                penalty_iterator = iter(penalty_loader)
                try:
                    penalty_batch = next(penalty_iterator)
                except StopIteration as error:
                    raise ValueError("penalty_loader produced no images") from error
            if not penalty_batch_seen:
                penalty_batch_seen = True
                print(
                    f"[EfficientAD] First ImageNet penalty batch ready with shape "
                    f"{tuple(penalty_batch['image'].shape)}. Training can proceed.",
                    flush=True,
                )

            train_batches_into_epoch += 1
            images = _clean_images(batch, device)
            penalty_images = penalty_batch["image"].to(device, non_blocking=True)
            self._validate_input_size(penalty_images, "ImageNet penalty")
            augmented = self._augment_autoencoder_input(images)
            train_batch_size = images.shape[0]
            penalty_batch_size = penalty_images.shape[0]
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                teacher_pair = self._teacher(torch.cat((images, augmented), dim=0))
                teacher, teacher_augmented = teacher_pair.split(
                    (train_batch_size, train_batch_size), dim=0
                )
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                student_all = self.student(
                    torch.cat((images, augmented, penalty_images), dim=0)
                )
                student, student_augmented, student_penalty = student_all.split(
                    (train_batch_size, train_batch_size, penalty_batch_size), dim=0
                )
                autoencoder = self.autoencoder(augmented)

            distance = (
                student[:, : self.channels].float() - teacher.float()
            ).square()
            threshold = torch.quantile(distance.detach(), self.hard_quantile)
            loss_hard = distance[distance >= threshold].mean()
            loss_penalty = (
                student_penalty[:, : self.channels].float().square().mean()
            )
            student_autoencoder = student_augmented[:, self.channels :]
            loss_autoencoder = F.mse_loss(
                autoencoder.float(), teacher_augmented.float()
            )
            loss_student_autoencoder = F.mse_loss(
                student_autoencoder.float(), autoencoder.float()
            )
            loss = (
                loss_hard + loss_penalty + loss_autoencoder
                + loss_student_autoencoder
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if self.fixed_training_duration:
                scheduler.step()
            if step % 50 == 0 or step == self.max_steps:
                progress.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False,
                )
                progress.update(step - progress.n)

            should_stop = False
            if (
                not self.fixed_training_duration
                and (step % self.validation_interval == 0 or step == self.max_steps)
            ):
                validation_loss, validation_images = self._clean_validation_loss(
                    validation_loader, device
                )
                scheduler.step(validation_loss)
                improved = (
                    best_state is None
                    or validation_loss
                    < best_validation_loss
                    * (1.0 - self.early_stopping_min_relative_improvement)
                )
                if improved:
                    best_validation_loss = validation_loss
                    best_step = step
                    checks_without_improvement = 0
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.state_dict().items()
                    }
                else:
                    checks_without_improvement += 1
                validation_history.append({
                    "step": step,
                    "clean_validation_loss": validation_loss,
                    "clean_validation_images": validation_images,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "improved": int(improved),
                })
                print(
                    f"[EfficientAD] Clean validation at step {step:,}: "
                    f"loss={validation_loss:.6g}, best={best_validation_loss:.6g} "
                    f"at step {best_step:,}, bad_checks="
                    f"{checks_without_improvement}/{self.early_stopping_patience}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}.",
                    flush=True,
                )
                self.train(True)
                should_stop = (
                    step >= self.early_stopping_min_steps
                    and checks_without_improvement >= self.early_stopping_patience
                )
                if should_stop:
                    stopped_early = True
                    print(
                        f"[EfficientAD] Early stopping at step {step:,}; "
                        f"restoring best step {best_step:,}.",
                        flush=True,
                    )

            early_stopping_state = {
                "best_state": best_state,
                "best_validation_loss": best_validation_loss,
                "best_step": best_step,
                "checks_without_improvement": checks_without_improvement,
                "validation_history": validation_history,
                "stopped_early": stopped_early,
            }
            if (
                checkpoint_path is not None
                and (
                    step % self.checkpoint_interval == 0
                    or step == self.max_steps
                    or should_stop
                )
            ):
                self._save_training_checkpoint(
                    self._training_checkpoint(
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        step=step,
                        penalty_loader=penalty_loader,
                        train_epoch_generator_state=train_epoch_generator_state,
                        train_batches_into_epoch=train_batches_into_epoch,
                        early_stopping_state=early_stopping_state,
                        optimization_complete=should_stop,
                    ),
                    checkpoint_path,
                )
                print(
                    f"[EfficientAD] Checkpoint saved at step {step:,}: "
                    f"{checkpoint_path}",
                    flush=True,
                )
            if should_stop:
                break

        if step > progress.n:
            progress.update(step - progress.n)
        progress.close()
        optimization_step = step
        if self.fixed_training_duration:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.state_dict().items()
            }
            best_step = optimization_step
            print(
                f"[EfficientAD] Fixed-duration optimization complete at "
                f"step {best_step:,}; validation was not used for model selection.",
                flush=True,
            )
        else:
            if best_state is None:
                raise RuntimeError(
                    "EfficientAD completed without a validation checkpoint"
                )
            self.load_state_dict(best_state)
            print(
                f"[EfficientAD] Selected clean-validation checkpoint from step "
                f"{best_step:,} (loss={best_validation_loss:.6g}).",
                flush=True,
            )
        print(
            "[EfficientAD] Optimization complete; starting clean calibration "
            f"from {'held-out training data' if self.spatial_calibration_enabled else 'validation'}.",
            flush=True,
        )
        self.eval()
        calibration_source = (
            calibration_loader
            if self.spatial_calibration_enabled
            else validation_loader
        )
        st_maps: list[torch.Tensor] = []
        ae_maps: list[torch.Tensor] = []
        roi_masks: list[torch.Tensor] = []
        normal_images = 0
        with torch.no_grad():
            for batch in tqdm(
                calibration_source, desc="EfficientAD calibration",
                unit="batch", leave=False,
            ):
                labels = torch.as_tensor(batch.get("label", 0))
                clean = labels == 0
                if clean.any():
                    images = batch["image"][clean].to(device, non_blocking=True)
                    map_st, map_ae = self._raw_maps(images)
                    st_maps.append(map_st.cpu())
                    ae_maps.append(map_ae.cpu())
                    if self.spatial_calibration_enabled:
                        if "target_mask" not in batch:
                            raise KeyError(
                                "spatial calibration requires target_mask in every batch"
                            )
                        target_mask = batch["target_mask"][clean].float()
                        target_mask = F.interpolate(
                            target_mask,
                            size=map_st.shape[-2:],
                            mode="nearest",
                        )
                        roi_masks.append(target_mask.cpu() > 0)
                    normal_images += images.shape[0]
        if not st_maps:
            raise ValueError("calibration requires at least one normal validation image")
        st_tensor = torch.cat(st_maps).float()
        ae_tensor = torch.cat(ae_maps).float()
        st_q90 = torch.quantile(st_tensor.flatten(), 0.90)
        st_q995 = torch.maximum(
            torch.quantile(st_tensor.flatten(), 0.995), st_q90 + 1e-6
        )
        ae_q90 = torch.quantile(ae_tensor.flatten(), 0.90)
        ae_q995 = torch.maximum(
            torch.quantile(ae_tensor.flatten(), 0.995), ae_q90 + 1e-6
        )
        self.map_st_q90.copy_(st_q90.to(device))
        self.map_st_q995.copy_(st_q995.to(device))
        self.map_ae_q90.copy_(ae_q90.to(device))
        self.map_ae_q995.copy_(ae_q995.to(device))
        if self.spatial_calibration_enabled:
            spatial = self._build_spatial_calibration(
                st_tensor, ae_tensor, torch.cat(roi_masks).float()
            )
            self.map_st_spatial_q_low.copy_(spatial["st_q_low"].to(device))
            self.map_st_spatial_q_high.copy_(spatial["st_q_high"].to(device))
            self.map_ae_spatial_q_low.copy_(spatial["ae_q_low"].to(device))
            self.map_ae_spatial_q_high.copy_(spatial["ae_q_high"].to(device))
            self.static_roi.copy_(spatial["roi"].to(device))
            scoring_ablation = self._select_validation_scoring(
                validation_loader, device
            )
        else:
            scoring_ablation = {
                "selection_split": "none",
                "selected": "global_max",
                "validation": {},
            }
        print(
            f"[EfficientAD] Calibration complete on {normal_images:,} normal images: "
            f"ST(q90={st_q90.item():.6g}, q99.5={st_q995.item():.6g}), "
            f"AE(q90={ae_q90.item():.6g}, q99.5={ae_q995.item():.6g}).",
            flush=True,
        )
        self.fitted.fill_(True)
        self.fit_summary = {
            "steps": optimization_step,
            "selected_step": best_step,
            "stopped_early": stopped_early,
            "fixed_training_duration": self.fixed_training_duration,
            "mixed_precision": self.mixed_precision,
            "best_clean_validation_loss": (
                None if self.fixed_training_duration else best_validation_loss
            ),
            "validation_interval": (
                None if self.fixed_training_duration else self.validation_interval
            ),
            "early_stopping_patience": (
                None if self.fixed_training_duration else self.early_stopping_patience
            ),
            "early_stopping_min_steps": (
                None if self.fixed_training_duration else self.early_stopping_min_steps
            ),
            "early_stopping_min_relative_improvement": (
                None
                if self.fixed_training_duration
                else self.early_stopping_min_relative_improvement
            ),
            "validation_history": validation_history,
            "normal_calibration_images": normal_images,
            "calibration_pixels_per_map": st_tensor.numel(),
            "spatial_calibration_enabled": self.spatial_calibration_enabled,
            "static_roi_fraction": float(self.static_roi.float().mean()),
            "image_score_ablation": scoring_ablation,
        }
        if checkpoint_path is not None:
            self._save_training_checkpoint(
                self._training_checkpoint(
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    step=optimization_step, penalty_loader=penalty_loader,
                    train_epoch_generator_state=train_epoch_generator_state,
                    train_batches_into_epoch=train_batches_into_epoch,
                    early_stopping_state={
                        "best_state": best_state,
                        "best_validation_loss": best_validation_loss,
                        "best_step": best_step,
                        "checks_without_improvement": checks_without_improvement,
                        "validation_history": validation_history,
                        "stopped_early": stopped_early,
                    },
                    optimization_complete=True,
                ),
                checkpoint_path,
            )
        return self

    def _smooth_spatial_statistic(self, value: torch.Tensor) -> torch.Tensor:
        sigma = self.spatial_calibration_smoothing_sigma
        if sigma == 0:
            return value
        kernel = min(63, 2 * math.ceil(3 * sigma) + 1)
        if kernel % 2 == 0:
            kernel -= 1
        return gaussian_blur(value, [kernel, kernel], [sigma, sigma])

    def _build_spatial_calibration(
        self,
        st_tensor: torch.Tensor,
        ae_tensor: torch.Tensor,
        roi_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        def statistics(
            values: torch.Tensor,
            global_low: torch.Tensor,
            global_high: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            low = torch.quantile(
                values, self.spatial_calibration_q_low, dim=0
            ).unsqueeze(0)
            high = torch.quantile(
                values, self.spatial_calibration_q_high, dim=0
            ).unsqueeze(0)
            low = self._smooth_spatial_statistic(low)
            high = self._smooth_spatial_statistic(high)
            global_scale = (global_high - global_low).clamp_min(1e-6)
            scale_floor = global_scale * self.spatial_scale_floor_fraction
            high = low + (high - low).clamp_min(scale_floor)
            return low, high

        st_low, st_high = statistics(
            st_tensor, self.map_st_q90.cpu(), self.map_st_q995.cpu()
        )
        ae_low, ae_high = statistics(
            ae_tensor, self.map_ae_q90.cpu(), self.map_ae_q995.cpu()
        )
        roi = (
            roi_masks.mean(dim=0, keepdim=True)
            >= self.static_roi_min_coverage
        )
        if not roi.any():
            raise ValueError(
                "static ROI is empty; lower static_roi_min_coverage or audit masks"
            )
        return {
            "st_q_low": st_low,
            "st_q_high": st_high,
            "ae_q_low": ae_low,
            "ae_q_high": ae_high,
            "roi": roi,
        }

    @staticmethod
    def _calibrate_map(
        value: torch.Tensor, q90: torch.Tensor, q995: torch.Tensor,
    ) -> torch.Tensor:
        return 0.1 * (value - q90) / (q995 - q90).clamp_min(1e-6)

    def _combine_calibrated_maps(
        self,
        map_st: torch.Tensor,
        map_ae: torch.Tensor,
        *,
        spatial: bool,
    ) -> torch.Tensor:
        if spatial:
            return 0.5 * (
                self._calibrate_map(
                    map_st,
                    self.map_st_spatial_q_low,
                    self.map_st_spatial_q_high,
                )
                + self._calibrate_map(
                    map_ae,
                    self.map_ae_spatial_q_low,
                    self.map_ae_spatial_q_high,
                )
            )
        return 0.5 * (
            self._calibrate_map(map_st, self.map_st_q90, self.map_st_q995)
            + self._calibrate_map(map_ae, self.map_ae_q90, self.map_ae_q995)
        )

    def _roi_at_size(self, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(
            self.static_roi.float(), size=size, mode="nearest"
        ).bool()

    def _aggregate_image_score(
        self,
        anomaly_map: torch.Tensor,
        *,
        use_static_roi: bool,
        topk_fraction: float,
    ) -> torch.Tensor:
        if use_static_roi:
            roi = self._roi_at_size(anomaly_map.shape[-2:]).expand(
                anomaly_map.shape[0], -1, -1, -1
            )
            values = anomaly_map.masked_select(roi).view(anomaly_map.shape[0], -1)
        else:
            values = anomaly_map.flatten(1)
        if topk_fraction == 0:
            return values.amax(1)
        k = max(1, math.ceil(values.shape[1] * topk_fraction))
        return values.topk(k, dim=1).values.mean(1)

    @staticmethod
    def _topk_candidate_name(fraction: float) -> str:
        if fraction == 0:
            return "spatial_max"
        return f"spatial_topk_{fraction:g}".replace(".", "_")

    @torch.no_grad()
    def _select_validation_scoring(
        self, validation_loader, device: torch.device,
    ) -> dict[str, Any]:
        candidates = [("global_max", False, 0.0)] + [
            (self._topk_candidate_name(fraction), True, fraction)
            for fraction in self.image_score_topk_candidates
        ]
        scores = {name: [] for name, _, _ in candidates}
        labels: list[int] = []
        self.eval()
        for batch in tqdm(
            validation_loader,
            desc="EfficientAD image-score ablation",
            unit="batch",
            leave=False,
        ):
            images = batch["image"].to(device, non_blocking=True)
            map_st, map_ae = self._raw_maps(images)
            native_maps = {
                False: self._combine_calibrated_maps(
                    map_st, map_ae, spatial=False
                ),
                True: self._combine_calibrated_maps(
                    map_st, map_ae, spatial=True
                ) * self.static_roi,
            }
            resized_maps = {
                spatial: F.interpolate(
                    value,
                    images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for spatial, value in native_maps.items()
            }
            for name, spatial, fraction in candidates:
                batch_scores = self._aggregate_image_score(
                    resized_maps[spatial],
                    use_static_roi=spatial,
                    topk_fraction=fraction,
                )
                scores[name].extend(batch_scores.cpu().tolist())
            labels.extend(torch.as_tensor(batch["label"]).int().tolist())

        validation = {
            name: {"image_auroc": _image_auc(values, labels)}
            for name, values in scores.items()
        }
        if not labels or any(
            not math.isfinite(result["image_auroc"])
            for result in validation.values()
        ):
            raise ValueError(
                "image-score selection requires both clean and anomalous "
                "validation images"
            )
        selected_name, selected_spatial, selected_fraction = max(
            candidates,
            key=lambda candidate: validation[candidate[0]]["image_auroc"],
        )
        self.selected_spatial_calibration.fill_(selected_spatial)
        self.selected_topk_fraction.fill_(selected_fraction)
        print(
            f"[EfficientAD] Validation selected {selected_name}: "
            f"image_AUROC={validation[selected_name]['image_auroc']:.6f}.",
            flush=True,
        )
        return {
            "selection_split": "validation",
            "selection_metric": "image_auroc",
            "selected": selected_name,
            "selected_spatial_calibration": selected_spatial,
            "selected_topk_fraction": selected_fraction,
            "validation": validation,
        }

    def configure_diagnostic_inference(self, mode: str) -> None:
        """Temporarily configure one calibration/ROI mode with max scoring."""
        modes = {
            "global": (False, False),
            "global_roi": (False, True),
            "spatial_roi": (True, True),
        }
        if mode not in modes:
            raise ValueError(
                f"Unknown EfficientAD diagnostic mode {mode!r}; "
                f"expected one of {tuple(modes)}"
            )
        spatial, use_static_roi = modes[mode]
        self._diagnostic_spatial_override = spatial
        self._diagnostic_static_roi_override = use_static_roi
        self._diagnostic_topk_fraction_override = 0.0
        self._diagnostic_global_topk_pixels_override = None

    def configure_diagnostic_global_topk_pixels(
        self, topk_pixels: int | None,
    ) -> None:
        """Use current resized max or a fixed number of native global pixels."""
        if topk_pixels is not None and topk_pixels < 1:
            raise ValueError("topk_pixels must be a positive integer or None")
        self._diagnostic_global_topk_pixels_override = topk_pixels

    def clear_diagnostic_inference(self) -> None:
        """Restore inference selected on validation during fit."""
        self._diagnostic_spatial_override = None
        self._diagnostic_static_roi_override = None
        self._diagnostic_topk_fraction_override = None
        self._diagnostic_global_topk_pixels_override = None

    @torch.no_grad()
    def predict_with_raw(self, images: torch.Tensor):
        if not self.is_fitted:
            raise RuntimeError("EfficientAD must be fitted before predict")
        map_st, map_ae = self._raw_maps(images)
        spatial = (
            bool(self.selected_spatial_calibration)
            if self._diagnostic_spatial_override is None
            else self._diagnostic_spatial_override
        )
        use_static_roi = (
            spatial
            if self._diagnostic_static_roi_override is None
            else self._diagnostic_static_roi_override
        )
        topk_fraction = (
            float(self.selected_topk_fraction)
            if self._diagnostic_topk_fraction_override is None
            else self._diagnostic_topk_fraction_override
        )
        anomaly_map = self._combine_calibrated_maps(
            map_st, map_ae, spatial=spatial
        )
        if use_static_roi:
            anomaly_map = anomaly_map * self.static_roi
        raw_map = F.interpolate(
            anomaly_map,
            images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        normalized_map = 0.5 + torch.atan(10.0 * raw_map) / math.pi
        if use_static_roi:
            normalized_map = normalized_map * self._roi_at_size(
                images.shape[-2:]
            )
        native_topk_pixels = self._diagnostic_global_topk_pixels_override
        if native_topk_pixels is None:
            raw_score = self._aggregate_image_score(
                raw_map,
                use_static_roi=use_static_roi,
                topk_fraction=topk_fraction,
            )
        else:
            if spatial or use_static_roi:
                raise RuntimeError(
                    "Native global top-k diagnostics require global calibration "
                    "without a static ROI"
                )
            native_values = anomaly_map.flatten(1)
            if native_topk_pixels > native_values.shape[1]:
                raise ValueError(
                    "topk_pixels exceeds the number of native anomaly-map pixels"
                )
            raw_score = native_values.topk(
                native_topk_pixels, dim=1
            ).values.mean(1)
        normalized_score = 0.5 + torch.atan(10.0 * raw_score) / math.pi
        return AnomalyPrediction(normalized_score, normalized_map), raw_score, raw_map

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        return self.predict_with_raw(images)[0]
