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
def _validation_auc(model, loader, device: torch.device, max_batches: int) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
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
    """Autoencoder from EfficientAD Table 8 for 256 x 256 inputs."""

    def __init__(self, output_channels: int = 384) -> None:
        super().__init__()
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
        for size, convolution in zip((3, 8, 15, 32, 63, 127), self.decoder_convs):
            value = F.interpolate(value, size=(size, size), mode="bilinear", align_corners=False)
            value = self.dropout(F.relu(convolution(value)))
        value = F.interpolate(value, size=(64, 64), mode="bilinear", align_corners=False)
        return self.output(F.relu(self.final_conv(value)))


class EfficientAD(AnomalyDetector):
    """Paper-faithful EfficientAD-S training and inference."""

    _NELSON_TO_LOCAL = {
        "0.weight": "conv1.weight", "0.bias": "conv1.bias",
        "3.weight": "conv2.weight", "3.bias": "conv2.bias",
        "6.weight": "conv3.weight", "6.bias": "conv3.bias",
        "8.weight": "conv4.weight", "8.bias": "conv4.bias",
    }

    def __init__(
        self, *, teacher_weights_path: str | None = None,
        require_teacher_weights: bool = True, channels: int = 384,
        max_steps: int = 70_000, learning_rate: float = 1e-4,
        weight_decay: float = 1e-5, hard_quantile: float = 0.999,
        checkpoint_interval: int = 1_000,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be at least 1")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        if not 0 < hard_quantile <= 1:
            raise ValueError("hard_quantile must be in (0, 1]")
        self.teacher = PDNSmall(channels)
        self.student = PDNSmall(2 * channels)
        self.autoencoder = EfficientADAutoencoder(channels)
        self.teacher.requires_grad_(False)
        self.channels = channels
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hard_quantile = hard_quantile
        self.checkpoint_interval = checkpoint_interval
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
        self.register_buffer("fitted", torch.tensor(False))
        self.fit_summary: dict[str, Any] = {}

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
            "channels": self.channels,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "hard_quantile": self.hard_quantile,
            "checkpoint_interval": self.checkpoint_interval,
            "teacher_source": self.teacher_source,
        }

    def _teacher(self, images: torch.Tensor) -> torch.Tensor:
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

    def _training_checkpoint(
        self, *, optimizer, scheduler, step: int, penalty_loader,
        train_epoch_generator_state=None, train_batches_into_epoch: int = 0,
    ) -> dict[str, Any]:
        penalty_state = None
        penalty_dataset = getattr(penalty_loader, "dataset", None)
        if hasattr(penalty_dataset, "state_dict"):
            penalty_state = penalty_dataset.state_dict()
        return {
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
            "train_epoch_generator_state": train_epoch_generator_state,
            "train_batches_into_epoch": train_batches_into_epoch,
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "penalty_dataset_state": penalty_state,
        }

    @staticmethod
    def _save_training_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(path)

    def fit(
        self, train_loader, *, device, validation_loader=None,
        penalty_loader=None, work_dir=None, resume=False,
    ) -> EfficientAD:
        if penalty_loader is None:
            raise ValueError("EfficientAD training requires an ImageNet penalty_loader")
        if validation_loader is None:
            raise ValueError("EfficientAD calibration requires a normal validation_loader")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        device = torch.device(device)
        print(
            f"[EfficientAD] Starting fit: device={device}, max_steps={self.max_steps:,}, "
            f"resume={resume}, checkpoint_interval={self.checkpoint_interval:,}.",
            flush=True,
        )
        self.to(device)
        self.train(True)
        optimizer = torch.optim.Adam(
            [*self.student.parameters(), *self.autoencoder.parameters()],
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, int(0.95 * self.max_steps)), gamma=0.1
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
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            self.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            step = int(checkpoint["step"])
            resume_train_epoch_state = checkpoint.get("train_epoch_generator_state")
            resume_train_batches = int(checkpoint.get("train_batches_into_epoch", 0))
            if checkpoint.get("penalty_dataset_state") is not None:
                penalty_dataset = getattr(penalty_loader, "dataset", None)
                if not hasattr(penalty_dataset, "load_state_dict"):
                    raise TypeError("The penalty dataset cannot restore its stream state")
                penalty_dataset.load_state_dict(checkpoint["penalty_dataset_state"])
            checkpoint_rng_state = checkpoint["torch_rng_state"]
            checkpoint_python_rng_state = checkpoint.get("python_rng_state")
            checkpoint_cuda_rng_state = checkpoint.get("cuda_rng_state")
            print(
                f"[EfficientAD] Resume state restored at step {step:,}; "
                f"train batches into epoch={resume_train_batches:,}.",
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
        if resume_train_epoch_state is not None:
            for _ in range(resume_train_batches):
                try:
                    next(train_iterator)
                except StopIteration as error:
                    raise RuntimeError("Saved train-loader position is invalid") from error
                train_batches_into_epoch += 1
        penalty_iterator = iter(penalty_loader)
        penalty_batch_seen = False
        print(
            "[EfficientAD] ImageNet penalty iterator created. The first next() call "
            "will perform the initial remote read.",
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
        while step < self.max_steps:
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
            with torch.no_grad():
                teacher = self._teacher(images)
            student = self.student(images)
            distance = (student[:, : self.channels] - teacher).square()
            threshold = torch.quantile(distance.detach(), self.hard_quantile)
            loss_hard = distance[distance >= threshold].mean()
            loss_penalty = self.student(penalty_images)[:, : self.channels].square().mean()

            augmented = self._augment_autoencoder_input(images)
            with torch.no_grad():
                teacher_augmented = self._teacher(augmented)
            autoencoder = self.autoencoder(augmented)
            student_autoencoder = self.student(augmented)[:, self.channels :]
            loss_autoencoder = F.mse_loss(autoencoder, teacher_augmented)
            loss_student_autoencoder = F.mse_loss(student_autoencoder, autoencoder)
            loss = (
                loss_hard + loss_penalty + loss_autoencoder
                + loss_student_autoencoder
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            progress.update(1)
            if step % 50 == 0 or step == self.max_steps:
                progress.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}", refresh=False,
                )
            if (
                checkpoint_path is not None
                and (step % self.checkpoint_interval == 0 or step == self.max_steps)
            ):
                self._save_training_checkpoint(
                    self._training_checkpoint(
                        optimizer=optimizer, scheduler=scheduler, step=step,
                        penalty_loader=penalty_loader,
                        train_epoch_generator_state=train_epoch_generator_state,
                        train_batches_into_epoch=train_batches_into_epoch,
                    ),
                    checkpoint_path,
                )
                print(
                    f"[EfficientAD] Checkpoint saved at step {step:,}: "
                    f"{checkpoint_path}",
                    flush=True,
                )

        progress.close()
        print(
            "[EfficientAD] Optimization complete; starting clean-validation "
            "quantile calibration.",
            flush=True,
        )
        self.eval()
        st_values: list[torch.Tensor] = []
        ae_values: list[torch.Tensor] = []
        normal_images = 0
        with torch.no_grad():
            for batch in tqdm(
                validation_loader, desc="EfficientAD calibration",
                unit="batch", leave=False,
            ):
                labels = torch.as_tensor(batch.get("label", 0))
                clean = labels == 0
                if clean.any():
                    images = batch["image"][clean].to(device, non_blocking=True)
                    map_st, map_ae = self._raw_maps(images)
                    map_st = F.interpolate(
                        map_st, images.shape[-2:], mode="bilinear", align_corners=False
                    )
                    map_ae = F.interpolate(
                        map_ae, images.shape[-2:], mode="bilinear", align_corners=False
                    )
                    st_values.append(map_st.flatten().cpu())
                    ae_values.append(map_ae.flatten().cpu())
                    normal_images += images.shape[0]
        if not st_values:
            raise ValueError("calibration requires at least one normal validation image")
        st_tensor = torch.cat(st_values).float()
        ae_tensor = torch.cat(ae_values).float()
        st_q90 = torch.quantile(st_tensor, 0.90)
        st_q995 = torch.maximum(torch.quantile(st_tensor, 0.995), st_q90 + 1e-6)
        ae_q90 = torch.quantile(ae_tensor, 0.90)
        ae_q995 = torch.maximum(torch.quantile(ae_tensor, 0.995), ae_q90 + 1e-6)
        self.map_st_q90.copy_(st_q90.to(device))
        self.map_st_q995.copy_(st_q995.to(device))
        self.map_ae_q90.copy_(ae_q90.to(device))
        self.map_ae_q995.copy_(ae_q995.to(device))
        print(
            f"[EfficientAD] Calibration complete on {normal_images:,} normal images: "
            f"ST(q90={st_q90.item():.6g}, q99.5={st_q995.item():.6g}), "
            f"AE(q90={ae_q90.item():.6g}, q99.5={ae_q995.item():.6g}).",
            flush=True,
        )
        self.fitted.fill_(True)
        self.fit_summary = {
            "steps": step,
            "normal_calibration_images": normal_images,
            "calibration_pixels_per_map": st_tensor.numel(),
            "scheduler_step_size": max(1, int(0.95 * self.max_steps)),
        }
        if checkpoint_path is not None:
            self._save_training_checkpoint(
                self._training_checkpoint(
                    optimizer=optimizer, scheduler=scheduler, step=step,
                    penalty_loader=penalty_loader,
                    train_epoch_generator_state=train_epoch_generator_state,
                    train_batches_into_epoch=train_batches_into_epoch,
                ),
                checkpoint_path,
            )
        return self

    @staticmethod
    def _calibrate_map(
        value: torch.Tensor, q90: torch.Tensor, q995: torch.Tensor,
    ) -> torch.Tensor:
        return 0.1 * (value - q90) / (q995 - q90).clamp_min(1e-6)

    @torch.no_grad()
    def predict_with_raw(self, images: torch.Tensor):
        if not self.is_fitted:
            raise RuntimeError("EfficientAD must be fitted before predict")
        map_st, map_ae = self._raw_maps(images)
        map_st = F.interpolate(
            map_st, images.shape[-2:], mode="bilinear", align_corners=False
        )
        map_ae = F.interpolate(
            map_ae, images.shape[-2:], mode="bilinear", align_corners=False
        )
        anomaly_map = 0.5 * (
            self._calibrate_map(map_st, self.map_st_q90, self.map_st_q995)
            + self._calibrate_map(map_ae, self.map_ae_q90, self.map_ae_q995)
        )
        raw_map = anomaly_map
        bounded_map = anomaly_map.clamp(0, 1)
        score = bounded_map.flatten(1).amax(1)
        return AnomalyPrediction(score, bounded_map), raw_map.flatten(1).amax(1), raw_map

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        return self.predict_with_raw(images)[0]
