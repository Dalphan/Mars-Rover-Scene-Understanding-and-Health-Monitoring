from __future__ import annotations

import copy
import math
import random
import warnings
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import gaussian_blur, pil_to_tensor
from tqdm import tqdm

from .base import AnomalyDetector, AnomalyPrediction
from .supersimplenet import _rand_perlin_2d, build_explicit_torchvision_backbone
from .trainable import (
    _capture_training_rng_state,
    _clean_images,
    _image_auc,
    _restore_training_rng_state,
    _save_checkpoint_atomic,
)


# Ported from the MIT-licensed official implementation:
# https://github.com/ETH-PBL/TinyGLASS


TINYGLASS_TRAINING_CHECKPOINT_FORMAT = "tinyglass_training_v4_configurable_las"


def _score_distribution(scores: torch.Tensor) -> dict[str, int | float | None]:
    scores = scores.detach().double().cpu().flatten()
    if scores.numel() == 0:
        return {
            "count": 0, "min": None, "max": None,
            "mean": None, "std": None,
            "greater_than_0_99_fraction": None,
            "greater_than_0_999_fraction": None,
        }
    return {
        "count": scores.numel(),
        "min": scores.min().item(),
        "max": scores.max().item(),
        "mean": scores.mean().item(),
        "std": scores.std(unbiased=False).item(),
        "greater_than_0_99_fraction": (scores > 0.99).double().mean().item(),
        "greater_than_0_999_fraction": (scores > 0.999).double().mean().item(),
    }


@torch.no_grad()
def _validation_diagnostics(
    model, loader, device: torch.device, max_batches: int | None,
) -> dict[str, Any]:
    if loader is None:
        raise ValueError(
            "TinyGLASS best-checkpoint selection requires validation_loader"
        )
    model.eval()
    score_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        prediction = model.predict(batch["image"].to(device, non_blocking=True))
        score_chunks.append(prediction.anomaly_score.detach().cpu())
        label_chunks.append(torch.as_tensor(batch["label"]).int().cpu())
    return _score_diagnostics(score_chunks, label_chunks)


def _score_diagnostics(
    score_chunks: list[torch.Tensor], label_chunks: list[torch.Tensor],
) -> dict[str, Any]:
    if not score_chunks:
        raise ValueError("validation_loader produced no images")
    scores = torch.cat(score_chunks).flatten()
    labels = torch.cat(label_chunks).flatten()
    if scores.shape != labels.shape:
        raise ValueError("Validation image scores and labels have different sizes")
    if not torch.isfinite(scores).all():
        raise ValueError("TinyGLASS validation scores must be finite")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("TinyGLASS validation labels must be binary")
    if not torch.any(labels == 0) or not torch.any(labels == 1):
        raise ValueError("TinyGLASS validation requires clean and anomaly samples")
    return {
        "image_auroc": _image_auc(scores.tolist(), labels.tolist()),
        "raw_image_score_distribution": {
            "all": _score_distribution(scores),
            "clean": _score_distribution(scores[labels == 0]),
            "anomaly": _score_distribution(scores[labels == 1]),
        },
    }


@torch.no_grad()
def _build_validation_feature_cache(
    model, loader, device: torch.device, max_batches: int | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if loader is None:
        raise ValueError(
            "TinyGLASS best-checkpoint selection requires validation_loader"
        )
    model.eval()
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        features = model.features(
            batch["image"].to(device, non_blocking=True)
        ).detach().cpu()
        labels = torch.as_tensor(batch["label"]).int().cpu().flatten()
        if features.shape[0] != labels.numel():
            raise ValueError("Validation features and labels have different sizes")
        cache.append((features, labels))
    if not cache:
        raise ValueError("validation_loader produced no images")
    return cache


@torch.no_grad()
def _cached_validation_diagnostics(
    model, cache: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device,
) -> dict[str, Any]:
    model.eval()
    score_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for features, labels in cache:
        patch_scores = model.discriminator(
            features.to(device, non_blocking=True)
        )
        score_chunks.append(patch_scores.flatten(1).amax(1).cpu())
        label_chunks.append(labels)
    return _score_diagnostics(score_chunks, label_chunks)


class TinyGLASSFeatureExtractor(nn.Module):
    """TinyGLASS ResNet-18 layer2/layer3 patch-grid embedding."""

    def __init__(
        self, *, pretrained: bool, weights_name: str,
        patch_size: int = 3, output_channels_per_layer: int = 64,
        freeze_backbone: bool = True,
        feature_grid_resolution: str = "layer3",
    ) -> None:
        super().__init__()
        backbone = build_explicit_torchvision_backbone(
            "resnet18", pretrained=pretrained, weights_name=weights_name
        )
        self.extractor = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        )
        self.freeze_backbone = bool(freeze_backbone)
        self.extractor.requires_grad_(not self.freeze_backbone)
        self.patch_size = patch_size
        self.output_channels_per_layer = output_channels_per_layer
        if feature_grid_resolution not in {"layer2", "layer3"}:
            raise ValueError(
                "feature_grid_resolution must be either 'layer2' or 'layer3'"
            )
        self.feature_grid_resolution = feature_grid_resolution

    def train(self, mode: bool = True):
        super().train(False if self.freeze_backbone else mode)
        return self

    def _patch_grid(self, feature: torch.Tensor) -> torch.Tensor:
        patches = F.unfold(
            feature, kernel_size=self.patch_size, stride=1,
            padding=self.patch_size // 2,
        )
        return patches.reshape(
            feature.shape[0], -1, feature.shape[-2], feature.shape[-1]
        )

    def _reduce(self, feature: torch.Tensor) -> torch.Tensor:
        channels = feature.shape[1]
        if channels % self.output_channels_per_layer:
            raise ValueError(
                f"Patch channels {channels} are not divisible by "
                f"{self.output_channels_per_layer}"
            )
        return feature.reshape(
            feature.shape[0], self.output_channels_per_layer,
            channels // self.output_channels_per_layer,
            *feature.shape[-2:],
        ).mean(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            self.extractor.eval()
            with torch.no_grad():
                features = self.extractor(images)
        else:
            features = self.extractor(images)
        layer2 = self._reduce(self._patch_grid(features["layer2"]))
        layer3 = self._reduce(self._patch_grid(features["layer3"]))
        if self.feature_grid_resolution == "layer2":
            layer3 = F.interpolate(
                layer3, layer2.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            layer2 = F.adaptive_avg_pool2d(layer2, layer3.shape[-2:])
        return torch.cat((layer2, layer3), dim=1)


class TinyGLASSDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 128, hidden_channels: int = 512) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class DTDTextureSampler:
    """Texture loader and random augmentation used by TinyGLASS LAS."""

    _EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root: str | None, input_size: tuple[int, int]) -> None:
        self.root = None if root is None else Path(root)
        self.input_size = input_size
        self.paths = [] if self.root is None else sorted(
            path for path in self.root.rglob("*")
            if path.suffix.lower() in self._EXTENSIONS
        )

    @staticmethod
    def _random_operations(image: Image.Image) -> Image.Image:
        operations = [
            lambda value: TF.adjust_contrast(value, random.uniform(0.8, 1.2)),
            lambda value: TF.adjust_brightness(value, random.uniform(0.8, 1.2)),
            lambda value: TF.adjust_hue(
                TF.adjust_saturation(value, random.uniform(0.8, 1.2)),
                random.uniform(-0.2, 0.2),
            ),
            TF.hflip,
            TF.vflip,
            lambda value: TF.rgb_to_grayscale(value, num_output_channels=3),
            TF.autocontrast,
            TF.equalize,
            lambda value: TF.rotate(
                value, random.uniform(-45, 45),
                interpolation=InterpolationMode.BILINEAR,
            ),
        ]
        for index in random.sample(range(len(operations)), 3):
            image = operations[index](image)
        return image

    def sample(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        if not self.paths:
            raise RuntimeError("TinyGLASS requires DTD textures for LAS")
        values = []
        for index in torch.randint(len(self.paths), (batch_size,)).tolist():
            with Image.open(self.paths[index]) as image:
                image = TF.resize(
                    image.convert("RGB"), list(self.input_size),
                    interpolation=InterpolationMode.BILINEAR, antialias=True,
                )
                image = self._random_operations(image)
                tensor = pil_to_tensor(image).float().div_(255.0)
            mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
            std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
            values.append((tensor - mean) / std)
        return torch.stack(values).to(device, non_blocking=True)


class TinyGLASSLAS(nn.Module):
    _MODES = {"texture", "hole", "mixed"}
    # Full-axis fractions relative to the equivalent wheel diameter. They are
    # derived from the Blender hole contract and its 0.485 m reference wheel.
    _AXIS_FRACTIONS = (
        ((0.027 / 0.485, 0.041 / 0.485), (0.010 / 0.485, 0.017 / 0.485)),
        ((0.044 / 0.485, 0.068 / 0.485), (0.015 / 0.485, 0.027 / 0.485)),
        ((0.073 / 0.485, 0.116 / 0.485), (0.022 / 0.485, 0.041 / 0.485)),
    )

    def __init__(
        self, *, input_size: tuple[int, int], blend_mean: float = 0.5,
        blend_std: float = 0.1, mode: str = "texture",
        hole_probability: float = 0.5,
        hole_luminance_range: tuple[float, float] = (0.18, 0.32),
        hole_severity_weights: tuple[float, float, float] = (0.5, 0.4, 0.1),
    ) -> None:
        super().__init__()
        if mode not in self._MODES:
            raise ValueError(f"LAS mode must be one of {sorted(self._MODES)}")
        if not 0 <= hole_probability <= 1:
            raise ValueError("LAS hole_probability must be in [0, 1]")
        if (
            len(hole_luminance_range) != 2
            or not 0 < hole_luminance_range[0] <= hole_luminance_range[1] < 1
        ):
            raise ValueError("LAS hole_luminance_range must satisfy 0 < min <= max < 1")
        if (
            len(hole_severity_weights) != 3
            or any(not math.isfinite(value) or value < 0 for value in hole_severity_weights)
            or sum(hole_severity_weights) <= 0
        ):
            raise ValueError(
                "LAS hole_severity_weights must contain three finite, "
                "non-negative values with a positive sum"
            )
        self.input_size = input_size
        self.blend_mean = blend_mean
        self.blend_std = blend_std
        self.mode = mode
        self.hole_probability = float(hole_probability)
        self.hole_luminance_range = tuple(float(v) for v in hole_luminance_range)
        severity_total = float(sum(hole_severity_weights))
        self.hole_severity_weights = tuple(
            float(value) / severity_total for value in hole_severity_weights
        )

    @property
    def requires_textures(self) -> bool:
        return self.mode in {"texture", "mixed"}

    @staticmethod
    def downsample_mask(
        image_mask: torch.Tensor, output_size: tuple[int, int],
    ) -> torch.Tensor:
        return F.adaptive_max_pool2d(image_mask, output_size=output_size)

    def _one_mask(
        self, device: torch.device, valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        height, width = self.input_size
        if valid_mask is not None:
            if valid_mask.shape != (1, height, width):
                raise ValueError(
                    "TinyGLASS LAS target mask must have shape "
                    f"(1, {height}, {width})"
                )
            valid_mask = valid_mask.to(device=device).bool().squeeze(0)
            if not valid_mask.any():
                raise ValueError("TinyGLASS LAS target mask cannot be empty")
        for _ in range(32):
            masks = []
            for _ in range(2):
                scale_y = 2 ** int(torch.randint(0, 6, (1,)).item())
                scale_x = 2 ** int(torch.randint(0, 6, (1,)).item())
                noise = _rand_perlin_2d(
                    (height, width), (scale_y, scale_x), device=device
                )
                angle = float(torch.empty(()).uniform_(-90, 90))
                noise = TF.rotate(
                    noise[None], angle,
                    interpolation=InterpolationMode.BILINEAR,
                )[0]
                masks.append((noise > 0.5).float())
            choice = float(torch.rand(()))
            if choice > 2 / 3:
                mask = torch.clamp(masks[0] + masks[1], 0, 1)
            elif choice > 1 / 3:
                mask = masks[0] * masks[1]
            else:
                mask = masks[0]
            if valid_mask is not None:
                mask = mask * valid_mask
            if mask.any():
                return mask.unsqueeze(0)
        raise RuntimeError(
            "Could not generate a non-empty TinyGLASS Perlin mask inside the "
            "target wheel"
        )

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(()).uniform_(low, high))

    def _hole_contour(
        self, yy: torch.Tensor, xx: torch.Tensor, *, center_y: int,
        center_x: int, semi_long: float, semi_short: float,
    ) -> torch.Tensor:
        angle = self._uniform(-math.pi, math.pi)
        cos_angle, sin_angle = math.cos(angle), math.sin(angle)
        dx, dy = xx - center_x, yy - center_y
        major = (dx * cos_angle + dy * sin_angle) / semi_long
        minor = (-dx * sin_angle + dy * cos_angle) / semi_short
        radius = torch.sqrt(major.square() + minor.square())
        theta = torch.atan2(minor, major)
        profile = int(torch.multinomial(
            torch.tensor((0.45, 0.40, 0.15), device=yy.device), 1
        ).item())
        if profile == 0:  # jagged slit
            frequencies, amplitudes = (3, 5, 7), (0.11, 0.07, 0.04)
        elif profile == 1:  # branched tear: one pronounced, smooth side lobe
            frequencies, amplitudes = (2, 4, 6), (0.09, 0.06, 0.03)
        else:  # peeled window
            frequencies, amplitudes = (2, 3), (0.07, 0.04)
        boundary = torch.ones_like(radius)
        for frequency, amplitude in zip(frequencies, amplitudes):
            phase = self._uniform(-math.pi, math.pi)
            boundary = boundary + amplitude * torch.sin(frequency * theta + phase)
        if profile == 1:
            branch_angle = self._uniform(-math.pi, math.pi)
            angular_distance = torch.atan2(
                torch.sin(theta - branch_angle), torch.cos(theta - branch_angle)
            )
            boundary = boundary + 0.18 * torch.exp(
                -angular_distance.square() / (2 * 0.28 ** 2)
            )
        return radius <= boundary.clamp_min(0.65)

    def _one_hole_mask(
        self, device: torch.device, valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        height, width = self.input_size
        if valid_mask.shape != (1, height, width):
            raise ValueError(
                "TinyGLASS hole LAS target mask must have shape "
                f"(1, {height}, {width})"
            )
        valid = valid_mask.to(device=device).bool().squeeze(0)
        locations = valid.nonzero(as_tuple=False)
        if locations.numel() == 0:
            raise ValueError("TinyGLASS hole LAS target mask cannot be empty")
        wheel_diameter = 2.0 * math.sqrt(float(valid.sum()) / math.pi)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # Re-sample size and centre together: perspective and partial visibility
        # can make the larger physical severity ranges impossible to contain.
        for _ in range(96):
            severity = int(torch.multinomial(
                torch.tensor(self.hole_severity_weights, device=device), 1
            ).item())
            long_range, short_range = self._AXIS_FRACTIONS[severity]
            long_axis = max(4.0, wheel_diameter * self._uniform(*long_range))
            short_axis = max(3.0, wheel_diameter * self._uniform(*short_range))
            ratio = long_axis / short_axis
            if not 2.0 <= ratio <= 4.5:
                continue
            center = locations[int(torch.randint(len(locations), (1,), device=device))]
            mask = self._hole_contour(
                yy, xx, center_y=int(center[0]), center_x=int(center[1]),
                semi_long=long_axis / 2, semi_short=short_axis / 2,
            )
            if mask.sum() >= 6 and not (mask & ~valid).any():
                return mask.float().unsqueeze(0)
        raise RuntimeError(
            "Could not place a connected hole-like LAS anomaly fully inside "
            "the target wheel"
        )

    def _hole_composite(
        self, image: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        rgb = (image * std + mean).clamp(0, 1)
        mask_4d = mask.unsqueeze(0)
        interior = 1 - F.max_pool2d(1 - mask_4d, 3, stride=1, padding=1)
        interior = interior.squeeze(0)
        rim = (mask - interior).clamp(0, 1)
        noise = torch.rand((1, *self.input_size), device=image.device)
        noise = F.avg_pool2d(noise.unsqueeze(0), 9, stride=1, padding=4).squeeze(0)
        noise = 0.85 + 0.30 * noise
        luminance = self._uniform(*self.hole_luminance_range)
        cavity = (rgb * luminance * noise).clamp(0, 1)
        rim_factor = self._uniform(0.45, 0.70)
        altered = cavity * interior + (rgb * rim_factor) * rim
        altered_normalized = (altered - mean) / std
        return image * (1 - mask) + altered_normalized * mask

    def forward(
        self, images: torch.Tensor, textures: torch.Tensor | None = None,
        target_masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_masks is not None:
            expected_shape = (images.shape[0], 1, *self.input_size)
            if target_masks.shape != expected_shape:
                raise ValueError(
                    "TinyGLASS LAS target masks must match the image batch: "
                    f"expected {expected_shape}, got {tuple(target_masks.shape)}"
                )
        if self.requires_textures and textures is None:
            raise ValueError(f"TinyGLASS LAS mode '{self.mode}' requires textures")
        augmented_images, masks = [], []
        for index in range(images.shape[0]):
            use_hole = self.mode == "hole" or (
                self.mode == "mixed" and float(torch.rand(())) < self.hole_probability
            )
            valid_mask = None if target_masks is None else target_masks[index]
            if use_hole:
                if valid_mask is None:
                    raise ValueError("TinyGLASS hole LAS requires target masks")
                mask = self._one_hole_mask(images.device, valid_mask)
                augmented = self._hole_composite(images[index], mask)
            else:
                mask = self._one_mask(images.device, valid_mask)
                beta = float(torch.normal(
                    self.blend_mean, self.blend_std, size=(), device=images.device
                ).clamp(0.2, 0.8))
                augmented = images[index] * (1 - mask) + (
                    (1 - beta) * textures[index] + beta * images[index]
                ) * mask
            augmented_images.append(augmented)
            masks.append(mask)
        return torch.stack(augmented_images), torch.stack(masks)


def _tinyglass_focal_loss(
    probabilities: torch.Tensor, targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    probabilities = probabilities.clamp(1e-5, 1 - 1e-5)
    targets = targets.float()
    pt = probabilities * targets + (1 - probabilities) * (1 - targets)
    return (-(1 - pt).pow(gamma) * torch.log(pt)).mean()


class TinyGLASSDeployment(nn.Module):
    def __init__(
        self, features: TinyGLASSFeatureExtractor,
        discriminator: TinyGLASSDiscriminator,
    ) -> None:
        super().__init__()
        self.features = copy.deepcopy(features)
        self.discriminator = copy.deepcopy(discriminator)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.features(images)).unsqueeze(1)


class TinyGLASS(AnomalyDetector):
    """TinyGLASS with dedicated LAS, GAS, patch embedding and discriminator."""

    def __init__(
        self, *, pretrained: bool = True,
        weights_name: str = "IMAGENET1K_V1",
        input_size: tuple[int, int] = (256, 256), patch_size: int = 3,
        epochs: int = 640, learning_rate: float = 1e-4,
        weight_decay: float = 1e-2, noise_std: float = 0.015,
        radius_quantile: float = 0.75, hard_mining_quantile: float = 0.5,
        gas_steps: int = 20, gas_step_size: float = 0.001,
        hypersphere_projection: bool = True,
        feature_grid_resolution: str = "layer3",
        max_samples_per_epoch: int | None = 392,
        texture_root: str | None = None,
        require_texture_dataset: bool = True,
        blend_mean: float = 0.5, blend_std: float = 0.1,
        gaussian_sigma: float = 4.0,
        las_restrict_to_target_mask: bool = False,
        las_mode: str = "texture", las_hole_probability: float = 0.5,
        las_hole_luminance_range: tuple[float, float] = (0.18, 0.32),
        las_hole_severity_weights: tuple[float, float, float] = (0.5, 0.4, 0.1),
        freeze_backbone: bool = True,
        backbone_learning_rate: float = 1e-5,
        fixed_training_duration: bool = False,
        validation_interval: int = 1, validation_batches: int | None = None,
        cache_validation_features: bool = True,
        early_stopping_patience: int | None = 20,
        early_stopping_min_delta: float = 0.0,
        checkpoint_interval: int = 1,
    ) -> None:
        super().__init__()
        if not 0 < radius_quantile < 1:
            raise ValueError("radius_quantile must be in (0, 1)")
        if not 0 <= hard_mining_quantile < 1:
            raise ValueError("hard_mining_quantile must be in [0, 1)")
        self.pretrained = pretrained
        self.weights_name = weights_name
        self.input_size = tuple(input_size)
        self.patch_size = patch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.radius_quantile = radius_quantile
        self.hard_mining_quantile = hard_mining_quantile
        self.gas_steps = gas_steps
        self.gas_step_size = gas_step_size
        self.hypersphere_projection = hypersphere_projection
        if feature_grid_resolution not in {"layer2", "layer3"}:
            raise ValueError(
                "feature_grid_resolution must be either 'layer2' or 'layer3'"
            )
        self.feature_grid_resolution = feature_grid_resolution
        self.max_samples_per_epoch = max_samples_per_epoch
        self.require_texture_dataset = require_texture_dataset
        self.blend_mean = blend_mean
        self.blend_std = blend_std
        self.gaussian_sigma = gaussian_sigma
        self.las_restrict_to_target_mask = bool(las_restrict_to_target_mask)
        self.las_mode = str(las_mode)
        self.las_hole_probability = float(las_hole_probability)
        self.las_hole_luminance_range = tuple(las_hole_luminance_range)
        self.las_hole_severity_weights = tuple(las_hole_severity_weights)
        if self.las_mode in {"hole", "mixed"} and not self.las_restrict_to_target_mask:
            raise ValueError(
                "TinyGLASS hole/mixed LAS requires las_restrict_to_target_mask=true"
            )
        self.freeze_backbone = bool(freeze_backbone)
        if backbone_learning_rate <= 0:
            raise ValueError("backbone_learning_rate must be positive")
        self.backbone_learning_rate = float(backbone_learning_rate)
        self.fixed_training_duration = fixed_training_duration
        if validation_interval < 1:
            raise ValueError("validation_interval must be at least 1")
        self.validation_interval = validation_interval
        if validation_batches is not None and validation_batches < 1:
            raise ValueError("validation_batches must be positive or None")
        self.validation_batches = validation_batches
        self.cache_validation_features = bool(cache_validation_features)
        if not self.freeze_backbone and self.cache_validation_features:
            warnings.warn(
                "TinyGLASS validation feature cache disabled because "
                "freeze_backbone=false",
                UserWarning,
                stacklevel=2,
            )
            self.cache_validation_features = False
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")
        if early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        if checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")
        self.checkpoint_interval = checkpoint_interval

        self.features = TinyGLASSFeatureExtractor(
            pretrained=pretrained, weights_name=weights_name,
            patch_size=patch_size, freeze_backbone=self.freeze_backbone,
            feature_grid_resolution=self.feature_grid_resolution,
        )
        self.discriminator = TinyGLASSDiscriminator()
        self.textures = DTDTextureSampler(texture_root, self.input_size)
        self.las = TinyGLASSLAS(
            input_size=self.input_size,
            blend_mean=blend_mean, blend_std=blend_std,
            mode=self.las_mode,
            hole_probability=self.las_hole_probability,
            hole_luminance_range=self.las_hole_luminance_range,
            hole_severity_weights=self.las_hole_severity_weights,
        )
        self.las_hole_severity_weights = self.las.hole_severity_weights
        self.register_buffer("center", torch.zeros(128))
        self.register_buffer("fitted", torch.tensor(False))
        self.fit_summary: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
        return self

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "backbone": "resnet18", "pretrained": self.pretrained,
            "weights_name": self.weights_name, "input_size": list(self.input_size),
            "patch_size": self.patch_size, "target_embedding_dimension": 128,
            "epochs": self.epochs, "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay, "noise_std": self.noise_std,
            "radius_quantile": self.radius_quantile,
            "hard_mining_quantile": self.hard_mining_quantile,
            "gas_steps": self.gas_steps, "gas_step_size": self.gas_step_size,
            "hypersphere_projection": self.hypersphere_projection,
            "feature_grid_resolution": self.feature_grid_resolution,
            "max_samples_per_epoch": self.max_samples_per_epoch,
            "require_texture_dataset": self.require_texture_dataset,
            "blend_mean": self.blend_mean, "blend_std": self.blend_std,
            "gaussian_sigma": self.gaussian_sigma,
            "las_restrict_to_target_mask": self.las_restrict_to_target_mask,
            "las_mode": self.las_mode,
            "las_hole_probability": self.las_hole_probability,
            "las_hole_luminance_range": list(self.las_hole_luminance_range),
            "las_hole_severity_weights": list(self.las_hole_severity_weights),
            "training_algorithm": TINYGLASS_TRAINING_CHECKPOINT_FORMAT,
            "freeze_backbone": self.freeze_backbone,
            "backbone_learning_rate": self.backbone_learning_rate,
            "fixed_training_duration": self.fixed_training_duration,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
            "cache_validation_features": self.cache_validation_features,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }

    def _training_state(self) -> dict[str, Any]:
        return {
            "features": self.features.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "center": self.center.detach().clone(),
        }

    def _load_training_state(self, state: dict[str, Any]) -> None:
        self.features.load_state_dict(state["features"])
        self.discriminator.load_state_dict(state["discriminator"])
        self.center.copy_(state["center"].to(self.center.device))

    @torch.no_grad()
    def _compute_center(self, train_loader, device: torch.device) -> int:
        total = torch.zeros_like(self.center, device=device)
        patches = 0
        features_were_training = self.features.training
        self.features.eval()
        try:
            for batch in tqdm(
                train_loader, desc="TinyGLASS feature center",
                unit="batch", leave=False,
            ):
                features = self.features(_clean_images(batch, device))
                flattened = features.permute(0, 2, 3, 1).reshape(-1, 128)
                total += flattened.sum(0)
                patches += flattened.shape[0]
        finally:
            self.features.train(features_were_training)
        if patches == 0:
            raise ValueError("train_loader produced no images")
        self.center.copy_(total / patches)
        return patches

    def _project_gas(
        self, gas: torch.Tensor, true: torch.Tensor,
        center: torch.Tensor, radius: torch.Tensor,
    ) -> torch.Tensor:
        gas_flat = gas.permute(0, 2, 3, 1).reshape(-1, 128)
        true_flat = true.permute(0, 2, 3, 1).reshape(-1, 128)
        base = center if self.hypersphere_projection else true_flat
        lower = radius if self.hypersphere_projection else gas_flat.new_tensor(0.5)
        vector = gas_flat - base
        norm = torch.norm(vector, dim=1).clamp_min(1e-10)
        alpha = torch.clamp(norm, lower, 2 * lower)
        projected = base + vector * (alpha / norm).unsqueeze(1)
        return projected.reshape_as(gas.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def _training_loss(
        self, true_features: torch.Tensor, fake_features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        batch, channels, height, width = true_features.shape
        true_flat = true_features.permute(0, 2, 3, 1).reshape(-1, channels)
        fake_flat = fake_features.permute(0, 2, 3, 1).reshape(-1, channels)
        mask_flat = feature_mask.reshape(-1).bool()
        center = self.center.reshape(1, -1).expand_as(true_flat)
        true_points = torch.cat((fake_flat[~mask_flat], true_flat), dim=0)
        true_centers = torch.cat((center[~mask_flat], center), dim=0)
        radius = torch.quantile(
            torch.norm(true_points - true_centers, dim=1), self.radius_quantile
        ).detach().clamp_min(1e-6)

        gas = (
            true_features + torch.normal(
                0, self.noise_std, size=true_features.shape,
                device=true_features.device,
            )
        ).detach().requires_grad_(True)
        bce_loss = None
        for gas_step in range(self.gas_steps + 1):
            joint_scores = self.discriminator(torch.cat((true_features, gas), dim=0))
            true_scores, gas_scores = joint_scores.split(batch, dim=0)
            bce_loss = (
                F.binary_cross_entropy(true_scores, torch.zeros_like(true_scores))
                + F.binary_cross_entropy(gas_scores, torch.ones_like(gas_scores))
            )
            if gas_step == self.gas_steps:
                break
            gradient = torch.autograd.grad(
                F.binary_cross_entropy(gas_scores, torch.ones_like(gas_scores)),
                gas,
            )[0]
            gradient = gradient / torch.norm(
                gradient, dim=1, keepdim=True
            ).clamp_min(1e-10)
            gas = (gas + self.gas_step_size * gradient).detach()
            if (gas_step + 1) % 5 == 0:
                gas = self._project_gas(gas, true_features, center, radius)
            gas.requires_grad_(True)

        if not mask_flat.any():
            return None
        if self.hypersphere_projection:
            anomalous = fake_flat[mask_flat]
            anomaly_center = center[mask_flat]
            vector = anomalous - anomaly_center
            norm = torch.norm(vector, dim=1).clamp_min(1e-10)
            alpha = torch.clamp(norm, 2 * radius, 4 * radius)
            fake_flat = fake_flat.clone()
            fake_flat[mask_flat] = anomaly_center + vector * (alpha / norm).unsqueeze(1)
            fake_features = fake_flat.reshape(
                batch, height, width, channels
            ).permute(0, 3, 1, 2)

        fake_scores = self.discriminator(fake_features)
        distance = (fake_scores - feature_mask.squeeze(1)).square()
        if self.hard_mining_quantile > 0:
            threshold = torch.quantile(distance.detach(), self.hard_mining_quantile)
            selected = distance >= threshold
            fake_scores = fake_scores[selected]
            target = feature_mask.squeeze(1)[selected]
        else:
            target = feature_mask.squeeze(1)
        return bce_loss + _tinyglass_focal_loss(fake_scores, target)

    def fit(
        self, train_loader, *, device, validation_loader=None,
        work_dir=None, resume=False,
    ) -> TinyGLASS:
        device = torch.device(device)
        if not self.fixed_training_duration and validation_loader is None:
            raise ValueError(
                "TinyGLASS best-checkpoint selection requires validation_loader"
            )
        self.to(device)
        if self.freeze_backbone:
            optimizer = torch.optim.AdamW(
                self.discriminator.parameters(), lr=self.learning_rate * 2,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": self.discriminator.parameters(),
                        "lr": self.learning_rate * 2,
                    },
                    {
                        "params": self.features.parameters(),
                        "lr": self.backbone_learning_rate,
                    },
                ],
                weight_decay=self.weight_decay,
            )
        best_auc = -math.inf
        best_state = None
        best_epoch = None
        validation_history: list[dict[str, Any]] = []
        no_improvement_validations = 0
        stopped_early = False
        steps = 0
        skipped_empty_masks = 0
        center_patches = 0
        center_patches_total = 0
        center_recomputations = 0
        start_epoch = 0
        completed_epoch = 0
        validation_cache_images = 0
        validation_cache_bytes = 0
        validation_feature_cache = None
        checkpoint_path = (
            None if work_dir is None else Path(work_dir) / "tinyglass_training.ckpt"
        )
        if resume:
            if checkpoint_path is None or not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "resume=true requires work_dir/tinyglass_training.ckpt"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if checkpoint.get("format") != TINYGLASS_TRAINING_CHECKPOINT_FORMAT:
                raise ValueError("Unsupported TinyGLASS training checkpoint")
            saved_config = checkpoint.get("model_config")
            if (
                not isinstance(saved_config, dict)
                or saved_config.get("freeze_backbone") != self.freeze_backbone
            ):
                raise ValueError(
                    "TinyGLASS freeze_backbone mismatch; start a new run with "
                    "resume=false"
                )
            if saved_config != self.checkpoint_config():
                raise ValueError("TinyGLASS training checkpoint config mismatch")
            self._load_training_state(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"])
            completed_epoch = start_epoch
            if not 0 <= start_epoch <= self.epochs:
                raise ValueError("Invalid epoch in TinyGLASS training checkpoint")
            steps = int(checkpoint.get("steps", 0))
            skipped_empty_masks = int(checkpoint.get("skipped_empty_masks", 0))
            center_patches = int(checkpoint.get("center_patches", 0))
            center_patches_total = int(
                checkpoint.get("center_patches_total", center_patches)
            )
            center_recomputations = int(
                checkpoint.get("center_recomputations", int(center_patches > 0))
            )
            best_auc = float(checkpoint.get("best_auc", -math.inf))
            best_state = checkpoint.get("best_state")
            best_epoch = checkpoint.get("best_epoch")
            validation_history = list(checkpoint.get("validation_history", []))
            no_improvement_validations = int(
                checkpoint.get("no_improvement_validations", 0)
            )
            stopped_early = bool(checkpoint.get("stopped_early", False))
            validation_cache_images = int(
                checkpoint.get("validation_cache_images", 0)
            )
            validation_cache_bytes = int(
                checkpoint.get("validation_cache_bytes", 0)
            )
            _restore_training_rng_state(checkpoint, train_loader)
        else:
            if self.las.requires_textures and not self.textures.paths:
                raise FileNotFoundError(
                    "TinyGLASS training requires texture_root=dtd/images"
                )
            if self.freeze_backbone:
                center_patches = self._compute_center(train_loader, device)
                center_patches_total += center_patches
                center_recomputations += 1
        if (
            self.las.requires_textures
            and not stopped_early
            and start_epoch < self.epochs
            and not self.textures.paths
        ):
            raise FileNotFoundError(
                "TinyGLASS resume requires the original DTD texture dataset"
            )
        if (
            not stopped_early
            and start_epoch < self.epochs
            and not self.fixed_training_duration
            and self.cache_validation_features
        ):
            validation_feature_cache = _build_validation_feature_cache(
                self, validation_loader, device, self.validation_batches
            )
            validation_cache_images = sum(
                labels.numel() for _, labels in validation_feature_cache
            )
            validation_cache_bytes = sum(
                features.numel() * features.element_size()
                for features, _ in validation_feature_cache
            )
        remaining_epochs = (
            () if stopped_early else range(start_epoch + 1, self.epochs + 1)
        )
        epoch_progress = tqdm(
            remaining_epochs,
            initial=start_epoch, total=self.epochs, desc="TinyGLASS training",
            unit="epoch", dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            completed_epoch = epoch
            if not self.freeze_backbone:
                center_patches = self._compute_center(train_loader, device)
                center_patches_total += center_patches
                center_recomputations += 1
            self.train(True)
            samples = 0
            produced = False
            for batch in train_loader:
                produced = True
                images = _clean_images(batch, device)
                textures = (
                    self.textures.sample(images.shape[0], device=device)
                    if self.las.requires_textures else None
                )
                target_masks = None
                if self.las_restrict_to_target_mask:
                    if "target_mask" not in batch:
                        raise KeyError(
                            "las_restrict_to_target_mask=true requires target_mask "
                            "in every training batch"
                        )
                    target_masks = batch["target_mask"].to(
                        device, non_blocking=True
                    )
                augmented, image_mask = self.las(
                    images, textures, target_masks=target_masks
                )
                if self.freeze_backbone:
                    with torch.no_grad():
                        true_features = self.features(images)
                        fake_features = self.features(augmented)
                else:
                    true_features = self.features(images)
                    fake_features = self.features(augmented)
                feature_mask = self.las.downsample_mask(
                    image_mask, output_size=true_features.shape[-2:]
                )
                loss = self._training_loss(
                    true_features.detach() if self.freeze_backbone else true_features,
                    fake_features.detach() if self.freeze_backbone else fake_features,
                    feature_mask,
                )
                if loss is None:
                    skipped_empty_masks += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_progress.set_postfix(
                    loss=f"{loss.detach().item():.4f}", refresh=False
                )
                steps += 1
                samples += images.shape[0]
                if self.max_samples_per_epoch and samples >= self.max_samples_per_epoch:
                    break
            if not produced:
                raise ValueError("train_loader produced no images")
            if (
                not self.fixed_training_duration
                and (epoch % self.validation_interval == 0 or epoch == self.epochs)
            ):
                self.fitted.fill_(True)
                diagnostics = (
                    _cached_validation_diagnostics(
                        self, validation_feature_cache, device
                    )
                    if validation_feature_cache is not None
                    else _validation_diagnostics(
                        self, validation_loader, device, self.validation_batches
                    )
                )
                diagnostics = {"epoch": epoch, **diagnostics}
                validation_history.append(diagnostics)
                auc = diagnostics["image_auroc"]
                saturation = diagnostics["raw_image_score_distribution"]["all"]
                epoch_progress.set_postfix(
                    val_auc=f"{auc:.4f}",
                    score_gt_0999=(
                        f"{saturation['greater_than_0_999_fraction']:.3f}"
                    ),
                    refresh=False,
                )
                if auc > best_auc + self.early_stopping_min_delta:
                    best_auc = auc
                    best_epoch = epoch
                    best_state = copy.deepcopy(self._training_state())
                    no_improvement_validations = 0
                else:
                    no_improvement_validations += 1
                if (
                    self.early_stopping_patience is not None
                    and no_improvement_validations >= self.early_stopping_patience
                ):
                    stopped_early = True
            if (
                checkpoint_path is not None
                and (
                    epoch % self.checkpoint_interval == 0
                    or epoch == self.epochs
                    or stopped_early
                )
            ):
                _save_checkpoint_atomic({
                    "format": TINYGLASS_TRAINING_CHECKPOINT_FORMAT,
                    "model_config": self.checkpoint_config(),
                    "model": self._training_state(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "steps": steps,
                    "skipped_empty_masks": skipped_empty_masks,
                    "center_patches": center_patches,
                    "center_patches_total": center_patches_total,
                    "center_recomputations": center_recomputations,
                    "best_auc": best_auc,
                    "best_state": best_state,
                    "best_epoch": best_epoch,
                    "validation_history": validation_history,
                    "no_improvement_validations": no_improvement_validations,
                    "stopped_early": stopped_early,
                    "validation_cache_images": validation_cache_images,
                    "validation_cache_bytes": validation_cache_bytes,
                    **_capture_training_rng_state(train_loader),
                }, checkpoint_path)
            if stopped_early:
                break
        if best_state is not None:
            self._load_training_state(best_state)
        self.fitted.fill_(True)
        self.eval()
        self.fit_summary = {
            "epochs": self.epochs, "steps": steps,
            "epochs_completed": completed_epoch,
            "center_patches": center_patches,
            "center_patches_total": center_patches_total,
            "center_recomputations": center_recomputations,
            "skipped_empty_masks": skipped_empty_masks,
            "las_restrict_to_target_mask": self.las_restrict_to_target_mask,
            "las_mode": self.las_mode,
            "las_requires_textures": self.las.requires_textures,
            "las_hole_severity_weights": list(self.las_hole_severity_weights),
            "fixed_training_duration": self.fixed_training_duration,
            "freeze_backbone": self.freeze_backbone,
            "backbone_learning_rate": self.backbone_learning_rate,
            "selected_validation_image_auroc": (
                None if best_state is None else best_auc
            ),
            "selected_epoch": best_epoch,
            "stopped_early": stopped_early,
            "no_improvement_validations": no_improvement_validations,
            "validation_feature_cache_enabled": (
                not self.fixed_training_duration and self.cache_validation_features
            ),
            "validation_cache_images": validation_cache_images,
            "validation_cache_bytes": validation_cache_bytes,
            "validation_history": validation_history,
        }
        if checkpoint_path is not None:
            _save_checkpoint_atomic({
                "format": TINYGLASS_TRAINING_CHECKPOINT_FORMAT,
                "model_config": self.checkpoint_config(),
                "model": self._training_state(),
                "optimizer": optimizer.state_dict(),
                "epoch": completed_epoch,
                "steps": steps,
                "skipped_empty_masks": skipped_empty_masks,
                "center_patches": center_patches,
                "center_patches_total": center_patches_total,
                "center_recomputations": center_recomputations,
                "best_auc": best_auc,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "validation_history": validation_history,
                "no_improvement_validations": no_improvement_validations,
                "stopped_early": stopped_early,
                "validation_cache_images": validation_cache_images,
                "validation_cache_bytes": validation_cache_bytes,
                **_capture_training_rng_state(train_loader),
            }, checkpoint_path)
        return self

    @torch.no_grad()
    def _patch_scores(self, images: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.features(images)).unsqueeze(1)

    @torch.no_grad()
    def predict_with_raw(self, images: torch.Tensor):
        if not self.is_fitted:
            raise RuntimeError("TinyGLASS must be fitted before predict")
        self.eval()
        patch_scores = self._patch_scores(images)
        anomaly_map = F.interpolate(
            patch_scores, images.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.gaussian_sigma > 0:
            kernel = 2 * math.ceil(3 * self.gaussian_sigma) + 1
            anomaly_map = gaussian_blur(
                anomaly_map, [kernel, kernel],
                [self.gaussian_sigma, self.gaussian_sigma],
            )
        anomaly_score = patch_scores.flatten(1).amax(1)
        prediction = AnomalyPrediction(
            anomaly_score.clamp(0, 1), anomaly_map.clamp(0, 1)
        )
        return prediction, anomaly_score, patch_scores

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        return self.predict_with_raw(images)[0]

    def to_deployment_module(self) -> nn.Module:
        if not self.is_fitted:
            raise RuntimeError("TinyGLASS must be fitted before export")
        return TinyGLASSDeployment(self.features, self.discriminator).eval()
