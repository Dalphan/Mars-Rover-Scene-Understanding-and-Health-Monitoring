from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


RGBTriplet = tuple[float, float, float]
ImageSize = tuple[int, int]
SigmaRange = tuple[float, float]
IMAGENET_MEAN: RGBTriplet = (0.485, 0.456, 0.406)
IMAGENET_STD: RGBTriplet = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Optional preprocessing and train-time image augmentation settings.

    ``resize`` follows the torchvision convention ``(height, width)``. Every
    scalar augmentation value is disabled when ``None`` and otherwise denotes
    its maximum intensity. Photometric transforms affect RGB only.
    """

    resize: ImageSize | None = None
    resize_shorter_side: int | None = None
    center_crop: ImageSize | None = None
    normalize_mean: RGBTriplet | None = None
    normalize_std: RGBTriplet | None = None
    augmentations_enabled: bool = True
    brightness: float | None = None
    contrast: float | None = None
    gamma: float | None = None
    saturation: float | None = None
    sensor_noise: float | None = None
    gaussian_noise: float | None = None
    gaussian_blur: SigmaRange | None = None

    def __post_init__(self) -> None:
        if self.resize is not None and self.resize_shorter_side is not None:
            raise ValueError("resize and resize_shorter_side are mutually exclusive")
        if self.resize is not None:
            if len(self.resize) != 2 or any(value < 1 for value in self.resize):
                raise ValueError("resize must be a positive (height, width) pair")
        if self.resize_shorter_side is not None and self.resize_shorter_side < 1:
            raise ValueError("resize_shorter_side must be positive")
        if self.center_crop is not None:
            if len(self.center_crop) != 2 or any(value < 1 for value in self.center_crop):
                raise ValueError("center_crop must be a positive (height, width) pair")

        if (self.normalize_mean is None) != (self.normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must be set together")
        if self.normalize_mean is not None:
            if len(self.normalize_mean) != 3 or len(self.normalize_std or ()) != 3:
                raise ValueError("normalization mean and std must contain three RGB values")
            if any(value <= 0 for value in self.normalize_std or ()):
                raise ValueError("normalization std values must be positive")

        for name in (
            "brightness",
            "contrast",
            "gamma",
            "saturation",
            "sensor_noise",
            "gaussian_noise",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")

        for name in ("brightness", "contrast", "gamma", "saturation"):
            value = getattr(self, name)
            if value is not None and value >= 1:
                raise ValueError(f"{name} must be smaller than 1")

        if self.gaussian_blur is not None:
            if len(self.gaussian_blur) != 2:
                raise ValueError("gaussian_blur must be a (min_sigma, max_sigma) pair")
            minimum, maximum = self.gaussian_blur
            if minimum <= 0 or maximum < minimum:
                raise ValueError("gaussian_blur requires 0 < min_sigma <= max_sigma")


class WheelPreprocessor:
    """Apply aligned geometric transforms and RGB-only photometric transforms."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    @staticmethod
    def _symmetric_factor(maximum_delta: float) -> float:
        return 1.0 + (2.0 * torch.rand(1).item() - 1.0) * maximum_delta

    @staticmethod
    def _odd_kernel_size(maximum_sigma: float) -> int:
        radius = max(1, round(3.0 * maximum_sigma))
        return 2 * radius + 1

    def __call__(
        self,
        image: torch.Tensor,
        target_mask: torch.Tensor,
        anomaly_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        config = self.config

        if config.resize is not None:
            image = TF.resize(
                image,
                config.resize,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            target_mask = TF.resize(
                target_mask,
                config.resize,
                interpolation=InterpolationMode.NEAREST,
            )
            anomaly_mask = TF.resize(
                anomaly_mask,
                config.resize,
                interpolation=InterpolationMode.NEAREST,
            )
        elif config.resize_shorter_side is not None:
            image = TF.resize(
                image,
                config.resize_shorter_side,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            target_mask = TF.resize(
                target_mask,
                config.resize_shorter_side,
                interpolation=InterpolationMode.NEAREST,
            )
            anomaly_mask = TF.resize(
                anomaly_mask,
                config.resize_shorter_side,
                interpolation=InterpolationMode.NEAREST,
            )

        if config.center_crop is not None:
            image = TF.center_crop(image, config.center_crop)
            target_mask = TF.center_crop(target_mask, config.center_crop)
            anomaly_mask = TF.center_crop(anomaly_mask, config.center_crop)

        # Every model-facing split uses one stable numerical contract.  Keeping
        # uint8 for evaluation while train-time augmentation returned float32
        # would either fail in the backbone or change the input scale by 255x.
        image = TF.convert_image_dtype(image, torch.float32)

        if config.augmentations_enabled:
            if config.brightness is not None:
                image = TF.adjust_brightness(image, self._symmetric_factor(config.brightness))
            if config.contrast is not None:
                image = TF.adjust_contrast(image, self._symmetric_factor(config.contrast))
            if config.gamma is not None:
                image = TF.adjust_gamma(image, self._symmetric_factor(config.gamma))
            if config.saturation is not None:
                image = TF.adjust_saturation(image, self._symmetric_factor(config.saturation))

            if config.gaussian_blur is not None:
                minimum, maximum = config.gaussian_blur
                sigma = minimum + torch.rand(1).item() * (maximum - minimum)
                kernel_size = self._odd_kernel_size(maximum)
                image = TF.gaussian_blur(image, kernel_size, sigma)

            if config.sensor_noise is not None and config.sensor_noise > 0:
                signal_scale = image.clamp(0.0, 1.0).sqrt()
                image = image + torch.randn_like(image) * signal_scale * config.sensor_noise
            if config.gaussian_noise is not None and config.gaussian_noise > 0:
                image = image + torch.randn_like(image) * config.gaussian_noise

        image = image.clamp(0.0, 1.0)
        if config.normalize_mean is not None:
            image = TF.normalize(
                image,
                mean=config.normalize_mean,
                std=config.normalize_std,
            )

        return image, target_mask, anomaly_mask

    def image_for_display(self, image: torch.Tensor) -> torch.Tensor:
        """Return a float RGB image in [0, 1], undoing normalization if needed."""
        if image.dtype == torch.uint8:
            return TF.convert_image_dtype(image, torch.float32)

        result = image.detach().clone()
        if self.config.normalize_mean is not None:
            mean = result.new_tensor(self.config.normalize_mean).view(3, 1, 1)
            std = result.new_tensor(self.config.normalize_std).view(3, 1, 1)
            result = result * std + mean
        return result.clamp(0.0, 1.0)
