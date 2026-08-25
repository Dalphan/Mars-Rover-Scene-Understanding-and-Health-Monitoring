from __future__ import annotations

import copy
import math
import random
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
from .trainable import _clean_images, _validation_auc


# Ported from the MIT-licensed official implementation:
# https://github.com/ETH-PBL/TinyGLASS


class TinyGLASSFeatureExtractor(nn.Module):
    """TinyGLASS ResNet-18 layer2/layer3 patch-grid embedding."""

    def __init__(
        self, *, pretrained: bool, weights_name: str,
        patch_size: int = 3, output_channels_per_layer: int = 64,
    ) -> None:
        super().__init__()
        backbone = build_explicit_torchvision_backbone(
            "resnet18", pretrained=pretrained, weights_name=weights_name
        )
        self.extractor = create_feature_extractor(
            backbone, return_nodes={"layer2": "layer2", "layer3": "layer3"}
        )
        self.extractor.requires_grad_(False)
        self.patch_size = patch_size
        self.output_channels_per_layer = output_channels_per_layer

    def train(self, mode: bool = True):
        super().train(False)
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
        self.extractor.eval()
        with torch.no_grad():
            features = self.extractor(images)
        layer2 = self._patch_grid(features["layer2"])
        layer3 = self._patch_grid(features["layer3"])
        layer2 = F.adaptive_avg_pool2d(layer2, layer3.shape[-2:])
        return torch.cat((self._reduce(layer2), self._reduce(layer3)), dim=1)


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
    def __init__(
        self, *, input_size: tuple[int, int], blend_mean: float = 0.5,
        blend_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.blend_mean = blend_mean
        self.blend_std = blend_std

    def _one_mask(self, device: torch.device) -> torch.Tensor:
        height, width = self.input_size
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
            if mask.any():
                return mask.unsqueeze(0)
        raise RuntimeError("Could not generate a non-empty TinyGLASS Perlin mask")

    def forward(
        self, images: torch.Tensor, textures: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masks = torch.stack([
            self._one_mask(images.device) for _ in range(images.shape[0])
        ])
        beta = torch.normal(
            self.blend_mean, self.blend_std, size=(images.shape[0], 1, 1, 1),
            device=images.device,
        ).clamp(0.2, 0.8)
        augmented = (
            images * (1 - masks)
            + ((1 - beta) * textures + beta * images) * masks
        )
        return augmented, masks


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
        max_samples_per_epoch: int | None = 392,
        texture_root: str | None = None,
        require_texture_dataset: bool = True,
        blend_mean: float = 0.5, blend_std: float = 0.1,
        gaussian_sigma: float = 4.0,
        fixed_training_duration: bool = True,
        validation_interval: int = 1, validation_batches: int = 64,
    ) -> None:
        super().__init__()
        if require_texture_dataset and not texture_root:
            raise FileNotFoundError("TinyGLASS requires texture_root=dtd/images")
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
        self.max_samples_per_epoch = max_samples_per_epoch
        self.require_texture_dataset = require_texture_dataset
        self.blend_mean = blend_mean
        self.blend_std = blend_std
        self.gaussian_sigma = gaussian_sigma
        self.fixed_training_duration = fixed_training_duration
        self.validation_interval = validation_interval
        self.validation_batches = validation_batches

        self.features = TinyGLASSFeatureExtractor(
            pretrained=pretrained, weights_name=weights_name,
            patch_size=patch_size,
        )
        self.discriminator = TinyGLASSDiscriminator()
        self.textures = DTDTextureSampler(texture_root, self.input_size)
        if require_texture_dataset and not self.textures.paths:
            raise FileNotFoundError(f"No DTD images found under {texture_root}")
        self.las = TinyGLASSLAS(
            input_size=self.input_size,
            blend_mean=blend_mean, blend_std=blend_std,
        )
        self.register_buffer("center", torch.zeros(128))
        self.register_buffer("fitted", torch.tensor(False))
        self.fit_summary: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted)

    def train(self, mode: bool = True):
        super().train(mode)
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
            "max_samples_per_epoch": self.max_samples_per_epoch,
            "require_texture_dataset": self.require_texture_dataset,
            "blend_mean": self.blend_mean, "blend_std": self.blend_std,
            "gaussian_sigma": self.gaussian_sigma,
            "fixed_training_duration": self.fixed_training_duration,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
        }

    @torch.no_grad()
    def _compute_center(self, train_loader, device: torch.device) -> int:
        total = torch.zeros_like(self.center, device=device)
        patches = 0
        for batch in tqdm(
            train_loader, desc="TinyGLASS feature center",
            unit="batch", leave=False,
        ):
            features = self.features(_clean_images(batch, device))
            flattened = features.permute(0, 2, 3, 1).reshape(-1, 128)
            total += flattened.sum(0)
            patches += flattened.shape[0]
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
            true_scores = self.discriminator(true_features)
            gas_scores = self.discriminator(gas)
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
        if resume:
            raise NotImplementedError("TinyGLASS resume is not implemented")
        device = torch.device(device)
        self.to(device)
        center_patches = self._compute_center(train_loader, device)
        optimizer = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.learning_rate * 2,
            weight_decay=self.weight_decay,
        )
        best_auc = -math.inf
        best_state = None
        steps = 0
        skipped_empty_masks = 0
        epoch_progress = tqdm(
            range(1, self.epochs + 1), desc="TinyGLASS training",
            unit="epoch", dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            self.train(True)
            samples = 0
            produced = False
            for batch in train_loader:
                produced = True
                images = _clean_images(batch, device)
                textures = self.textures.sample(images.shape[0], device=device)
                augmented, image_mask = self.las(images, textures)
                with torch.no_grad():
                    true_features = self.features(images)
                    fake_features = self.features(augmented)
                feature_mask = F.interpolate(
                    image_mask, size=true_features.shape[-2:], mode="nearest"
                )
                loss = self._training_loss(
                    true_features.detach(), fake_features.detach(), feature_mask
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
                auc = _validation_auc(
                    self, validation_loader, device, self.validation_batches
                )
                if not math.isnan(auc) and auc > best_auc:
                    best_auc = auc
                    best_state = copy.deepcopy(self.state_dict())
        if best_state is not None:
            self.load_state_dict(best_state)
        self.fitted.fill_(True)
        self.fit_summary = {
            "epochs": self.epochs, "steps": steps,
            "center_patches": center_patches,
            "skipped_empty_masks": skipped_empty_masks,
            "fixed_training_duration": self.fixed_training_duration,
            "selected_validation_image_auroc": (
                None if self.fixed_training_duration or best_state is None else best_auc
            ),
        }
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
