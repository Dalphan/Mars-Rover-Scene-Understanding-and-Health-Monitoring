from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from .base import AnomalyDetector, AnomalyPrediction
from .trainable import _clean_images, _validation_auc


# Ported from the MIT-licensed official implementation:
# https://github.com/blaz-r/SuperSimpleNet

def build_explicit_torchvision_backbone(
    backbone: str, *, pretrained: bool, weights_name: str,
) -> nn.Module:
    weights = None
    if pretrained:
        weights_enum = get_model_weights(backbone)
        try:
            weights = getattr(weights_enum, weights_name)
        except AttributeError as error:
            raise ValueError(
                f"Unknown torchvision weights {weights_name!r} for {backbone!r}"
            ) from error
    return get_model(backbone, weights=weights)


def _init_supersimplenet_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(module.weight, 1)


def _rand_perlin_2d(
    shape: tuple[int, int], res: tuple[int, int], *, device: torch.device,
) -> torch.Tensor:
    """Torch port of the Perlin generator used by official SuperSimpleNet."""
    delta = (res[0] / shape[0], res[1] / shape[1])
    repeats = (shape[0] // res[0], shape[1] // res[1])
    y = torch.arange(0, res[0], delta[0], device=device)
    x = torch.arange(0, res[1], delta[1], device=device)
    grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tiled(y_slice, x_slice):
        return (
            gradients[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
            .repeat_interleave(repeats[0], 0)
            .repeat_interleave(repeats[1], 1)
        )

    def dot(gradient, shift):
        offsets = torch.stack(
            (grid[:shape[0], :shape[1], 0] + shift[0],
             grid[:shape[0], :shape[1], 1] + shift[1]),
            dim=-1,
        )
        return (offsets * gradient[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tiled((0, -1), (0, -1)), (0, 0))
    n10 = dot(tiled((1, None), (0, -1)), (-1, 0))
    n01 = dot(tiled((0, -1), (1, None)), (0, -1))
    n11 = dot(tiled((1, None), (1, None)), (-1, -1))
    fade = lambda value: 6 * value**5 - 15 * value**4 + 10 * value**3
    blend = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(
        torch.lerp(n00, n10, blend[..., 0]),
        torch.lerp(n01, n11, blend[..., 0]),
        blend[..., 1],
    )


def _ssn_focal_loss(
    probabilities: torch.Tensor, targets: torch.Tensor, gamma: float = 4.0,
    reduction: str | None = "mean",
) -> torch.Tensor:
    probabilities = probabilities.float()
    targets = targets.float()
    ce = F.binary_cross_entropy(probabilities, targets, reduction="none")
    pt = probabilities * targets + (1 - probabilities) * (1 - targets)
    loss = ce * (1 - pt).pow(gamma)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


class SuperSimpleFeatureExtractor(nn.Module):
    def __init__(
        self, *, backbone: str, pretrained: bool, weights_name: str,
        layers: tuple[str, ...], patch_size: int, input_size: tuple[int, int],
    ) -> None:
        super().__init__()
        model = build_explicit_torchvision_backbone(
            backbone, pretrained=pretrained, weights_name=weights_name
        )
        self.extractor = create_feature_extractor(
            model, return_nodes={layer: layer for layer in layers}
        )
        self.extractor.requires_grad_(False)
        self.layers = layers
        self.pooler = nn.AvgPool2d(patch_size, stride=1, padding=patch_size // 2)
        with torch.no_grad():
            outputs = self.extractor(torch.zeros(1, 3, *input_size))
        self.feature_channels = sum(value.shape[1] for value in outputs.values())

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.extractor.eval()
        with torch.no_grad():
            features = list(self.extractor(images).values())
        height, width = features[0].shape[-2:]
        features = [
            F.interpolate(
                feature, size=(height * 2, width * 2),
                mode="bilinear", align_corners=True,
            )
            for feature in features
        ]
        return self.pooler(torch.cat(features, dim=1))


class SuperSimpleFeatureAdaptor(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, 1)
        self.apply(_init_supersimplenet_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class SuperSimpleDiscriminator(nn.Module):
    def __init__(self, channels: int, *, hidden_channels: int, stop_grad: bool) -> None:
        super().__init__()
        self.stop_grad = stop_grad
        self.segmentor = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=False),
        )
        self.decision_features = nn.Sequential(
            nn.Conv2d(channels + 1, 128, 5, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Linear(128 * 2 + 2, 1)
        self.apply(_init_supersimplenet_weights)

    def parameter_groups(self):
        return (
            self.segmentor.parameters(),
            [*self.decision_features.parameters(), *self.score.parameters()],
        )

    def forward(
        self, segmentation_features: torch.Tensor,
        classification_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anomaly_map = self.segmentor(segmentation_features)
        decision_map = anomaly_map.detach() if self.stop_grad else anomaly_map
        decision = self.decision_features(
            torch.cat((classification_features, decision_map), dim=1)
        )
        map_for_pooling = decision_map if self.stop_grad else anomaly_map
        pooled = torch.cat((
            F.adaptive_max_pool2d(decision, 1),
            F.adaptive_avg_pool2d(decision, 1),
            F.adaptive_max_pool2d(map_for_pooling, 1),
            F.adaptive_avg_pool2d(map_for_pooling, 1),
        ), dim=1).flatten(1)
        return anomaly_map, self.score(pooled).flatten()


class SuperSimpleAnomalyGenerator(nn.Module):
    def __init__(
        self, *, noise_std: float, threshold: float,
        perlin_range: tuple[int, int] = (0, 6),
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.threshold = threshold
        self.perlin_range = perlin_range

    def _masks(
        self, batch_size: int, height: int, width: int, device: torch.device,
    ) -> torch.Tensor:
        power_height = 1 << (height - 1).bit_length()
        power_width = 1 << (width - 1).bit_length()
        masks = []
        for _ in range(batch_size):
            scale_y = 2 ** int(torch.randint(*self.perlin_range, (1,)).item())
            scale_x = 2 ** int(torch.randint(*self.perlin_range, (1,)).item())
            noise = _rand_perlin_2d(
                (power_height, power_width), (scale_y, scale_x), device=device
            )
            noise = F.interpolate(
                noise[None, None], size=(height, width),
                mode="bilinear", align_corners=False,
            )[0]
            mask = (noise > self.threshold).float()
            if torch.rand(()) > 0.5:
                mask.zero_()  # official no_anomaly="empty"
            masks.append(mask)
        return torch.stack(masks)

    def forward(
        self, features: torch.Tensor, adapted: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, height, width = adapted.shape
        features = torch.cat((features, features), dim=0)
        adapted = torch.cat((adapted, adapted), dim=0)
        mask = self._masks(batch * 2, height, width, adapted.device)
        noise = torch.normal(0, self.noise_std, size=adapted.shape, device=adapted.device)
        return features + noise * mask, adapted + noise * mask, mask


class SuperSimpleNet(AnomalyDetector):
    """JIMS SuperSimpleNet in the official unsupervised MVTec configuration."""

    def __init__(
        self, *, backbone: str = "wide_resnet50_2",
        pretrained: bool = True, weights_name: str = "IMAGENET1K_V1",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        input_size: tuple[int, int] = (256, 256), patch_size: int = 3,
        epochs: int = 300, noise_std: float = 0.015,
        perlin_threshold: float = 0.2, adaptor_learning_rate: float = 1e-4,
        segmentation_learning_rate: float = 2e-4,
        decision_learning_rate: float = 2e-4, scheduler_gamma: float = 0.4,
        stop_grad: bool = True, adapt_classification_features: bool = False,
        gradient_clip: bool = False, margin: float = 0.5,
        gaussian_sigma: float = 4.0, fixed_training_duration: bool = True,
        validation_interval: int = 4, validation_batches: int = 64,
        max_samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.weights_name = weights_name
        self.layers = tuple(layers)
        self.input_size = tuple(input_size)
        self.patch_size = patch_size
        self.epochs = epochs
        self.noise_std = noise_std
        self.perlin_threshold = perlin_threshold
        self.adaptor_learning_rate = adaptor_learning_rate
        self.segmentation_learning_rate = segmentation_learning_rate
        self.decision_learning_rate = decision_learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.stop_grad = stop_grad
        self.adapt_classification_features = adapt_classification_features
        self.gradient_clip = gradient_clip
        self.margin = margin
        self.gaussian_sigma = gaussian_sigma
        self.fixed_training_duration = fixed_training_duration
        self.validation_interval = validation_interval
        self.validation_batches = validation_batches
        self.max_samples_per_epoch = max_samples_per_epoch

        self.features = SuperSimpleFeatureExtractor(
            backbone=backbone, pretrained=pretrained, weights_name=weights_name,
            layers=self.layers, patch_size=patch_size, input_size=self.input_size,
        )
        channels = self.features.feature_channels
        self.adaptor = SuperSimpleFeatureAdaptor(channels)
        self.discriminator = SuperSimpleDiscriminator(
            channels, hidden_channels=1024, stop_grad=stop_grad
        )
        self.anomaly_generator = SuperSimpleAnomalyGenerator(
            noise_std=noise_std, threshold=perlin_threshold
        )
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
            "backbone": self.backbone_name, "pretrained": self.pretrained,
            "weights_name": self.weights_name, "layers": list(self.layers),
            "input_size": list(self.input_size), "patch_size": self.patch_size,
            "epochs": self.epochs, "noise_std": self.noise_std,
            "perlin_threshold": self.perlin_threshold,
            "adaptor_learning_rate": self.adaptor_learning_rate,
            "segmentation_learning_rate": self.segmentation_learning_rate,
            "decision_learning_rate": self.decision_learning_rate,
            "scheduler_gamma": self.scheduler_gamma, "stop_grad": self.stop_grad,
            "adapt_classification_features": self.adapt_classification_features,
            "gradient_clip": self.gradient_clip, "margin": self.margin,
            "gaussian_sigma": self.gaussian_sigma,
            "fixed_training_duration": self.fixed_training_duration,
            "validation_interval": self.validation_interval,
            "validation_batches": self.validation_batches,
            "max_samples_per_epoch": self.max_samples_per_epoch,
        }

    def _logits(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        adapted = self.adaptor(features)
        classification = adapted if self.adapt_classification_features else features
        return self.discriminator(adapted, classification)

    def fit(
        self, train_loader, *, device, validation_loader=None,
        work_dir=None, resume=False,
    ) -> SuperSimpleNet:
        if resume:
            raise NotImplementedError("SuperSimpleNet resume is not implemented")
        device = torch.device(device)
        self.to(device)
        segmentation_parameters, decision_parameters = self.discriminator.parameter_groups()
        optimizer = torch.optim.AdamW([
            {"params": self.adaptor.parameters(), "lr": self.adaptor_learning_rate},
            {"params": segmentation_parameters, "lr": self.segmentation_learning_rate,
             "weight_decay": 1e-5},
            {"params": decision_parameters, "lr": self.decision_learning_rate,
             "weight_decay": 1e-5},
        ])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.epochs * 0.8), int(self.epochs * 0.9)],
            gamma=self.scheduler_gamma,
        )
        best_auc = -math.inf
        best_state = None
        steps = 0
        epoch_progress = tqdm(
            range(1, self.epochs + 1), desc="SuperSimpleNet training",
            unit="epoch", dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            self.train(True)
            samples = 0
            produced = False
            for batch in train_loader:
                produced = True
                images = _clean_images(batch, device)
                with torch.no_grad():
                    features = self.features(images)
                adapted = self.adaptor(features)
                noisy_features, noisy_adapted, target_mask = self.anomaly_generator(
                    features, adapted
                )
                classification = (
                    noisy_adapted if self.adapt_classification_features
                    else noisy_features
                )
                anomaly_map, score = self.discriminator(noisy_adapted, classification)
                target_label = target_mask.flatten(1).amax(1)

                focal = _ssn_focal_loss(
                    torch.sigmoid(anomaly_map), target_mask, reduction=None
                )
                truncated = torch.zeros_like(anomaly_map)
                normal = target_mask == 0
                anomalous = target_mask > 0
                truncated[normal] = torch.clamp(anomaly_map[normal] + self.margin, min=0)
                truncated[anomalous] = torch.clamp(-anomaly_map[anomalous] + self.margin, min=0)
                good_loss = truncated[normal].mean() if normal.any() else 0.0
                bad_loss = truncated[anomalous].mean() if anomalous.any() else 0.0
                loss = good_loss + bad_loss + focal.mean()
                loss = loss + _ssn_focal_loss(torch.sigmoid(score), target_label)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
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
            scheduler.step()
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
            "fixed_training_duration": self.fixed_training_duration,
            "selected_validation_image_auroc": (
                None if self.fixed_training_duration or best_state is None else best_auc
            ),
        }
        return self

    @torch.no_grad()
    def predict_with_raw(self, images: torch.Tensor):
        if not self.is_fitted:
            raise RuntimeError("SuperSimpleNet must be fitted before predict")
        self.eval()
        raw_map, raw_score = self._logits(images)
        raw_map = F.interpolate(
            raw_map, images.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.gaussian_sigma > 0:
            kernel = 2 * math.ceil(3 * self.gaussian_sigma) + 1
            raw_map = gaussian_blur(
                raw_map, [kernel, kernel], [self.gaussian_sigma, self.gaussian_sigma]
            )
        anomaly_map = torch.sigmoid(raw_map)
        anomaly_score = torch.sigmoid(raw_score)
        prediction = AnomalyPrediction(anomaly_score, anomaly_map)
        return prediction, raw_score, raw_map

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        return self.predict_with_raw(images)[0]
