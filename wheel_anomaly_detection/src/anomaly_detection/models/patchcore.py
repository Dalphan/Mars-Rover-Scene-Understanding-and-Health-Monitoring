from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor

from .base import AnomalyDetector


class PatchCore(AnomalyDetector):
    """Frozen PatchCore feature extractor.

    This milestone only builds locally aggregated patch embeddings. Coreset
    selection, memory-bank fitting, and nearest-neighbour scoring are added in
    the training and inference milestone.
    """

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        pretrained: bool = True,
        layers: Sequence[str] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        pool_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one feature node")
        if not 0 < coreset_sampling_ratio <= 1:
            raise ValueError("coreset_sampling_ratio must be in (0, 1]")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be at least 1")
        if pool_kernel_size < 1 or pool_kernel_size % 2 == 0:
            raise ValueError("pool_kernel_size must be a positive odd integer")

        weights = get_model_weights(backbone).DEFAULT if pretrained else None
        backbone_model = get_model(backbone, weights=weights)
        return_nodes = {layer: layer for layer in layers}
        self.feature_extractor = create_feature_extractor(
            backbone_model,
            return_nodes=return_nodes,
        )
        self.feature_extractor.requires_grad_(False)

        self.backbone = backbone
        self.pretrained = pretrained
        self.layers = tuple(layers)
        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        self.num_neighbors = int(num_neighbors)
        self.pool_kernel_size = int(pool_kernel_size)
        self.register_buffer("memory_bank", torch.empty(0), persistent=True)
        self.train(False)

    @property
    def is_fitted(self) -> bool:
        """Return whether a patch memory bank has been attached."""
        return self.memory_bank.numel() > 0

    def checkpoint_config(self) -> dict[str, object]:
        """Return settings that must match the stored PatchCore memory bank."""
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "pool_kernel_size": self.pool_kernel_size,
        }

    def train(self, mode: bool = True) -> PatchCore:
        """Keep the frozen pretrained feature extractor in evaluation mode."""
        super().train(False)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return a dense patch embedding map without anomaly scores."""
        features = self.feature_extractor(images)
        target_height = max(feature.shape[-2] for feature in features.values())
        target_width = max(feature.shape[-1] for feature in features.values())
        aggregated = []

        for layer in self.layers:
            feature = F.avg_pool2d(
                features[layer],
                kernel_size=self.pool_kernel_size,
                stride=1,
                padding=self.pool_kernel_size // 2,
            )
            if feature.shape[-2:] != (target_height, target_width):
                feature = F.interpolate(
                    feature,
                    size=(target_height, target_width),
                    mode="bilinear",
                    align_corners=False,
                )
            aggregated.append(feature)

        return torch.cat(aggregated, dim=1)

    def _prepare_state_dict_for_load(self, state_dict) -> None:
        memory_bank = state_dict.get("memory_bank")
        if memory_bank is not None and memory_bank.shape != self.memory_bank.shape:
            self.memory_bank = torch.empty_like(memory_bank)
