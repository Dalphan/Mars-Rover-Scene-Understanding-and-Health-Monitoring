from __future__ import annotations

from omegaconf import DictConfig

from .base import AnomalyDetector


def build_model(cfg: DictConfig) -> AnomalyDetector:
    """Build the anomaly-detection model selected by Hydra."""
    if cfg.model.name == "patchcore":
        from .patchcore import PatchCore

        return PatchCore(
            backbone=str(cfg.model.backbone),
            pretrained=bool(cfg.model.pretrained),
            layers=tuple(cfg.model.layers),
            coreset_sampling_ratio=float(cfg.model.coreset_sampling_ratio),
            num_neighbors=int(cfg.model.num_neighbors),
            pool_kernel_size=int(cfg.model.pool_kernel_size),
        )

    raise ValueError(
        f"Unknown model name {cfg.model.name!r}. Expected one of: patchcore."
    )
