from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AnomalyPrediction:
    """Model-independent anomaly output with normalized scores."""

    anomaly_score: torch.Tensor
    anomaly_map: torch.Tensor

    def __post_init__(self) -> None:
        if self.anomaly_score.ndim != 1:
            raise ValueError("anomaly_score must have shape [B]")
        if self.anomaly_map.ndim != 4 or self.anomaly_map.shape[1] != 1:
            raise ValueError("anomaly_map must have shape [B, 1, H, W]")
        if self.anomaly_score.shape[0] != self.anomaly_map.shape[0]:
            raise ValueError("anomaly_score and anomaly_map batch sizes must match")
        for name, value in (
            ("anomaly_score", self.anomaly_score),
            ("anomaly_map", self.anomaly_map),
        ):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite floating-point values")
            if value.numel() and (value.min() < 0 or value.max() > 1):
                raise ValueError(f"{name} must be normalized to [0, 1]")


class AnomalyDetector(nn.Module):
    """Small common interface shared by all anomaly detectors."""

    @property
    def is_fitted(self) -> bool:
        raise NotImplementedError

    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        *,
        device: torch.device | str,
        validation_loader: Iterable[Mapping[str, Any]] | None = None,
        work_dir: str | Path | None = None,
        resume: bool = False,
    ) -> AnomalyDetector:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        raise NotImplementedError

    def checkpoint_config(self) -> dict[str, Any]:
        """Return behavior-affecting constructor settings for strict reloads."""
        return {}

    def save(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> Path:
        """Save model state and small run metadata in one checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "model_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "model_config": self.checkpoint_config(),
                "model_state_dict": self.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )
        return path

    def _prepare_state_dict_for_load(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Allow models with dynamically sized buffers to prepare before load."""

    def load(
        self,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load a trusted project checkpoint and return its metadata."""
        payload = torch.load(Path(path), map_location=map_location)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("Checkpoint does not contain model_state_dict")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema {payload.get('schema_version')!r}; "
                f"expected {CHECKPOINT_SCHEMA_VERSION}"
            )
        expected_class = f"{type(self).__module__}.{type(self).__qualname__}"
        if payload.get("model_class") != expected_class:
            raise ValueError(
                f"Checkpoint model class {payload.get('model_class')!r} does not "
                f"match {expected_class!r}"
            )
        saved_config = payload.get("model_config")
        expected_config = self.checkpoint_config()
        if saved_config != expected_config:
            raise ValueError(
                "Checkpoint model configuration does not match the current model: "
                f"saved={saved_config!r}, current={expected_config!r}"
            )
        state_dict = payload["model_state_dict"]
        self._prepare_state_dict_for_load(state_dict)
        self.load_state_dict(state_dict)
        return dict(payload.get("metadata", {}))
