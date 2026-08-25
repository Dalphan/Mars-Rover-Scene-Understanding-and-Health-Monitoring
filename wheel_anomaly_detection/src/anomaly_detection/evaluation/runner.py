from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from tqdm.auto import tqdm

from ..models import AnomalyDetector
from .diagnostics import AnomalyDiagnostics
from .metrics import AnomalyMetrics, update_metrics_from_batch


@torch.no_grad()
def evaluate_anomaly_detector(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    histogram_bins: int = 2048,
    restrict_pixels_to_target_mask: bool = False,
    description: str = "Evaluation",
    diagnostics: AnomalyDiagnostics | None = None,
) -> dict[str, float]:
    """Evaluate one fitted detector and optionally collect diagnostics."""
    if not detector.is_fitted:
        raise RuntimeError("The anomaly detector must be fitted before evaluation")
    device = torch.device(device)
    detector.to(device)
    detector.train(False)
    metrics = AnomalyMetrics(histogram_bins=histogram_bins)
    num_images = 0

    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device, non_blocking=True)
        if diagnostics is None:
            prediction = detector.predict(images)
            raw_image_scores = None
        else:
            predict_with_raw = getattr(detector, "predict_with_raw", None)
            if predict_with_raw is None:
                raise RuntimeError(
                    "Diagnostics require a detector implementing predict_with_raw"
                )
            prediction, raw_image_scores, _ = predict_with_raw(images)
        update_metrics_from_batch(
            metrics,
            prediction,
            batch,
            restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
        )
        if diagnostics is not None:
            diagnostics.update(prediction, raw_image_scores, batch)
        num_images += images.shape[0]
    if num_images == 0:
        raise ValueError("Evaluation loader produced no images")
    return metrics.compute()
