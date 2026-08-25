from __future__ import annotations

import torch
from omegaconf import DictConfig

from ..models import AnomalyPrediction


ESSENTIAL_METRIC_NAMES = (
    "image_auroc",
    "image_average_precision",
    "pixel_auroc",
    "pixel_average_precision",
)


class ExactBinaryMetrics:
    """Exact binary AUROC and average precision for small score collections."""

    def __init__(self) -> None:
        self._scores: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        scores = scores.detach().reshape(-1).cpu()
        targets = targets.detach().reshape(-1).cpu().bool()
        if scores.numel() != targets.numel():
            raise ValueError("scores and targets must contain the same number of values")
        if not scores.numel():
            return
        if not scores.is_floating_point() or not torch.isfinite(scores).all():
            raise ValueError("scores must contain finite floating-point values")
        if scores.min() < 0 or scores.max() > 1:
            raise ValueError("scores must be normalized to [0, 1]")
        self._scores.append(scores.clone())
        self._targets.append(targets.clone())

    def compute(self) -> dict[str, float]:
        if not self._scores:
            raise ValueError("AUROC and average precision require both target classes")
        scores = torch.cat(self._scores)
        targets = torch.cat(self._targets)
        positives = int(targets.sum())
        negatives = targets.numel() - positives
        if positives == 0 or negatives == 0:
            raise ValueError("AUROC and average precision require both target classes")

        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        sorted_targets = targets[order]
        _, group_counts = torch.unique_consecutive(sorted_scores, return_counts=True)
        group_ends = group_counts.cumsum(0) - 1
        true_positives = sorted_targets.cumsum(0)[group_ends].double()
        false_positives = (~sorted_targets).cumsum(0)[group_ends].double()
        recall = true_positives / positives
        false_positive_rate = false_positives / negatives
        precision = true_positives / (true_positives + false_positives)

        zero = torch.zeros(1, dtype=torch.float64)
        auroc = torch.trapezoid(
            torch.cat((zero, recall)),
            torch.cat((zero, false_positive_rate)),
        )
        recall_increment = recall - torch.cat((zero, recall[:-1]))
        average_precision = (precision * recall_increment).sum()
        return {
            "auroc": float(auroc),
            "average_precision": float(average_precision),
        }


class BinaryHistogramMetrics:
    """Memory-bounded binary AUROC and average precision accumulator."""

    def __init__(self, num_bins: int = 2048) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        self.num_bins = int(num_bins)
        self.positive_histogram = torch.zeros(num_bins, dtype=torch.int64)
        self.negative_histogram = torch.zeros(num_bins, dtype=torch.int64)

    @torch.no_grad()
    def update(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        scores = scores.detach().reshape(-1).cpu()
        targets = targets.detach().reshape(-1).cpu().bool()
        if scores.numel() != targets.numel():
            raise ValueError("scores and targets must contain the same number of values")
        if valid_mask is not None:
            valid_mask = valid_mask.detach().reshape(-1).cpu().bool()
            if valid_mask.numel() != scores.numel():
                raise ValueError("valid_mask must match scores")
            scores = scores[valid_mask]
            targets = targets[valid_mask]
        if not scores.numel():
            return
        if not scores.is_floating_point() or not torch.isfinite(scores).all():
            raise ValueError("scores must contain finite floating-point values")
        if scores.min() < 0 or scores.max() > 1:
            raise ValueError("scores must be normalized to [0, 1]")

        bins = (scores * self.num_bins).long().clamp(max=self.num_bins - 1)
        self.positive_histogram += torch.bincount(
            bins[targets], minlength=self.num_bins
        )
        self.negative_histogram += torch.bincount(
            bins[~targets], minlength=self.num_bins
        )

    def compute(self) -> dict[str, float]:
        positives = int(self.positive_histogram.sum())
        negatives = int(self.negative_histogram.sum())
        if positives == 0 or negatives == 0:
            raise ValueError("AUROC and average precision require both target classes")

        true_positives = self.positive_histogram.flip(0).cumsum(0).double()
        false_positives = self.negative_histogram.flip(0).cumsum(0).double()
        recall = true_positives / positives
        false_positive_rate = false_positives / negatives
        precision = true_positives / (true_positives + false_positives).clamp_min(1)

        zero = torch.zeros(1, dtype=torch.float64)
        auroc = torch.trapezoid(
            torch.cat((zero, recall)),
            torch.cat((zero, false_positive_rate)),
        )
        recall_increment = recall - torch.cat((zero, recall[:-1]))
        average_precision = (precision * recall_increment).sum()
        return {
            "auroc": float(auroc),
            "average_precision": float(average_precision),
        }


class AnomalyMetrics:
    """Accumulate the four project metrics at image and pixel level."""

    def __init__(self, histogram_bins: int = 2048) -> None:
        self.image = ExactBinaryMetrics()
        self.pixel = BinaryHistogramMetrics(histogram_bins)

    @torch.no_grad()
    def update(
        self,
        prediction: AnomalyPrediction,
        labels: torch.Tensor,
        anomaly_masks: torch.Tensor,
        *,
        valid_pixel_mask: torch.Tensor | None = None,
    ) -> None:
        labels = labels.reshape(-1)
        if labels.shape != prediction.anomaly_score.shape:
            raise ValueError("labels must match anomaly_score shape")

        anomaly_masks = anomaly_masks > 0
        if anomaly_masks.ndim == 3:
            anomaly_masks = anomaly_masks.unsqueeze(1)
        if anomaly_masks.shape != prediction.anomaly_map.shape:
            raise ValueError("anomaly_masks must match anomaly_map shape")
        if valid_pixel_mask is not None:
            valid_pixel_mask = valid_pixel_mask > 0
            if valid_pixel_mask.ndim == 3:
                valid_pixel_mask = valid_pixel_mask.unsqueeze(1)
            if valid_pixel_mask.shape != prediction.anomaly_map.shape:
                raise ValueError("valid_pixel_mask must match anomaly_map shape")

        self.image.update(prediction.anomaly_score, labels)
        self.pixel.update(
            prediction.anomaly_map,
            anomaly_masks,
            valid_mask=valid_pixel_mask,
        )

    def compute(self) -> dict[str, float]:
        image = self.image.compute()
        pixel = self.pixel.compute()
        return {
            "image_auroc": image["auroc"],
            "image_average_precision": image["average_precision"],
            "pixel_auroc": pixel["auroc"],
            "pixel_average_precision": pixel["average_precision"],
        }


def build_metrics(cfg: DictConfig) -> AnomalyMetrics:
    """Build the fixed essential metric set from Hydra."""
    configured = tuple(str(name) for name in cfg.evaluation.metrics)
    if configured != ESSENTIAL_METRIC_NAMES:
        raise ValueError(
            f"evaluation.metrics must be exactly {ESSENTIAL_METRIC_NAMES}, got {configured}"
        )
    return AnomalyMetrics(histogram_bins=int(cfg.evaluation.histogram_bins))


def update_metrics_from_batch(
    metrics: AnomalyMetrics,
    prediction: AnomalyPrediction,
    batch: dict[str, torch.Tensor],
    *,
    restrict_pixels_to_target_mask: bool = False,
) -> None:
    """Update metrics using the common dataset batch contract."""
    valid_pixel_mask = batch["target_mask"] if restrict_pixels_to_target_mask else None
    metrics.update(
        prediction,
        batch["label"],
        batch["anomaly_mask"],
        valid_pixel_mask=valid_pixel_mask,
    )
