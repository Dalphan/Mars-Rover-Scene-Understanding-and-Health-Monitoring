from __future__ import annotations

import csv
import json
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..data import WheelPreprocessor
from ..models import AnomalyPrediction
from .metrics import BinaryHistogramMetrics, ExactBinaryMetrics
from .visualization import plot_anomaly_visualizations


DEFAULT_GROUP_FIELDS = ("severity", "camera_pose", "lighting", "wear")


def _score_statistics(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    scores = torch.tensor(values, dtype=torch.float64)
    return {
        "count": scores.numel(),
        "minimum": float(scores.min()),
        "q25": float(torch.quantile(scores, 0.25)),
        "median": float(torch.quantile(scores, 0.50)),
        "q75": float(torch.quantile(scores, 0.75)),
        "maximum": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std(unbiased=False)),
    }


def _connected_components(mask: torch.Tensor) -> list[list[int]]:
    """Return 4-connected positive regions as flattened pixel indices."""
    mask = mask.detach().cpu().bool().squeeze()
    if mask.ndim != 2:
        raise ValueError("PRO masks must be two-dimensional after squeezing")
    height, width = mask.shape
    remaining = set(torch.nonzero(mask.flatten(), as_tuple=False).flatten().tolist())
    components: list[list[int]] = []
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = [start]
        while queue:
            index = queue.popleft()
            row, column = divmod(index, width)
            neighbours = []
            if row > 0:
                neighbours.append(index - width)
            if row + 1 < height:
                neighbours.append(index + width)
            if column > 0:
                neighbours.append(index - 1)
            if column + 1 < width:
                neighbours.append(index + 1)
            for neighbour in neighbours:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
                    component.append(neighbour)
        components.append(component)
    return components


class PerRegionOverlap:
    """Memory-bounded PRO curve and normalized AUPRO accumulator."""

    def __init__(self, num_bins: int = 256, max_fpr: float = 0.30) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if not 0 < max_fpr <= 1:
            raise ValueError("max_fpr must be in (0, 1]")
        self.num_bins = int(num_bins)
        self.max_fpr = float(max_fpr)
        self.normal_histogram = torch.zeros(num_bins, dtype=torch.int64)
        self.region_overlap_sum = torch.zeros(num_bins, dtype=torch.float64)
        self.num_regions = 0

    @torch.no_grad()
    def update(
        self,
        scores: torch.Tensor,
        anomaly_masks: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        scores = scores.detach().cpu()
        anomaly_masks = anomaly_masks.detach().cpu() > 0
        if scores.ndim == 3:
            scores = scores.unsqueeze(1)
        if anomaly_masks.ndim == 3:
            anomaly_masks = anomaly_masks.unsqueeze(1)
        if scores.shape != anomaly_masks.shape:
            raise ValueError("scores and anomaly_masks must have matching shapes")
        if valid_mask is None:
            valid_mask = torch.ones_like(anomaly_masks, dtype=torch.bool)
        else:
            valid_mask = valid_mask.detach().cpu() > 0
            if valid_mask.ndim == 3:
                valid_mask = valid_mask.unsqueeze(1)
            if valid_mask.shape != scores.shape:
                raise ValueError("valid_mask must match scores")

        for score, anomaly_mask, valid in zip(scores, anomaly_masks, valid_mask):
            score = score.squeeze(0)
            anomaly_mask = anomaly_mask.squeeze(0) & valid.squeeze(0)
            valid = valid.squeeze(0)
            normal_scores = score[valid & ~anomaly_mask]
            if normal_scores.numel():
                bins = (normal_scores * self.num_bins).long().clamp(
                    min=0, max=self.num_bins - 1
                )
                self.normal_histogram += torch.bincount(
                    bins, minlength=self.num_bins
                )
            flattened_scores = score.flatten()
            for component in _connected_components(anomaly_mask):
                component_scores = flattened_scores[component]
                bins = (component_scores * self.num_bins).long().clamp(
                    min=0, max=self.num_bins - 1
                )
                histogram = torch.bincount(bins, minlength=self.num_bins)
                self.region_overlap_sum += (
                    histogram.flip(0).cumsum(0).double() / len(component)
                )
                self.num_regions += 1

    @staticmethod
    def _normalized_area(
        fpr: torch.Tensor,
        pro: torch.Tensor,
        max_fpr: float,
    ) -> float:
        below = fpr <= max_fpr
        clipped_fpr = fpr[below]
        clipped_pro = pro[below]
        if clipped_fpr[-1] < max_fpr:
            upper_index = int(torch.nonzero(fpr > max_fpr)[0])
            lower_index = upper_index - 1
            denominator = fpr[upper_index] - fpr[lower_index]
            weight = (
                0.0
                if denominator == 0
                else float((max_fpr - fpr[lower_index]) / denominator)
            )
            interpolated = pro[lower_index] + weight * (
                pro[upper_index] - pro[lower_index]
            )
            clipped_fpr = torch.cat(
                (clipped_fpr, torch.tensor([max_fpr], dtype=torch.float64))
            )
            clipped_pro = torch.cat((clipped_pro, interpolated.reshape(1)))
        area = torch.trapezoid(clipped_pro, clipped_fpr) / max_fpr
        return float(area)

    def compute(self) -> dict[str, Any]:
        total_normal = int(self.normal_histogram.sum())
        if total_normal == 0 or self.num_regions == 0:
            raise ValueError("PRO requires normal pixels and at least one anomaly region")
        fpr = self.normal_histogram.flip(0).cumsum(0).double() / total_normal
        pro = self.region_overlap_sum / self.num_regions
        fpr = torch.cat((torch.zeros(1, dtype=torch.float64), fpr))
        pro = torch.cat((torch.zeros(1, dtype=torch.float64), pro))

        reported_max_fprs = sorted({0.05, 0.10, self.max_fpr})
        aupro_by_max_fpr = {
            f"{max_fpr:.2f}": self._normalized_area(fpr, pro, max_fpr)
            for max_fpr in reported_max_fprs
        }
        return {
            "aupro": aupro_by_max_fpr[f"{self.max_fpr:.2f}"],
            "max_fpr": self.max_fpr,
            "aupro_by_max_fpr": aupro_by_max_fpr,
            "num_regions": self.num_regions,
            "curve": [
                {"false_positive_rate": float(x), "pro": float(y)}
                for x, y in zip(fpr, pro)
            ],
        }


class AnomalyDiagnostics:
    """Collect score distributions, grouped image/pixel metrics, and PRO."""

    def __init__(
        self,
        manifest_rows: Sequence[Mapping[str, str]],
        *,
        preprocessor: WheelPreprocessor,
        histogram_bins: int = 2048,
        pro_bins: int = 256,
        pro_max_fpr: float = 0.30,
        group_fields: Sequence[str] = DEFAULT_GROUP_FIELDS,
        num_extreme_examples: int = 6,
        restrict_pixels_to_target_mask: bool = False,
    ) -> None:
        if num_extreme_examples < 1:
            raise ValueError("num_extreme_examples must be at least 1")
        self.preprocessor = preprocessor
        self.histogram_bins = int(histogram_bins)
        self.group_fields = tuple(group_fields)
        self.num_extreme_examples = int(num_extreme_examples)
        self.restrict_pixels_to_target_mask = bool(restrict_pixels_to_target_mask)
        self.pro = PerRegionOverlap(pro_bins, pro_max_fpr)
        self.records: list[dict[str, Any]] = []
        self.group_metrics: dict[str, dict[str, BinaryHistogramMetrics]] = {
            field: {} for field in self.group_fields
        }
        self.group_image_counts: dict[str, dict[str, int]] = {
            field: {} for field in self.group_fields
        }
        self.top_clean: list[dict[str, Any]] = []
        self.bottom_hole: list[dict[str, Any]] = []
        self.anomalous_pixels = 0
        self.valid_pixels = 0
        self.pair_groups = {
            row.get("pair_id", ""): {
                field: row.get(field, "") for field in self.group_fields
            }
            for row in manifest_rows
            if row.get("condition") == "hole" and row.get("pair_id")
        }

    @staticmethod
    def _batch_value(metadata: Mapping[str, Any], key: str, index: int) -> Any:
        values = metadata.get(key)
        if values is None:
            return ""
        value = values[index]
        return value.item() if isinstance(value, torch.Tensor) else value

    def _group_value(
        self, metadata: Mapping[str, Any], field: str, index: int
    ) -> str:
        value = str(self._batch_value(metadata, field, index) or "")
        if value:
            return value
        pair_id = str(self._batch_value(metadata, "pair_id", index) or "")
        return str(self.pair_groups.get(pair_id, {}).get(field, ""))

    def _example(
        self,
        batch: Mapping[str, Any],
        prediction: AnomalyPrediction,
        raw_scores: torch.Tensor,
        index: int,
    ) -> dict[str, Any]:
        metadata = batch.get("metadata", {})
        return {
            "image": self.preprocessor.image_for_display(batch["image"][index]).cpu(),
            "target_mask": batch["target_mask"][index].cpu(),
            "anomaly_mask": batch["anomaly_mask"][index].cpu(),
            "anomaly_map": prediction.anomaly_map[index].detach().cpu(),
            "anomaly_score": float(prediction.anomaly_score[index].detach().cpu()),
            "raw_anomaly_score": float(raw_scores[index].detach().cpu()),
            "label": int(batch["label"][index]),
            "image_id": self._batch_value(metadata, "image_id", index),
        }

    @torch.no_grad()
    def update(
        self,
        prediction: AnomalyPrediction,
        raw_image_scores: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> None:
        raw_image_scores = raw_image_scores.detach().reshape(-1).cpu()
        labels = torch.as_tensor(batch["label"]).reshape(-1).cpu()
        if raw_image_scores.shape != labels.shape:
            raise ValueError("raw_image_scores must match batch labels")
        anomaly_masks = batch["anomaly_mask"] > 0
        valid_mask = (
            batch["target_mask"] > 0
            if self.restrict_pixels_to_target_mask
            else torch.ones_like(anomaly_masks, dtype=torch.bool)
        )
        self.anomalous_pixels += int((anomaly_masks & valid_mask).sum())
        self.valid_pixels += int(valid_mask.sum())
        self.pro.update(prediction.anomaly_map, anomaly_masks, valid_mask)
        metadata = batch.get("metadata", {})

        for index, label_tensor in enumerate(labels):
            label = int(label_tensor)
            condition = "hole" if label else "clean"
            record = {
                "image_id": self._batch_value(metadata, "image_id", index),
                "pair_id": self._batch_value(metadata, "pair_id", index),
                "condition": condition,
                "raw_image_score": float(raw_image_scores[index]),
                "normalized_image_score": float(prediction.anomaly_score[index].cpu()),
            }
            for field in self.group_fields:
                value = self._group_value(metadata, field, index)
                record[field] = value
                if not value:
                    continue
                accumulator = self.group_metrics[field].setdefault(
                    value, BinaryHistogramMetrics(self.histogram_bins)
                )
                accumulator.update(
                    prediction.anomaly_map[index],
                    anomaly_masks[index],
                    valid_mask=valid_mask[index],
                )
                counts = self.group_image_counts[field]
                counts[value] = counts.get(value, 0) + 1
            self.records.append(record)

            example = self._example(batch, prediction, raw_image_scores, index)
            if label == 0:
                self.top_clean.append(example)
                self.top_clean.sort(
                    key=lambda item: item["raw_anomaly_score"], reverse=True
                )
                del self.top_clean[self.num_extreme_examples :]
            else:
                self.bottom_hole.append(example)
                self.bottom_hole.sort(key=lambda item: item["raw_anomaly_score"])
                del self.bottom_hole[self.num_extreme_examples :]

    def compute(self) -> dict[str, Any]:
        if not self.records or self.valid_pixels == 0:
            raise ValueError("Diagnostics require at least one evaluated image")
        grouped: dict[str, dict[str, Any]] = {}
        for field, values in self.group_metrics.items():
            grouped[field] = {}
            for value, metric in sorted(values.items()):
                positives = int(metric.positive_histogram.sum())
                negatives = int(metric.negative_histogram.sum())
                result = metric.compute() if positives and negatives else None
                grouped[field][value] = {
                    "pixel_average_precision": (
                        None if result is None else result["average_precision"]
                    ),
                    "anomalous_pixel_fraction": (
                        positives / (positives + negatives)
                        if positives + negatives
                        else 0.0
                    ),
                    "num_images": self.group_image_counts[field][value],
                }
        image_grouped: dict[str, dict[str, Any]] = {}
        for field in self.group_fields:
            image_grouped[field] = {}
            values = sorted(
                {str(row[field]) for row in self.records if row[field]}
            )
            for value in values:
                records = [row for row in self.records if row[field] == value]
                labels = torch.tensor(
                    [row["condition"] == "hole" for row in records],
                    dtype=torch.bool,
                )
                positives = int(labels.sum())
                negatives = labels.numel() - positives
                metrics = None
                if positives and negatives:
                    accumulator = ExactBinaryMetrics()
                    accumulator.update(
                        torch.tensor(
                            [row["normalized_image_score"] for row in records],
                            dtype=torch.float32,
                        ),
                        labels,
                    )
                    metrics = accumulator.compute()
                image_grouped[field][value] = {
                    "image_auroc": None if metrics is None else metrics["auroc"],
                    "image_average_precision": (
                        None if metrics is None else metrics["average_precision"]
                    ),
                    "num_images": len(records),
                    "num_clean_images": negatives,
                    "num_anomalous_images": positives,
                }

        raw_scores = {
            condition: [
                row["raw_image_score"]
                for row in self.records
                if row["condition"] == condition
            ]
            for condition in ("clean", "hole")
        }
        return {
            "raw_image_score_distribution": {
                condition: _score_statistics(values)
                for condition, values in raw_scores.items()
            },
            "image_metrics_by_group": image_grouped,
            "pixel_average_precision_by_group": grouped,
            "anomalous_pixel_fraction": self.anomalous_pixels / self.valid_pixels,
            "random_pixel_ap_baseline": self.anomalous_pixels / self.valid_pixels,
            "pro": self.pro.compute(),
        }


def save_anomaly_diagnostics(
    diagnostics: AnomalyDiagnostics,
    output_dir: str | Path,
    *,
    split: str,
    colormap: str = "magma",
    overlay_alpha: float = 0.55,
    dpi: int = 150,
) -> tuple[dict[str, Any], list[Path]]:
    """Save diagnostic JSON, score/PRO CSVs, plots, and extreme examples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = diagnostics.compute()
    json_result = {**result, "pro": {k: v for k, v in result["pro"].items() if k != "curve"}}
    summary_path = output_dir / f"{split}_diagnostics.json"
    summary_path.write_text(json.dumps(json_result, indent=2) + "\n", encoding="utf-8")

    score_path = output_dir / f"{split}_image_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics.records[0]))
        writer.writeheader()
        writer.writerows(diagnostics.records)

    pro_path = output_dir / f"{split}_pro_curve.csv"
    with pro_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["false_positive_rate", "pro"])
        writer.writeheader()
        writer.writerows(result["pro"]["curve"])

    score_figure = Figure(figsize=(8, 5), layout="constrained")
    FigureCanvasAgg(score_figure)
    axis = score_figure.subplots()
    for condition, color in (("clean", "tab:blue"), ("hole", "tab:orange")):
        values = [
            row["raw_image_score"]
            for row in diagnostics.records
            if row["condition"] == condition
        ]
        axis.hist(values, bins=50, density=True, alpha=0.55, label=condition, color=color)
    axis.set_title(f"{split.title()} raw image-score distribution")
    axis.set_xlabel("Raw PatchCore image score")
    axis.set_ylabel("Density")
    axis.legend()
    score_plot_path = output_dir / f"{split}_raw_score_distribution.png"
    score_figure.savefig(score_plot_path, dpi=dpi, bbox_inches="tight")
    score_figure.clear()

    paths = [summary_path, score_path, pro_path, score_plot_path]
    for name, examples, title in (
        ("highest_clean_scores", diagnostics.top_clean, "Clean images with highest raw scores"),
        ("lowest_hole_scores", diagnostics.bottom_hole, "Hole images with lowest raw scores"),
    ):
        if not examples:
            continue
        figure = plot_anomaly_visualizations(
            examples,
            title=f"{split.title()} — {title}",
            colormap=colormap,
            overlay_alpha=overlay_alpha,
        )
        path = output_dir / f"{split}_{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        figure.clear()
        paths.append(path)
    return json_result, paths
