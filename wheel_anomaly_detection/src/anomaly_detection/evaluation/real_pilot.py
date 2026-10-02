from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm import tqdm

from ..models import AnomalyDetector
from .diagnostics import PerRegionOverlap
from .metrics import BinaryHistogramMetrics, ExactBinaryMetrics


def _batch_metadata_value(
    metadata: Mapping[str, Any], key: str, index: int
) -> Any:
    values = metadata.get(key)
    if values is None:
        return ""
    if isinstance(values, torch.Tensor):
        value = values[index]
        return value.item() if value.ndim == 0 else value.tolist()
    value = values[index]
    return value.item() if isinstance(value, torch.Tensor) else value


@torch.no_grad()
def evaluate_real_wheel_pilot(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    image_for_display: Callable[[torch.Tensor], torch.Tensor],
    histogram_bins: int = 2048,
    pro_bins: int = 256,
    pro_max_fpr: float = 0.30,
    description: str = "Real-wheel zero-shot inference",
) -> dict[str, Any]:
    """Evaluate a frozen detector once on every real-wheel crop."""
    if not detector.is_fitted:
        raise RuntimeError("The anomaly detector must be fitted before evaluation")

    device = torch.device(device)
    detector.to(device)
    detector.train(False)
    image_metric = ExactBinaryMetrics()
    pixel_metric = BinaryHistogramMetrics(histogram_bins)
    pro_metric = PerRegionOverlap(pro_bins, pro_max_fpr)
    records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    scores_at_zero = 0
    scores_at_one = 0

    for batch in tqdm(loader, desc=description):
        images = torch.as_tensor(batch["image"])
        labels = torch.as_tensor(batch["label"]).reshape(-1).cpu()
        masks = torch.as_tensor(batch["anomaly_mask"])
        if images.ndim != 4 or images.shape[0] != labels.numel():
            raise ValueError("image batch and labels have incompatible shapes")
        if images.shape[0] != 1:
            raise ValueError(
                "The real zero-shot protocol requires batch_size=1 so each "
                "frame is inferred independently"
            )

        images_device = images.to(device, non_blocking=True)
        predict_with_raw = getattr(detector, "predict_with_raw", None)
        if predict_with_raw is None:
            prediction = detector.predict(images_device)
            raw_image_scores = prediction.anomaly_score
        else:
            prediction, raw_image_scores, _ = predict_with_raw(images_device)
        raw_image_scores = raw_image_scores.detach().reshape(-1).cpu()
        if raw_image_scores.shape != labels.shape:
            raise ValueError("raw image scores must match labels")
        if prediction.anomaly_map.shape != masks.shape:
            raise ValueError(
                "predicted anomaly maps and approximate masks must have "
                f"matching shapes: {prediction.anomaly_map.shape} != {masks.shape}"
            )

        image_metric.update(raw_image_scores, labels)
        pixel_metric.update(prediction.anomaly_map, masks)
        pro_metric.update(prediction.anomaly_map, masks)
        normalized_scores = prediction.anomaly_score.detach().reshape(-1).cpu()
        scores_at_zero += int((normalized_scores == 0).sum())
        scores_at_one += int((normalized_scores == 1).sum())
        metadata = batch.get("metadata", {})

        for index, label_tensor in enumerate(labels):
            label = int(label_tensor)
            image_id = str(_batch_metadata_value(metadata, "image_id", index))
            condition = str(_batch_metadata_value(metadata, "condition", index))
            sol = _batch_metadata_value(metadata, "sol", index)
            anomaly_map = prediction.anomaly_map[index].detach().cpu()
            record = {
                "image_id": image_id,
                "sol": int(sol) if str(sol) else None,
                "condition": condition or ("hole" if label else "no_visible_hole"),
                "label": label,
                "raw_anomaly_score": float(raw_image_scores[index]),
                "normalized_anomaly_score": float(normalized_scores[index]),
                "anomaly_map_min": float(anomaly_map.min()),
                "anomaly_map_mean": float(anomaly_map.mean()),
                "anomaly_map_max": float(anomaly_map.max()),
            }
            records.append(record)
            samples.append(
                {
                    **record,
                    "image": image_for_display(images[index]).detach().cpu(),
                    "anomaly_mask": masks[index].detach().cpu(),
                    "anomaly_map": anomaly_map,
                }
            )

    if not records:
        raise ValueError("Evaluation loader produced no images")
    image_result = image_metric.compute()
    pixel_result = pixel_metric.compute()
    pro_result = pro_metric.compute()
    positive_pixels = int(pixel_metric.positive_histogram.sum())
    negative_pixels = int(pixel_metric.negative_histogram.sum())
    total_pixels = positive_pixels + negative_pixels
    saturated = scores_at_zero + scores_at_one
    metrics = {
        "image_auroc": image_result["auroc"],
        "image_average_precision": image_result["average_precision"],
        "pixel_auroc": pixel_result["auroc"],
        "pixel_average_precision": pixel_result["average_precision"],
        "aupro": pro_result["aupro"],
        "aupro_max_fpr": pro_result["max_fpr"],
        "aupro_by_max_fpr": pro_result["aupro_by_max_fpr"],
        "pro_num_regions": pro_result["num_regions"],
        "random_pixel_ap_baseline": positive_pixels / total_pixels,
        "approximate_anomaly_pixel_fraction": positive_pixels / total_pixels,
        "num_images": len(records),
        "num_clean_images": sum(record["label"] == 0 for record in records),
        "num_hole_images": sum(record["label"] == 1 for record in records),
        "normalized_image_scores_at_zero": scores_at_zero,
        "normalized_image_scores_at_one": scores_at_one,
        "normalized_image_score_saturation_fraction": saturated / len(records),
        "pixel_metric_domain": "entire_resized_crop",
        "pixel_ground_truth": "approximate_hole_masks",
        "target_wheel_metrics_available": False,
    }
    return {
        "metrics": metrics,
        "records": records,
        "samples": samples,
        "pro_curve": pro_result["curve"],
    }


def plot_real_wheel_predictions(
    samples: list[dict[str, Any]],
    *,
    title: str,
    colormap: str = "magma",
    overlay_alpha: float = 0.55,
) -> Figure:
    """Plot the real crop, normalized anomaly map and overlay for each frame."""
    if not samples:
        raise ValueError("samples cannot be empty")
    if not 0 <= overlay_alpha <= 1:
        raise ValueError("overlay_alpha must be in [0, 1]")

    figure = Figure(figsize=(14, 4.1 * len(samples)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(samples), 3, squeeze=False)
    heatmap_artist = None
    for row, sample in enumerate(samples):
        image = sample["image"].permute(1, 2, 0).numpy()
        anomaly_mask = sample["anomaly_mask"].squeeze().numpy() > 0
        anomaly_map = sample["anomaly_map"].squeeze().numpy()
        image_id = sample.get("image_id") or "unknown"
        condition = sample.get("condition") or (
            "hole" if sample.get("label") else "no_visible_hole"
        )
        score_text = (
            f"raw={sample['raw_anomaly_score']:.4g} | "
            f"normalized={sample['normalized_anomaly_score']:.3f}"
        )

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{image_id} — {condition}\n{score_text}")
        heatmap_artist = axes[row, 1].imshow(
            anomaly_map, cmap=colormap, vmin=0, vmax=1
        )
        axes[row, 1].set_title("Normalized anomaly map")
        axes[row, 2].imshow(image)
        axes[row, 2].imshow(
            anomaly_map,
            cmap=colormap,
            vmin=0,
            vmax=1,
            alpha=overlay_alpha,
        )
        if anomaly_mask.any() and not anomaly_mask.all():
            for axis in axes[row, 1:]:
                axis.contour(
                    anomaly_mask,
                    levels=[0.5],
                    colors="lime",
                    linewidths=1.4,
                )
        axes[row, 2].set_title("Overlay\nlime=approximate hole boundary")
        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.colorbar(
        heatmap_artist,
        ax=axes[:, 1:].ravel().tolist(),
        label="Normalized anomaly score",
        shrink=0.8,
    )
    return figure


def save_real_wheel_pilot_results(
    evaluation: Mapping[str, Any],
    output_dir: str | Path,
    *,
    context: Mapping[str, Any],
    rows_per_figure: int = 4,
    colormap: str = "magma",
    overlay_alpha: float = 0.55,
    dpi: int = 150,
) -> tuple[list[Figure], dict[str, Any]]:
    """Persist metrics, per-image scores, PRO curve and paginated figures."""
    if rows_per_figure < 1:
        raise ValueError("rows_per_figure must be at least 1")
    if dpi < 1:
        raise ValueError("dpi must be at least 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    metrics_payload = {"context": dict(context), "metrics": evaluation["metrics"]}
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )

    records = list(evaluation["records"])
    scores_path = output_dir / "image_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    pro_curve_path = output_dir / "pro_curve.csv"
    with pro_curve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["false_positive_rate", "pro"])
        writer.writeheader()
        writer.writerows(evaluation["pro_curve"])

    figures: list[Figure] = []
    figure_paths: list[Path] = []
    samples = list(evaluation["samples"])
    for start in range(0, len(samples), rows_per_figure):
        page = start // rows_per_figure + 1
        page_samples = samples[start : start + rows_per_figure]
        figure = plot_real_wheel_predictions(
            page_samples,
            title=f"{context['model_name']} real-wheel zero-shot — page {page}",
            colormap=colormap,
            overlay_alpha=overlay_alpha,
        )
        figure_path = output_dir / f"predictions_page_{page:02d}.png"
        figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
        figures.append(figure)
        figure_paths.append(figure_path)

    return figures, {
        "metrics": metrics_path,
        "image_scores": scores_path,
        "pro_curve": pro_curve_path,
        "figures": figure_paths,
    }
