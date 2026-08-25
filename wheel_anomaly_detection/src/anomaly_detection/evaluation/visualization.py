from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..data import WheelPreprocessor
from ..models import AnomalyDetector


@torch.no_grad()
def collect_anomaly_visualization_samples(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    preprocessor: WheelPreprocessor,
    num_clean: int = 3,
    num_anomalous: int = 3,
) -> list[dict[str, Any]]:
    """Collect a small, deterministic set of clean and anomalous predictions."""
    if num_clean < 0 or num_anomalous < 0 or num_clean + num_anomalous == 0:
        raise ValueError("At least one non-negative visualization quota is required")
    if not detector.is_fitted:
        raise RuntimeError("The anomaly detector must be fitted before visualization")

    device = torch.device(device)
    detector.to(device)
    detector.train(False)
    remaining = {0: int(num_clean), 1: int(num_anomalous)}
    samples: list[dict[str, Any]] = []

    for batch in loader:
        labels = torch.as_tensor(batch["label"]).reshape(-1)
        selected = [
            index
            for index, label in enumerate(labels.tolist())
            if label in remaining and remaining[label] > 0
        ]
        if not selected:
            continue

        images = batch["image"][selected].to(device, non_blocking=True)
        prediction = detector.predict(images)
        metadata = batch.get("metadata", {})
        image_ids = metadata.get("image_id", [None] * labels.numel())
        for prediction_index, batch_index in enumerate(selected):
            label = int(labels[batch_index])
            if remaining[label] == 0:
                continue
            samples.append(
                {
                    "image": preprocessor.image_for_display(
                        batch["image"][batch_index]
                    ).cpu(),
                    "target_mask": batch["target_mask"][batch_index].cpu(),
                    "anomaly_mask": batch["anomaly_mask"][batch_index].cpu(),
                    "anomaly_map": prediction.anomaly_map[prediction_index].cpu(),
                    "anomaly_score": float(
                        prediction.anomaly_score[prediction_index].cpu()
                    ),
                    "label": label,
                    "image_id": image_ids[batch_index],
                }
            )
            remaining[label] -= 1
        if not any(remaining.values()):
            break

    if any(remaining.values()):
        raise ValueError(
            "The loader does not contain enough samples for the requested quotas: "
            f"missing clean={remaining[0]}, anomalous={remaining[1]}"
        )
    return samples


def plot_anomaly_visualizations(
    samples: list[dict[str, Any]],
    *,
    title: str,
    colormap: str = "magma",
    overlay_alpha: float = 0.55,
) -> Figure:
    """Plot input, ground truth, normalized anomaly map, and image overlay."""
    if not samples:
        raise ValueError("samples cannot be empty")
    if not 0 <= overlay_alpha <= 1:
        raise ValueError("overlay_alpha must be in [0, 1]")

    figure = Figure(figsize=(16, 3.7 * len(samples)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(samples), 4, squeeze=False)
    heatmap_artist = None
    for row, sample in enumerate(samples):
        image = sample["image"].permute(1, 2, 0).numpy()
        target_mask = sample["target_mask"].squeeze().numpy() > 0
        anomaly_mask = sample["anomaly_mask"].squeeze().numpy() > 0
        anomaly_map = sample["anomaly_map"].squeeze().numpy()
        label = "anomalous" if sample["label"] else "clean"
        image_id = sample.get("image_id") or "unknown"
        raw_score = sample.get("raw_anomaly_score")
        score_text = f"score={sample['anomaly_score']:.3f}"
        if raw_score is not None:
            score_text += f" | raw={raw_score:.3f}"

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(
            f"Input — {label}\n{score_text} | {image_id}"
        )

        axes[row, 1].imshow(anomaly_mask, cmap="gray", vmin=0, vmax=1)
        if target_mask.any() and not target_mask.all():
            axes[row, 1].contour(target_mask, levels=[0.5], colors="cyan", linewidths=1)
        axes[row, 1].set_title("Ground truth\nwhite=hole, cyan=wheel")

        heatmap_artist = axes[row, 2].imshow(
            anomaly_map, cmap=colormap, vmin=0, vmax=1
        )
        if target_mask.any() and not target_mask.all():
            axes[row, 2].contour(target_mask, levels=[0.5], colors="cyan", linewidths=1)
        axes[row, 2].set_title("Anomaly map\nnormalized score [0, 1]")

        axes[row, 3].imshow(image)
        axes[row, 3].imshow(
            anomaly_map,
            cmap=colormap,
            vmin=0,
            vmax=1,
            alpha=overlay_alpha,
        )
        if anomaly_mask.any() and not anomaly_mask.all():
            axes[row, 3].contour(
                anomaly_mask, levels=[0.5], colors="lime", linewidths=1.5
            )
        axes[row, 3].set_title("Overlay\ngreen=ground-truth boundary")

        for axis in axes[row]:
            axis.axis("off")

    figure.suptitle(title, fontsize=15)
    figure.colorbar(
        heatmap_artist,
        ax=axes[:, 2:].ravel().tolist(),
        label="Normalized anomaly score",
        shrink=0.8,
    )
    return figure


def save_anomaly_visualizations(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    device: torch.device | str,
    preprocessor: WheelPreprocessor,
    title: str,
    num_clean: int = 3,
    num_anomalous: int = 3,
    colormap: str = "magma",
    overlay_alpha: float = 0.55,
    dpi: int = 150,
) -> Path:
    """Collect representative predictions and save one comparison figure."""
    if dpi < 1:
        raise ValueError("dpi must be at least 1")
    samples = collect_anomaly_visualization_samples(
        detector,
        loader,
        device=device,
        preprocessor=preprocessor,
        num_clean=num_clean,
        num_anomalous=num_anomalous,
    )
    figure = plot_anomaly_visualizations(
        samples,
        title=title,
        colormap=colormap,
        overlay_alpha=overlay_alpha,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    figure.clear()
    return output_path
