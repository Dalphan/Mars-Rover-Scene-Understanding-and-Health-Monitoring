"""Evaluation utilities for visual anomaly detection."""

from .diagnostics import (
    DEFAULT_GROUP_FIELDS,
    AnomalyDiagnostics,
    PerRegionOverlap,
    save_anomaly_diagnostics,
)
from .metrics import (
    ESSENTIAL_METRIC_NAMES,
    AnomalyMetrics,
    BinaryHistogramMetrics,
    ExactBinaryMetrics,
    build_metrics,
    update_metrics_from_batch,
)
from .runner import evaluate_anomaly_detector
from .visualization import (
    collect_anomaly_visualization_samples,
    plot_anomaly_visualizations,
    save_anomaly_visualizations,
)

__all__ = [
    "ESSENTIAL_METRIC_NAMES",
    "DEFAULT_GROUP_FIELDS",
    "AnomalyDiagnostics",
    "AnomalyMetrics",
    "BinaryHistogramMetrics",
    "ExactBinaryMetrics",
    "PerRegionOverlap",
    "build_metrics",
    "collect_anomaly_visualization_samples",
    "evaluate_anomaly_detector",
    "plot_anomaly_visualizations",
    "save_anomaly_diagnostics",
    "save_anomaly_visualizations",
    "update_metrics_from_batch",
]
