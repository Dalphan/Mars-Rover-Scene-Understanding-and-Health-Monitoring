from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import fmean, median


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_timings(timings_ms: Iterable[float]) -> dict[str, float | int]:
    """Return stable latency statistics for strictly positive millisecond timings."""
    values = [float(value) for value in timings_ms]
    if not values:
        raise ValueError("timings_ms must contain at least one value")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timings_ms must contain only finite, positive values")

    ordered = sorted(values)
    mean_ms = fmean(values)
    variance = fmean((value - mean_ms) ** 2 for value in values)
    median_ms = median(values)
    return {
        "num_measurements": len(values),
        "min_latency_ms": ordered[0],
        "max_latency_ms": ordered[-1],
        "mean_latency_ms": mean_ms,
        "latency_std_ms": math.sqrt(variance),
        "median_latency_ms": median_ms,
        "p50_latency_ms": _percentile(ordered, 50.0),
        "p90_latency_ms": _percentile(ordered, 90.0),
        "p95_latency_ms": _percentile(ordered, 95.0),
        "p99_latency_ms": _percentile(ordered, 99.0),
        "fps_from_mean": 1000.0 / mean_ms,
        "fps_from_median": 1000.0 / median_ms,
    }


def compute_speedup(baseline_latency_ms: float, candidate_latency_ms: float) -> float:
    baseline = float(baseline_latency_ms)
    candidate = float(candidate_latency_ms)
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("latencies must be finite")
    if baseline <= 0.0 or candidate <= 0.0:
        raise ValueError("latencies must be positive")
    return baseline / candidate
