from __future__ import annotations

import pytest

from src.quantization.benchmark import compute_speedup, summarize_timings
from src.quantization.calibration import (
    select_calibration_indices,
    select_class_coverage_indices,
    select_nested_calibration_indices,
)
from src.quantization.config import PTQConfig, QATConfig


def test_nested_calibration_subsets_are_deterministic_and_nested():
    subsets = select_nested_calibration_indices(100, [32, 8, 16], seed=42)
    repeated = select_nested_calibration_indices(100, [8, 16, 32], seed=42)
    assert subsets == repeated
    assert subsets[8] == subsets[16][:8]
    assert subsets[16] == subsets[32][:16]
    assert len(set(subsets[32])) == 32
    assert select_calibration_indices(100, 16, seed=42) == subsets[16]


def test_class_coverage_sampler_covers_labels_before_filling():
    labels = [{0}, {1}, {2}, {0, 1}, {1, 2}, {0, 2}]
    selected = select_class_coverage_indices(labels, 3, seed=7)
    covered = set().union(*(labels[index] for index in selected))
    assert covered == {0, 1, 2}
    assert selected == select_class_coverage_indices(labels, 3, seed=7)


def test_timing_summary_and_speedup():
    summary = summarize_timings([1.0, 2.0, 3.0, 4.0])
    assert summary["mean_latency_ms"] == pytest.approx(2.5)
    assert summary["median_latency_ms"] == pytest.approx(2.5)
    assert summary["p95_latency_ms"] == pytest.approx(3.85)
    assert summary["fps_from_mean"] == pytest.approx(400.0)
    assert compute_speedup(10.0, 4.0) == pytest.approx(2.5)


def test_configs_validate_modes_and_required_fields():
    PTQConfig().validate()
    QATConfig().validate()
    with pytest.raises(ValueError, match="benchmark_batch_size=1"):
        PTQConfig(benchmark_batch_size=2).validate()
    with pytest.raises(ValueError, match="source_checkpoint"):
        QATConfig(mode="qat_int8").validate()
    QATConfig(mode="qat_int8", source_checkpoint="best.ckpt").validate()
