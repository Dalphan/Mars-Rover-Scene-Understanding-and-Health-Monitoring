from __future__ import annotations

import pytest

from src.quantization.benchmark import compute_speedup, summarize_timings
from src.quantization.calibration import (
    select_calibration_indices,
    select_class_coverage_indices,
    select_nested_calibration_indices,
)
from src.quantization.config import (
    PTQConfig,
    QATConfig,
    build_qat_quantize_config,
    get_qat_quantizer_exclusions,
)


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


def test_qat_exclusions_target_only_known_t4_incompatible_model_stages():
    assert get_qat_quantizer_exclusions("smp", "deeplabv3", "resnet34") == (
        "*encoder.layer3*",
        "*encoder.layer4*",
    )
    expected = (
        "*encoder.features.14*",
        "*encoder.features.15*",
        "*encoder.features.16*",
        "*encoder.features.17*",
        "*encoder.features.18*",
    )
    assert (
        get_qat_quantizer_exclusions("smp", "deeplabv3plus", "mobilenet_v2")
        == expected
    )
    assert get_qat_quantizer_exclusions("smp", "unet", "mobilenet_v2") == ()
    assert get_qat_quantizer_exclusions("smp", "deeplabv3plus", "resnet34") == ()
    assert get_qat_quantizer_exclusions("smp", "deeplabv3", "mobilenet_v2") == ()
    assert (
        get_qat_quantizer_exclusions(
            "segformer_b0", "deeplabv3plus", "mobilenet_v2"
        )
        == ()
    )


@pytest.mark.parametrize(
    "base_config",
    [
        {"quant_cfg": [{"quantizer_name": "*", "enable": True}]},
        {"quant_cfg": {"*": {"enable": True}}},
    ],
)
def test_qat_config_builder_copies_config_and_appends_selected_exclusion(base_config):
    original = repr(base_config)
    config, exclusions = build_qat_quantize_config(
        base_config,
        model_name="smp",
        smp_architecture="deeplabv3plus",
        smp_encoder_name="mobilenet_v2",
    )
    assert exclusions == (
        "*encoder.features.14*",
        "*encoder.features.15*",
        "*encoder.features.16*",
        "*encoder.features.17*",
        "*encoder.features.18*",
    )
    assert repr(base_config) == original
    assert config != base_config


def test_qat_config_builder_rejects_unknown_modelopt_quant_cfg_shape():
    with pytest.raises(TypeError, match="quant_cfg"):
        build_qat_quantize_config(
            {"quant_cfg": None},
            model_name="smp",
            smp_architecture="deeplabv3plus",
            smp_encoder_name="mobilenet_v2",
        )
