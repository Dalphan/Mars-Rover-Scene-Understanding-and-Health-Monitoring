from __future__ import annotations

import gc

import torch

from src.quantization.qat_export import export_qat_onnx
from src.quantization.qat_parity import parity
from src.quantization.qat_tensorrt import (
    TensorRTRunner,
    benchmark_snapshot,
    benchmark_tensorrt,
    build_tensorrt_engine,
    compare_with_conv_only,
    reduction_percent,
)


def _validate_engine_parity(
    engine_path, sample, reference_output, cfg, label: str
):
    runner = TensorRTRunner(engine_path, cfg.tensorrt.device_id)
    candidate_output = runner.run(sample).float().cpu().numpy()
    comparison = parity(reference_output, candidate_output)
    if comparison["prediction_agreement"] < float(
        cfg.export.trt_min_prediction_agreement
    ):
        raise RuntimeError(f"TensorRT {label} parity failed: {comparison}")
    del runner
    return comparison


def _benchmark_fp16(experiment, cfg, engine_info, fp16_trt_parity):
    if not cfg.steps.run_trt_fp16_baseline:
        return None
    runner = TensorRTRunner(experiment.paths.engine_fp16, cfg.tensorrt.device_id)
    benchmark = benchmark_tensorrt(
        runner,
        experiment.val_loader,
        cfg,
        cfg.dataset.val_split,
        engine_info,
        "fp16_same_qat_trained_weights",
        fp16_trt_parity,
    )
    experiment.results["artifacts"]["tensorrt_engine_fp16"] = engine_info
    experiment.results["benchmarks"][
        "onnx_modelopt_qat_weights_fp16_tensorrt"
    ] = benchmark
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return benchmark


def _record_int8_fp16_comparison(results, int8_benchmark, fp16_benchmark):
    if fp16_benchmark is None:
        return
    int8_snapshot = benchmark_snapshot(int8_benchmark)
    fp16_snapshot = benchmark_snapshot(fp16_benchmark)
    results["comparisons"]["int8_vs_fp16_tensorrt"] = {
        "protocol": "same post-QAT weights, direct TensorRT, batch 1",
        "fp16": fp16_snapshot,
        "int8": int8_snapshot,
        "validation_miou_delta_int8_minus_fp16": (
            int8_snapshot["validation_miou"] - fp16_snapshot["validation_miou"]
        ),
        "mean_latency_reduction_percent": reduction_percent(
            int8_snapshot["mean_latency_ms"], fp16_snapshot["mean_latency_ms"]
        ),
        "energy_per_image_reduction_percent": reduction_percent(
            int8_snapshot["energy_per_image_mj"],
            fp16_snapshot["energy_per_image_mj"],
        ),
    }


def run_qat_deployment(
    experiment, tensor_quantizer_type, quantizer_audit, cfg, device
):
    """Export ONNX, build TensorRT engines and benchmark validation data."""

    paths = experiment.paths
    results = experiment.results
    int8_artifact, fp16_artifact, sample, fallback_output, fp16_output = (
        export_qat_onnx(
            experiment.student,
            experiment.val_loader,
            tensor_quantizer_type,
            quantizer_audit,
            cfg,
            paths,
            device,
        )
    )
    results["artifacts"]["onnx_qat_int8"] = int8_artifact
    if fp16_artifact:
        results["artifacts"]["onnx_qat_weights_fp16"] = fp16_artifact

    int8_engine = build_tensorrt_engine(
        paths.onnx_int8, paths.engine_int8, str(cfg.tensorrt.build_mode), cfg
    )
    fp16_engine = None
    if cfg.steps.run_trt_fp16_baseline:
        fp16_engine = build_tensorrt_engine(
            paths.onnx_fp16,
            paths.engine_fp16,
            str(cfg.tensorrt.fp16_build_mode),
            cfg,
        )

    trt_parity = _validate_engine_parity(
        paths.engine_int8, sample, fallback_output, cfg, "INT8"
    )
    fp16_trt_parity = None
    if cfg.steps.run_trt_fp16_baseline:
        fp16_trt_parity = _validate_engine_parity(
            paths.engine_fp16, sample, fp16_output, cfg, "FP16"
        )

    experiment.teacher.cpu()
    experiment.student.cpu()
    del sample
    gc.collect()
    torch.cuda.empty_cache()

    fp16_benchmark = _benchmark_fp16(
        experiment, cfg, fp16_engine, fp16_trt_parity
    )
    trt_runner = TensorRTRunner(paths.engine_int8, cfg.tensorrt.device_id)
    int8_benchmark = benchmark_tensorrt(
        trt_runner,
        experiment.val_loader,
        cfg,
        cfg.dataset.val_split,
        int8_engine,
        "int8_qdq_with_fp16_fallback",
        trt_parity,
    )
    results["artifacts"]["tensorrt_engine_int8"] = int8_engine
    results["benchmarks"]["onnx_modelopt_qat_int8_tensorrt"] = int8_benchmark
    results["comparisons"]["vs_segformer_qat_conv_only"] = compare_with_conv_only(
        int8_benchmark, cfg
    )
    _record_int8_fp16_comparison(results, int8_benchmark, fp16_benchmark)
    return trt_runner
