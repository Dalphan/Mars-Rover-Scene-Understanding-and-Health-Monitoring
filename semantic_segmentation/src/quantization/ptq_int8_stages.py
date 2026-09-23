from __future__ import annotations

from src.quantization.core import (
    build_dataset,
    build_onnx_runner,
    compare_onnx_sessions,
    evaluate_onnx,
    measure_onnx_latency,
    measure_runner_efficiency,
    sha256,
)
from src.quantization.ptq_calibration import (
    CalibrationReader,
    create_calibration_manifest,
    verify_calibration_reader,
)
from src.quantization.ptq_config import operator_scope
from src.quantization.ptq_int8 import create_int8
from src.quantization.ptq_reporting import persist_results
from src.quantization.ptq_tensorrt import create_trt_session, probe_trt_int8


def run_int8_stages(
    cfg,
    paths,
    results,
    fp32_session,
    evaluation_loader,
    latency_sample,
):
    calibration_dataset = calibration_reader = calibration_manifest = None
    if cfg.steps.run_int8_calibration_preparation:
        calibration_dataset = build_dataset(
            cfg.dataset,
            cfg.dataset.calibration_split,
            cfg.data.max_calibration_samples,
            cfg.data.calibration_selection,
            cfg.seed,
        )
        calibration_reader = CalibrationReader(
            calibration_dataset, fp32_session.get_inputs()[0].name
        )
        verification = verify_calibration_reader(calibration_reader, fp32_session, cfg)
        calibration_manifest = create_calibration_manifest(
            calibration_dataset, calibration_reader, verification, cfg, paths.manifest
        )
        results["artifacts"]["int8_calibration_manifest"] = {
            "format": "json",
            "path": str(paths.manifest),
            "size_mib": paths.manifest.stat().st_size / 2**20,
            "sha256": sha256(paths.manifest),
        }
        results["calibration"]["int8_preparation"] = calibration_manifest
        persist_results(results, cfg, paths)

    if cfg.steps.run_onnx_int8_quantization:
        if calibration_reader is None:
            raise RuntimeError(
                "run_onnx_int8_quantization requires calibration preparation"
            )
        info = create_int8(paths.fp32, paths.int8, calibration_reader, cfg)
        info["calibration_subset_fingerprint"] = calibration_manifest[
            "subset_fingerprint"
        ]
        int8_key = (
            f"onnx_int8_{str(cfg.int8.calibration_method).lower()}_"
            f"{cfg.data.calibration_selection}_{operator_scope(cfg.int8.op_types)}"
        )
        results["artifacts"][int8_key] = {
            "format": "onnx",
            "precision": "int8",
            **info,
        }
        persist_results(results, cfg, paths)
    elif (cfg.steps.run_trt_int8_probe or cfg.steps.run_onnx_int8_benchmark) and not paths.int8.is_file():
        raise FileNotFoundError(f"Existing INT8 ONNX not found: {paths.int8}")

    if cfg.steps.run_trt_int8_probe:
        results["calibration"]["tensorrt_int8_probe"] = probe_trt_int8(
            cfg, paths.int8, fp32_session, latency_sample
        )
        persist_results(results, cfg, paths)

    int8_session = None
    if cfg.steps.run_onnx_int8_benchmark:
        int8_session, _ = create_trt_session(cfg, paths.int8)
        run_once, _ = build_onnx_runner(int8_session, latency_sample)
        int8_benchmark = {
            "runtime": "onnxruntime_tensorrt_ep",
            "precision": "int8_qdq_with_fp16_fallback",
            "device": int8_session.get_providers()[0],
            "providers": int8_session.get_providers(),
            "numerical_parity_with_onnx_fp32": compare_onnx_sessions(
                fp32_session, int8_session, latency_sample
            ),
            "metrics": evaluate_onnx(
                int8_session,
                evaluation_loader,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_onnx_latency(
                int8_session,
                latency_sample,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
            "efficiency": measure_runner_efficiency(
                run_once,
                cfg.benchmark.efficiency_warmup_runs,
                cfg.benchmark.efficiency_min_runs,
                cfg.benchmark.efficiency_min_seconds,
                cfg.tensorrt.device_id,
                cfg.benchmark.nvml_sample_interval_seconds,
            ),
        }
        results["benchmarks"]["onnx_int8_tensorrt"] = int8_benchmark
        persist_results(results, cfg, paths)
    return int8_session
