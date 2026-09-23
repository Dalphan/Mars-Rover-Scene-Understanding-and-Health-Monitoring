from __future__ import annotations

from src.quantization.core import (
    compare_onnx_sessions,
    compare_pytorch_and_onnx,
    create_onnx_session,
    evaluate_onnx,
    evaluate_torch,
    export_onnx_fp32,
    measure_onnx_latency,
    measure_pytorch_latency,
    resolve_ort_providers,
)
from src.quantization.ptq_fp16 import convert_fp16
from src.quantization.ptq_reporting import persist_results


def run_precision_stages(
    cfg,
    paths,
    model,
    device,
    results,
    evaluation_loader,
    latency_sample,
):
    if cfg.steps.run_fp32_baseline:
        results["benchmarks"]["pytorch_fp32_cuda"] = {
            "runtime": "pytorch",
            "precision": "fp32",
            "device": str(device),
            "artifact_key": "pytorch_fp32",
            "metrics": evaluate_torch(
                model,
                evaluation_loader,
                device,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_pytorch_latency(
                model,
                latency_sample,
                device,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
        }
        persist_results(results, cfg, paths)

    if cfg.steps.run_onnx_export:
        info = export_onnx_fp32(
            model,
            paths.fp32,
            cfg.dataset.image_size,
            cfg.dataset.num_classes,
            cfg.onnx.opset_version,
        )
        results["artifacts"]["onnx_fp32"] = {
            "format": "onnx",
            "precision": "fp32",
            **info,
        }
        persist_results(results, cfg, paths)
    elif not paths.fp32.is_file():
        raise FileNotFoundError(f"Existing FP32 ONNX not found: {paths.fp32}")

    providers = resolve_ort_providers(
        bool(cfg.runtime.prefer_gpu), bool(cfg.runtime.prefer_tensorrt)
    )
    fp32_session = create_onnx_session(paths.fp32, providers)
    if cfg.steps.run_onnx_fp32_benchmark:
        results["benchmarks"]["onnx_fp32_cuda"] = {
            "runtime": "onnxruntime",
            "precision": "fp32",
            "device": fp32_session.get_providers()[0],
            "providers": fp32_session.get_providers(),
            "artifact_key": "onnx_fp32",
            "numerical_parity_with_pytorch": compare_pytorch_and_onnx(
                model, fp32_session, latency_sample, device
            ),
            "metrics": evaluate_onnx(
                fp32_session,
                evaluation_loader,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_onnx_latency(
                fp32_session,
                latency_sample,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
        }
        persist_results(results, cfg, paths)

    fp16_session = None
    if cfg.steps.run_onnx_fp16:
        info = convert_fp16(paths.fp32, paths.fp16, cfg.onnx.fp16_keep_io_types)
        results["artifacts"]["onnx_fp16"] = {
            "format": "onnx",
            "precision": "fp16",
            **info,
        }
        fp16_session = create_onnx_session(paths.fp16, providers)
        results["benchmarks"]["onnx_fp16_cuda"] = {
            "runtime": "onnxruntime",
            "precision": "fp16",
            "device": fp16_session.get_providers()[0],
            "providers": fp16_session.get_providers(),
            "artifact_key": "onnx_fp16",
            "numerical_parity_with_onnx_fp32": compare_onnx_sessions(
                fp32_session, fp16_session, latency_sample
            ),
            "metrics": evaluate_onnx(
                fp16_session,
                evaluation_loader,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_onnx_latency(
                fp16_session,
                latency_sample,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
        }
        persist_results(results, cfg, paths)
    return providers, fp32_session
