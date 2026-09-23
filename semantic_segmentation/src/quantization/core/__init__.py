"""Shared, modular building blocks for PTQ and QAT pipelines."""

from src.quantization.core.benchmark import (
    NvmlMonitor,
    build_onnx_runner,
    measure_onnx_latency,
    measure_pytorch_latency,
    measure_runner_efficiency,
)
from src.quantization.core.data import (
    QuantizationTransform,
    SegmentationQuantizationDataset,
    build_dataloader,
    build_dataset,
    segmentation_collate_fn,
    select_dataset_subset,
)
from src.quantization.core.dependencies import require_module, save_json, set_seed, sha256
from src.quantization.core.metrics import evaluate_torch, quantization_metrics
from src.quantization.core.models import (
    build_inference_model,
    ensure_checkpoint,
    load_trained_model,
    validate_model_checkpoint,
)
from src.quantization.core.onnx import (
    GPU_ORT_PROVIDERS,
    compare_onnx_sessions,
    compare_pytorch_and_onnx,
    create_onnx_session,
    evaluate_onnx,
    export_onnx_fp32,
    prepare_onnx_input,
    resolve_ort_providers,
)

__all__ = [
    "GPU_ORT_PROVIDERS",
    "NvmlMonitor",
    "QuantizationTransform",
    "SegmentationQuantizationDataset",
    "build_dataloader",
    "build_dataset",
    "build_inference_model",
    "build_onnx_runner",
    "compare_onnx_sessions",
    "compare_pytorch_and_onnx",
    "create_onnx_session",
    "ensure_checkpoint",
    "evaluate_onnx",
    "evaluate_torch",
    "export_onnx_fp32",
    "load_trained_model",
    "measure_onnx_latency",
    "measure_pytorch_latency",
    "measure_runner_efficiency",
    "prepare_onnx_input",
    "quantization_metrics",
    "require_module",
    "resolve_ort_providers",
    "save_json",
    "segmentation_collate_fn",
    "select_dataset_subset",
    "set_seed",
    "sha256",
    "validate_model_checkpoint",
]
