"""Public TensorRT API assembled from focused engine and benchmark modules."""

from src.quantization.qat_benchmark import (
    benchmark_snapshot,
    benchmark_tensorrt,
    compare_with_conv_only,
    evaluate_tensorrt,
    measure_trt_latency,
    reduction_percent,
)
from src.quantization.qat_engine import TensorRTRunner, build_tensorrt_engine

__all__ = [
    "TensorRTRunner",
    "benchmark_snapshot",
    "benchmark_tensorrt",
    "build_tensorrt_engine",
    "compare_with_conv_only",
    "evaluate_tensorrt",
    "measure_trt_latency",
    "reduction_percent",
]
