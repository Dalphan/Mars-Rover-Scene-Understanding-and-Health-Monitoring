from __future__ import annotations

import logging
import math
import platform
import time
from typing import Any

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def count_trainable_parameters(model: nn.Module) -> dict[str, int]:
    """Return parameter counts while preserving the existing result keys."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def analyze_model(
    model: nn.Module,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    warmup_iterations: int = 50,
    measurement_iterations: int = 100,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Profile one eager FP32 model without permanently changing its state."""
    logger = logger or logging.getLogger(__name__)
    if input_shape[0] != 1:
        raise ValueError("Model analysis currently defines FPS for batch size 1")
    if warmup_iterations < 0 or measurement_iterations <= 0:
        raise ValueError("warmup_iterations must be >= 0 and measurement_iterations > 0")

    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("Cannot analyze a model without parameters") from error
    device = parameter.device
    if parameter.dtype != torch.float32:
        raise ValueError(f"Model analysis requires FP32 parameters, got {parameter.dtype}")

    was_training = model.training
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    use_cuda = device.type == "cuda"
    sample = torch.randn(input_shape, dtype=torch.float32, device=device)
    timings_ms: list[float] = []

    try:
        model.eval()
        torch.backends.cudnn.benchmark = use_cuda
        with torch.inference_mode():
            flop_analysis = FlopCountAnalysis(model, sample)
            total_flops = float(flop_analysis.total())
            unsupported_ops = dict(flop_analysis.unsupported_ops())
            if unsupported_ops:
                logger.warning("Unsupported FLOP operators: %s", unsupported_ops)

            for _ in range(warmup_iterations):
                model(sample)

            if use_cuda:
                torch.cuda.synchronize(device)
                events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
                for _ in range(measurement_iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    model(sample)
                    end.record()
                    events.append((start, end))
                torch.cuda.synchronize(device)
                timings_ms = [start.elapsed_time(end) for start, end in events]
            else:
                for _ in range(measurement_iterations):
                    start_ns = time.perf_counter_ns()
                    model(sample)
                    timings_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
    finally:
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        model.train(was_training)

    counts = count_trainable_parameters(model)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    mean_latency_ms = sum(timings_ms) / len(timings_ms)
    variance = sum((value - mean_latency_ms) ** 2 for value in timings_ms) / len(timings_ms)
    latency_std_ms = math.sqrt(variance)
    fps = 1000.0 / mean_latency_ms
    gflops = total_flops / 1e9
    gpu_name = torch.cuda.get_device_name(device) if use_cuda else None
    results: dict[str, Any] = {
        **counts,
        "model_size_mb": parameter_bytes / (1024**2),
        "flops_per_inference": total_flops,
        "gflops_per_inference": gflops,
        "mean_latency_ms": mean_latency_ms,
        "latency_std_ms": latency_std_ms,
        "fps": fps,
        "effective_gflops_per_second": gflops * fps,
        "unsupported_flop_operators": unsupported_ops,
        "input_shape": list(input_shape),
        "batch_size": input_shape[0],
        "input_resolution": list(input_shape[-2:]),
        "device_type": device.type,
        "device_name": gpu_name or platform.processor() or "CPU",
        "dtype": str(sample.dtype),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "execution_mode": "eager",
        "cudnn_benchmark": use_cuda,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
    }
    logger.info(
        "Model analysis\n"
        "  Model: %s\n  Input shape: %s\n  Device: %s (%s)\n  Dtype: %s\n"
        "  PyTorch: %s  CUDA: %s\n  Execution mode: eager\n"
        "  Total parameters: %d\n  Trainable parameters: %d\n"
        "  Parameter size: %.3f MB\n  FLOPs/inference: %.4f GFLOPs\n"
        "  Mean latency: %.3f ms\n  Latency std: %.3f ms\n  FPS: %.3f\n"
        "  Effective throughput: %.3f GFLOP/s\n  Warmup iterations: %d\n"
        "  Measurement iterations: %d\n  cuDNN benchmark: %s\n"
        "  Unsupported FLOP operators: %s",
        getattr(model, "variant", model.__class__.__name__),
        list(input_shape),
        results["device_name"],
        device.type,
        sample.dtype,
        torch.__version__,
        torch.version.cuda,
        counts["total"],
        counts["trainable"],
        results["model_size_mb"],
        gflops,
        mean_latency_ms,
        latency_std_ms,
        fps,
        results["effective_gflops_per_second"],
        warmup_iterations,
        measurement_iterations,
        use_cuda,
        unsupported_ops or "none",
    )
    return results
