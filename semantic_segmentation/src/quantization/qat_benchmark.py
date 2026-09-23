from __future__ import annotations

import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.metrics.segmentation_metrics import (
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.quantization.core import measure_runner_efficiency, quantization_metrics


@torch.inference_mode()
def evaluate_tensorrt(runner, loader, cfg, split):
    confmat = create_confusion_matrix(int(cfg.dataset.num_classes))
    for batch in tqdm(loader, desc=f"TensorRT {split}"):
        predictions = runner.run(batch["image"]).argmax(1).cpu()
        update_confusion_matrix(
            confmat,
            predictions,
            batch["mask"],
            int(cfg.dataset.num_classes),
            int(cfg.dataset.ignore_index),
        )
    metrics = quantization_metrics(
        confmat,
        cfg.dataset.num_classes,
        cfg.dataset.ignore_index,
        cfg.dataset.class_names,
    )
    metrics.update(split=str(split), samples=len(loader.dataset))
    return metrics


def measure_trt_latency(runner, images, cfg):
    if images.shape[0] != 1:
        raise ValueError("TensorRT latency requires batch size 1")
    images = images.to(
        device=f"cuda:{cfg.tensorrt.device_id}", dtype=runner.input_dtype
    )
    for _ in range(int(cfg.benchmark.latency_warmup_runs)):
        runner.run(images)
    values = []
    for _ in range(int(cfg.benchmark.latency_measured_runs)):
        started = time.perf_counter()
        runner.run(images)
        values.append((time.perf_counter() - started) * 1000.0)
    mean_ms = float(np.mean(values))
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "images_per_second": 1000.0 / mean_ms,
    }


def benchmark_tensorrt(runner, loader, cfg, split, engine_info, precision, parity):
    images = next(iter(loader))["image"]
    return {
        "runtime": "tensorrt",
        "execution": engine_info["build_mode"],
        "precision": precision,
        "mixed_provider_execution": False,
        "parity": parity,
        "metrics": evaluate_tensorrt(runner, loader, cfg, split),
        "latency": measure_trt_latency(runner, images, cfg),
        "efficiency": measure_runner_efficiency(
            lambda: runner.run(images),
            cfg.benchmark.efficiency_warmup_runs,
            cfg.benchmark.efficiency_min_runs,
            cfg.benchmark.efficiency_min_seconds,
            cfg.tensorrt.device_id,
            cfg.benchmark.nvml_sample_interval_seconds,
        ),
        "engine": engine_info,
    }


def benchmark_snapshot(benchmark):
    return {
        "validation_miou": benchmark["metrics"]["miou"],
        "mean_latency_ms": benchmark["latency"]["mean_ms"],
        "p95_latency_ms": benchmark["latency"]["p95_ms"],
        "average_sampled_power_w": benchmark["efficiency"][
            "average_sampled_power_w"
        ],
        "energy_per_image_mj": benchmark["efficiency"]["energy_per_image_mj"],
        "images_per_joule": benchmark["efficiency"]["images_per_joule"],
        "peak_gpu_memory_mib": benchmark["efficiency"]["peak_gpu_memory_mib"],
        "engine_size_mib": benchmark["engine"]["size_mib"],
    }


def reduction_percent(current, reference):
    return float(100.0 * (reference - current) / reference)


def compare_with_conv_only(int8_benchmark, cfg):
    current = benchmark_snapshot(int8_benchmark)
    reference = OmegaConf.to_container(
        cfg.reference.segformer_qat_conv_only, resolve=True
    )
    comparison = {
        "protocol": "same validation split and measurement protocol",
        "baseline_variant": "int8_conv_only_qdq",
        "current_variant": str(cfg.model.variant),
        "baseline": reference,
        "current": current,
        "validation_miou_delta_current_minus_conv_only": float(
            current["validation_miou"] - reference["validation_miou"]
        ),
        "mean_latency_reduction_percent": reduction_percent(
            current["mean_latency_ms"], reference["mean_latency_ms"]
        ),
        "p95_latency_reduction_percent": reduction_percent(
            current["p95_latency_ms"], reference["p95_latency_ms"]
        ),
        "average_power_reduction_percent": reduction_percent(
            current["average_sampled_power_w"],
            reference["average_sampled_power_w"],
        ),
        "energy_per_image_reduction_percent": reduction_percent(
            current["energy_per_image_mj"], reference["energy_per_image_mj"]
        ),
        "images_per_joule_increase_percent": float(
            100.0
            * (current["images_per_joule"] / reference["images_per_joule"] - 1.0)
        ),
        "peak_gpu_memory_reduction_percent": reduction_percent(
            current["peak_gpu_memory_mib"], reference["peak_gpu_memory_mib"]
        ),
        "engine_size_reduction_percent": reduction_percent(
            current["engine_size_mib"], reference["engine_size_mib"]
        ),
    }
    comparison["validation_accuracy_guardrail"] = bool(
        current["validation_miou"]
        >= reference["validation_miou"]
        - float(cfg.reference.max_validation_miou_drop_vs_conv_only)
    )
    comparison["energy_improved"] = bool(
        current["energy_per_image_mj"] < reference["energy_per_image_mj"]
    )
    comparison["selected_on_validation"] = bool(
        comparison["validation_accuracy_guardrail"]
        and comparison["energy_improved"]
    )
    return comparison
