from __future__ import annotations

import json
from pathlib import Path

import torch

from src.quantization.core import require_module, save_json, sha256


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, build_mode: str, cfg):
    """Build or reuse a TensorRT engine whose metadata matches the ONNX input."""

    trt = require_module("tensorrt")
    model_hash = sha256(onnx_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = engine_path.with_suffix(engine_path.suffix + ".json")
    cache_hit = False
    if engine_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_hit = (
            metadata.get("onnx_sha256") == model_hash
            and metadata.get("build_mode") == build_mode
            and metadata.get("tensorrt_version") == trt.__version__
            and metadata.get("builder_optimization_level")
            == int(cfg.tensorrt.builder_optimization_level)
        )
    if not cache_hit:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        if build_mode == "strongly_typed":
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        elif build_mode in {"weakly_typed_explicit_qdq", "weakly_typed_fp16"}:
            flags = 0
        else:
            raise ValueError(f"Unsupported TensorRT build mode: {build_mode}")
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise RuntimeError({"stage": "onnx_parse", "errors": errors})
        build_config = builder.create_builder_config()
        if build_mode in {"weakly_typed_explicit_qdq", "weakly_typed_fp16"}:
            build_config.set_flag(trt.BuilderFlag.FP16)
        build_config.builder_optimization_level = int(
            cfg.tensorrt.builder_optimization_level
        )
        build_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(cfg.tensorrt.workspace_gib) * 2**30,
        )
        serialized = builder.build_serialized_network(network, build_config)
        if serialized is None:
            raise RuntimeError(f"TensorRT did not build an engine in mode {build_mode}")
        engine_path.write_bytes(bytes(serialized))
        save_json(
            {
                "onnx_sha256": model_hash,
                "build_mode": build_mode,
                "tensorrt_version": trt.__version__,
                "builder_optimization_level": int(
                    cfg.tensorrt.builder_optimization_level
                ),
            },
            metadata_path,
        )
    return {
        "path": str(engine_path),
        "sha256": sha256(engine_path),
        "size_mib": engine_path.stat().st_size / 2**20,
        "onnx_sha256": model_hash,
        "cache_hit": cache_hit,
        "build_mode": build_mode,
        "strongly_typed": build_mode == "strongly_typed",
        "tensorrt_version": trt.__version__,
        "fp16_builder_flag": build_mode
        in {"weakly_typed_explicit_qdq", "weakly_typed_fp16"},
        "builder_optimization_level": int(cfg.tensorrt.builder_optimization_level),
    }


class TensorRTRunner:
    """Minimal torch-backed runner for the fixed images/logits TensorRT contract."""

    def __init__(self, engine_path: Path, device_id: int):
        trt = require_module("tensorrt")
        self.trt = trt
        self.device_id = int(device_id)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Unable to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device_id)
        names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
        ]
        inputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        outputs = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if inputs != ["images"] or outputs != ["logits"]:
            raise RuntimeError({"engine_inputs": inputs, "engine_outputs": outputs})
        self.input_name, self.output_name = inputs[0], outputs[0]
        type_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        self.input_dtype = type_map[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtype = type_map[self.engine.get_tensor_dtype(self.output_name)]
        self.output = None

    @torch.inference_mode()
    def run(self, images):
        with torch.cuda.stream(self.stream):
            inputs = images.to(
                device=f"cuda:{self.device_id}",
                dtype=self.input_dtype,
                non_blocking=True,
            ).contiguous()
            expected = tuple(self.engine.get_tensor_shape(self.input_name))
            if -1 in expected:
                if not self.context.set_input_shape(self.input_name, tuple(inputs.shape)):
                    raise RuntimeError(f"Invalid TensorRT shape: {tuple(inputs.shape)}")
            elif tuple(inputs.shape) != expected:
                raise ValueError({"expected": expected, "received": tuple(inputs.shape)})
            output_shape = tuple(self.context.get_tensor_shape(self.output_name))
            if any(dimension < 0 for dimension in output_shape):
                raise RuntimeError(f"Unresolved TensorRT output shape: {output_shape}")
            if self.output is None or tuple(self.output.shape) != output_shape:
                self.output = torch.empty(
                    output_shape,
                    device=f"cuda:{self.device_id}",
                    dtype=self.output_dtype,
                )
            self.context.set_tensor_address(self.input_name, inputs.data_ptr())
            self.context.set_tensor_address(self.output_name, self.output.data_ptr())
            if not self.context.execute_async_v3(
                stream_handle=self.stream.cuda_stream
            ):
                raise RuntimeError("TensorRT execution failed")
        self.stream.synchronize()
        return self.output

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
