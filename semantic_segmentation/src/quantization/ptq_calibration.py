from __future__ import annotations

from src.quantization.ptq_common import operator_scope, persist_results
from src.quantization.ptq_int8 import create_int8
from src.quantization.ptq_runtime import create_trt_session, probe_trt_int8

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.quantization.core import save_json

class CalibrationReader:
    def __init__(self, dataset, input_name: str) -> None:
        self.dataset = dataset
        self.input_name = str(input_name)
        self.start_index = 0
        self.end_index = len(dataset)
        self.next_index = 0

    def __len__(self):
        return len(self.dataset)

    def set_range(self, start_index, end_index):
        start_index, end_index = int(start_index), int(end_index)
        if not 0 <= start_index < end_index <= len(self.dataset):
            raise ValueError(f"Invalid calibration range {start_index}:{end_index}")
        self.start_index, self.end_index = start_index, end_index
        self.next_index = start_index

    def get_next(self):
        if self.next_index >= self.end_index:
            return None
        sample = self.dataset[self.next_index]
        self.next_index += 1
        images = np.ascontiguousarray(
            sample["image"].unsqueeze(0).numpy(), dtype=np.float32
        )
        return {self.input_name: images}

    def rewind(self):
        self.start_index = 0
        self.end_index = len(self.dataset)
        self.next_index = 0


def verify_calibration_reader(reader, session, cfg: DictConfig) -> dict:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_input = (1, 3, *tuple(cfg.dataset.image_size))
    expected_output = (
        1,
        int(cfg.dataset.num_classes),
        *tuple(cfg.dataset.image_size),
    )
    sample_count = inference_runs = 0
    observed_min, observed_max = float("inf"), float("-inf")
    reader.rewind()
    while True:
        model_inputs = reader.get_next()
        if model_inputs is None:
            break
        images = model_inputs[input_name]
        if images.dtype != np.float32 or tuple(images.shape) != expected_input:
            raise ValueError(
                f"Invalid calibration tensor dtype/shape: {images.dtype}, {images.shape}"
            )
        if not np.isfinite(images).all():
            raise ValueError("Calibration input contains NaN or infinity")
        sample_count += 1
        observed_min = min(observed_min, float(images.min()))
        observed_max = max(observed_max, float(images.max()))
        if inference_runs < int(cfg.data.calibration_dry_run_samples):
            logits = session.run([output_name], model_inputs)[0]
            if tuple(logits.shape) != expected_output or not np.isfinite(logits).all():
                raise ValueError("Calibration dry-run returned invalid logits")
            inference_runs += 1
    if sample_count != len(reader):
        raise ValueError(f"Calibration reader yielded {sample_count}/{len(reader)} samples")
    reader.rewind()
    return {
        "status": "passed",
        "samples": sample_count,
        "input_name": input_name,
        "input_dtype": "float32",
        "input_shape": list(expected_input),
        "normalized_value_min": observed_min,
        "normalized_value_max": observed_max,
        "dry_run_samples": inference_runs,
        "dry_run_output_shape": list(expected_output),
        "all_inputs_finite": True,
        "all_dry_run_outputs_finite": True,
        "rewind_ok": True,
    }


def class_presence(dataset, class_names) -> tuple[dict, list, list]:
    names = {int(key): str(value) for key, value in class_names.items()}
    name_to_id = {name: class_id for class_id, name in names.items()}
    counts = {name: 0 for name in names.values()}
    unknown = set()
    for labels in dataset.dataset["class_labels"]:
        ids = set()
        for value in labels:
            if isinstance(value, str) and value in name_to_id:
                ids.add(name_to_id[value])
                continue
            try:
                class_id = int(value)
            except (TypeError, ValueError):
                unknown.add(str(value))
                continue
            if class_id in names:
                ids.add(class_id)
            else:
                unknown.add(str(value))
        for class_id in ids:
            counts[names[class_id]] += 1
    return counts, [name for name, count in counts.items() if count == 0], sorted(unknown)


def create_calibration_manifest(dataset, reader, verification, cfg, path: Path):
    counts, missing, unknown = class_presence(dataset, cfg.dataset.class_names)
    selected = dataset.selected_source_indices
    unique = len(set(selected)) == len(selected)
    in_range = all(0 <= index < dataset.source_sample_count for index in selected)
    if not unique or not in_range:
        raise ValueError("Calibration source indices are invalid")
    manifest = {
        "dataset_repo": str(cfg.dataset.repo_id),
        "split": str(cfg.dataset.calibration_split),
        "source_dataset_fingerprint": dataset.source_fingerprint,
        "source_sample_count": dataset.source_sample_count,
        "subset_fingerprint": dataset.subset_fingerprint,
        "selection": str(cfg.data.calibration_selection),
        "seed": int(cfg.seed),
        "sample_count": len(dataset),
        "batch_size": 1,
        "input_name": reader.input_name,
        "input_dtype": "float32",
        "input_shape": [1, 3, *list(cfg.dataset.image_size)],
        "normalization_mean": list(cfg.dataset.image_mean),
        "normalization_std": list(cfg.dataset.image_std),
        "masks_used_for_calibration": False,
        "class_presence_unit": "images, not pixels",
        "class_presence_counts": counts,
        "missing_classes": missing,
        "unknown_class_labels": unknown,
        "selected_indices_unique": unique,
        "selected_indices_in_source_range": in_range,
        "selected_source_indices": selected,
        "verification": verification,
    }
    save_json(manifest, path)
    return manifest

from src.quantization.core import (
    build_dataset,
    build_onnx_runner,
    compare_onnx_sessions,
    evaluate_onnx,
    measure_onnx_latency,
    measure_runner_efficiency,
    sha256,
)


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
