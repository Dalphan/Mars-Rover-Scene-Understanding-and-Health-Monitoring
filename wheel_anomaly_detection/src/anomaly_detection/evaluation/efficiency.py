"""Batch-one inference measurements, independent of task accuracy metrics."""

from __future__ import annotations

import math
import platform
import sys
import threading
import time
from typing import Any

import torch


def _latency_summary(milliseconds: list[float]) -> dict[str, float]:
    if not milliseconds:
        raise ValueError("Latency samples cannot be empty")
    ordered = sorted(float(value) for value in milliseconds)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "minimum_ms": ordered[0],
        "mean_ms": sum(ordered) / len(ordered),
        "median_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "maximum_ms": ordered[-1],
    }


def _cuda_memory_baseline(device: torch.device) -> tuple[int, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    baseline = (
        torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    )
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _cuda_memory_report(
    device: torch.device, baseline: tuple[int, int] | None,
) -> dict[str, float] | None:
    if baseline is None:
        return None
    allocated, reserved = baseline
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "baseline_allocated_mib": allocated / 2**20,
        "baseline_reserved_mib": reserved / 2**20,
        "peak_allocated_mib": peak_allocated / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
        "incremental_peak_allocated_mib": (peak_allocated - allocated) / 2**20,
        "incremental_peak_reserved_mib": (peak_reserved - reserved) / 2**20,
    }


@torch.inference_mode()
def benchmark_anomaly_inference(
    detector: Any,
    loader: Any,
    *,
    device: torch.device | str,
    warmup_runs: int = 10,
    timed_runs: int = 50,
    split: str = "validation",
) -> dict[str, Any]:
    """Measure predict-only and sequential batch-one pipeline latency.

    ``predict_only`` reuses one preprocessed, device-resident input. The other
    measurements include the loader's complete work, which for the synthetic
    dataset includes annotation I/O and aligned mask transforms. They are not
    a mask-free deployment benchmark. CUDA allocator peaks include all live
    tensors on the selected device, not just the model, and exclude driver
    allocations. No power sensor or energy estimate is used.
    """
    if getattr(loader, "batch_size", None) != 1:
        raise ValueError("Inference benchmark loader must use batch_size=1")
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Sequential inference benchmark requires num_workers=0")
    if not isinstance(warmup_runs, int) or not isinstance(timed_runs, int):
        raise TypeError("warmup_runs and timed_runs must be integers")
    if warmup_runs < 1 or timed_runs < 1:
        raise ValueError("warmup_runs and timed_runs must be positive")
    if not detector.is_fitted:
        raise RuntimeError("Fit or restore the detector before benchmarking")

    device = torch.device(device)
    uses_cuda = device.type == "cuda"
    was_training = detector.training
    detector.to(device)
    detector.eval()

    def synchronize() -> None:
        if uses_cuda:
            torch.cuda.synchronize(device)

    def next_batch(iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            try:
                return next(iterator), iterator
            except StopIteration as error:
                raise ValueError("Inference benchmark loader is empty") from error

    try:
        iterator = iter(loader)
        batch, iterator = next_batch(iterator)
        images = batch["image"].to(device, non_blocking=uses_cuda)
        if images.ndim != 4 or images.shape[0] != 1:
            raise ValueError("Inference benchmark requires one BCHW image")
        input_shape, input_dtype = list(images.shape), str(images.dtype)
        for _ in range(warmup_runs):
            prediction = detector.predict(images)
            del prediction
        synchronize()
        baseline = _cuda_memory_baseline(device)
        predict_ms = []
        for _ in range(timed_runs):
            synchronize()
            start = time.perf_counter()
            prediction = detector.predict(images)
            synchronize()
            predict_ms.append((time.perf_counter() - start) * 1_000)
            # Do not retain the previous output across the next forward.
            del prediction
        predict_memory = _cuda_memory_report(device, baseline)
        del images, batch, iterator

        iterator = iter(loader)
        for _ in range(warmup_runs):
            batch, iterator = next_batch(iterator)
            images = batch["image"].to(device, non_blocking=uses_cuda)
            prediction = detector.predict(images)
            synchronize()
            del prediction, images, batch
        baseline = _cuda_memory_baseline(device)
        preprocessing_ms, accelerator_ms, pipeline_ms = [], [], []
        iterator = iter(loader)
        for _ in range(timed_runs):
            synchronize()
            start = time.perf_counter()
            batch, iterator = next_batch(iterator)
            preprocessing_end = time.perf_counter()
            images = batch["image"].to(device, non_blocking=uses_cuda)
            if list(images.shape) != input_shape or str(images.dtype) != input_dtype:
                raise ValueError("Benchmark inputs must have one stable shape and dtype")
            prediction = detector.predict(images)
            synchronize()
            end = time.perf_counter()
            preprocessing_ms.append((preprocessing_end - start) * 1_000)
            accelerator_ms.append((end - preprocessing_end) * 1_000)
            pipeline_ms.append((end - start) * 1_000)
            del prediction, images, batch
        pipeline_memory = _cuda_memory_report(device, baseline)

        predict_summary = _latency_summary(predict_ms)
        accelerator_summary = _latency_summary(accelerator_ms)
        pipeline_summary = _latency_summary(pipeline_ms)
        parameters = list(detector.parameters())
        buffers = list(detector.buffers())
        bank = getattr(detector, "memory_bank", None)
        return {
            "schema_version": 1,
            "scope": (
                "predict_only: one device-resident preprocessed image; pipeline: "
                "sequential loader (including annotation I/O and mask transforms), "
                "host-to-device transfer and detector.predict; no worker prefetch"
            ),
            "memory_scope": (
                "CUDA allocator totals include the model, memory bank and other live "
                "tensors on this device; driver allocations are excluded"
            ),
            "split": split,
            "batch_size": 1,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if uses_cuda else None,
            "input_shape": input_shape,
            "input_dtype": input_dtype,
            "model": type(detector).__name__,
            "gaussian_sigma": getattr(detector, "gaussian_sigma", None),
            "total_parameters": sum(parameter.numel() for parameter in parameters),
            "trainable_parameters": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
            "parameters_mib": sum(
                parameter.numel() * parameter.element_size() for parameter in parameters
            ) / 2**20,
            "buffers_mib": sum(
                buffer.numel() * buffer.element_size() for buffer in buffers
            ) / 2**20,
            "memory_bank_shape": list(bank.shape) if bank is not None else None,
            "memory_bank_mib": (
                bank.numel() * bank.element_size() / 2**20 if bank is not None else None
            ),
            "latency": {
                "predict_only": predict_summary,
                "decode_and_preprocess": _latency_summary(preprocessing_ms),
                "host_to_device_and_predict": accelerator_summary,
                "end_to_end": pipeline_summary,
            },
            "throughput_images_per_second": 1_000 / accelerator_summary["mean_ms"],
            "predict_only_images_per_second": 1_000 / predict_summary["mean_ms"],
            "end_to_end_images_per_second": 1_000 / pipeline_summary["mean_ms"],
            "cuda_memory": pipeline_memory,
            "predict_only_cuda_memory": predict_memory,
            "environment": {
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
            "energy_joules_per_image": None,
            "energy_note": "Not measured; latency alone is not an energy measurement",
        }
    finally:
        detector.train(was_training)


def _nvml_handle_for_cuda(nvml: Any, device: torch.device) -> tuple[Any, str]:
    """Match by CUDA UUID; never assume CUDA and NVML ordinals are identical."""
    properties = torch.cuda.get_device_properties(device)
    cuda_uuid = getattr(properties, "uuid", None)
    if cuda_uuid:
        uuid_text = cuda_uuid.decode("ascii") if isinstance(cuda_uuid, bytes) else str(cuda_uuid)
        if not uuid_text.startswith(("GPU-", "MIG-")):
            uuid_text = "GPU-" + uuid_text
        try:
            return nvml.nvmlDeviceGetHandleByUUID(uuid_text), "cuda_uuid"
        except Exception as error:
            raise RuntimeError(f"Cannot match the CUDA UUID to NVML: {error}") from error
    if nvml.nvmlDeviceGetCount() == 1:
        return nvml.nvmlDeviceGetHandleByIndex(0), "single_nvml_device"
    raise RuntimeError(
        "CUDA UUID is unavailable and multiple NVML GPUs exist; "
        "refusing to measure an ambiguous physical device"
    )


def _nvml_counter_energy_window(
    predict_once: Any, nvml: Any, handle: Any, duration_seconds: float,
) -> dict[str, Any]:
    """Counter units are mJ; inference failures propagate rather than being hidden."""
    method = "nvml_total_energy_counter"
    try:
        before = int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
        if before < 0:
            raise ValueError("negative energy counter")
    except Exception as error:
        return {"status": "unavailable", "method": method, "reason": str(error)}
    start = time.perf_counter()
    count = 0
    while count == 0 or time.perf_counter() - start < duration_seconds:
        predict_once()
        count += 1
    try:
        after = int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception as error:
        return {"status": "unavailable", "method": method, "reason": str(error)}
    elapsed = time.perf_counter() - start
    if after <= before:
        return {
            "status": "unavailable", "method": method,
            "reason": "Energy counter did not advance or was reset during measurement",
        }
    joules = (after - before) / 1_000.0
    return {
        "status": "measured", "method": method,
        "inferences": count, "duration_seconds": elapsed,
        "gpu_energy_joules": joules, "energy_joules_per_image": joules / count,
        "average_gpu_power_watts": joules / elapsed,
        "counter_start_millijoules": before,
        "counter_end_millijoules": after,
        "power_samples": [],
    }


def _nvml_power_energy_window(
    predict_once: Any, nvml: Any, handle: Any,
    duration_seconds: float, sample_interval_seconds: float,
) -> dict[str, Any]:
    """Integrate measured GPU power, with explicit boundary samples and no TDP estimate."""
    method = "nvml_power_integration"

    def read_watts() -> float:
        watts = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1_000.0
        if not math.isfinite(watts) or watts <= 0:
            raise ValueError("NVML power reading must be positive and finite")
        return watts

    try:
        first_watts = read_watts()
    except Exception as error:
        return {"status": "unavailable", "method": method, "reason": str(error)}
    start = time.perf_counter()
    samples = [(start, first_watts)]
    errors = []
    stop = threading.Event()

    def sample_power() -> None:
        while not stop.wait(sample_interval_seconds):
            try:
                watts = read_watts()
                samples.append((time.perf_counter(), watts))
            except Exception as error:
                errors.append(str(error))
                return

    worker = threading.Thread(target=sample_power, daemon=True)
    worker.start()
    count = 0
    try:
        while count == 0 or time.perf_counter() - start < duration_seconds:
            predict_once()
            count += 1
            if errors:
                break
    finally:
        stop.set()
        worker.join()
    if errors:
        return {"status": "unavailable", "method": method, "reason": errors[0]}
    try:
        final_watts = read_watts()
    except Exception as error:
        return {"status": "unavailable", "method": method, "reason": str(error)}
    samples.append((time.perf_counter(), final_watts))
    elapsed = samples[-1][0] - start
    # Two endpoints alone are not adequate for the sampled-power fallback.
    if len(samples) < 3:
        return {
            "status": "unavailable", "method": method,
            "reason": "Too few power samples; increase the measurement window",
        }
    joules = sum(
        (right[0] - left[0]) * (left[1] + right[1]) * 0.5
        for left, right in zip(samples, samples[1:])
    )
    return {
        "status": "measured", "method": method,
        "inferences": count, "duration_seconds": elapsed,
        "gpu_energy_joules": joules, "energy_joules_per_image": joules / count,
        "average_gpu_power_watts": joules / elapsed,
        "power_sample_interval_seconds_requested": sample_interval_seconds,
        "power_sample_count": len(samples),
        "power_samples": [
            {"elapsed_seconds": timestamp - start, "power_watts": watts}
            for timestamp, watts in samples
        ],
    }


@torch.inference_mode()
def measure_anomaly_inference_energy(
    detector: Any, loader: Any, *, device: torch.device | str,
    repeats: int = 3, duration_seconds: float = 30.0,
    warmup_seconds: float = 5.0, sample_interval_seconds: float = 0.1,
    method: str = "auto",
) -> dict[str, Any]:
    """Measure device-wide GPU J/image in a separate sustained predict-only phase.

    The model and one preprocessed input are already GPU resident. Each complete
    predict is synchronized, and its output discarded. Includes GPU idle/boundary
    energy, excludes CPU, I/O and host-to-device energy. Raw total energy is not
    idle-subtracted and is not attributed to an individual process. Sensor errors
    yield an explicit unavailable/partial report; prediction errors propagate.
    """
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("Energy repeats must be a positive integer")
    for name, value in (
        ("duration_seconds", duration_seconds),
        ("sample_interval_seconds", sample_interval_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be non-negative and finite")
    if method not in {"auto", "counter", "power"}:
        raise ValueError("Energy method must be auto, counter or power")
    if getattr(loader, "batch_size", None) != 1 or getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Energy requires batch_size=1 and num_workers=0")
    if not detector.is_fitted:
        raise RuntimeError("Restore the detector before measuring energy")
    device = torch.device(device)
    report = {
        "schema_version": 1,
        "status": "unavailable", "reason": None, "method": None,
        "metric": "device_wide_gpu_energy_per_inference",
        "scope": (
            "predict-only repeated synchronized inference; model and one preprocessed "
            "image are CUDA resident; excludes input transfer, CPU and loader work"
        ),
        "device": str(device), "batch_size": 1,
        "repeats_requested": repeats,
        "duration_seconds_per_repeat_requested": duration_seconds,
        "warmup_seconds_requested": warmup_seconds,
        "sample_interval_seconds_requested": sample_interval_seconds,
        "method_requested": method,
        "idle_subtracted": False, "process_attributed": False,
        "energy_joules_per_image": None,
        "average_gpu_power_watts": None,
        "energy_joules_per_image_std": None,
        "total_inferences": 0, "runs": [], "warnings": [],
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        report["reason"] = "CUDA GPU is required for NVML energy measurements"
        return report
    try:
        import pynvml
    except ImportError:
        report["reason"] = "Install nvidia-ml-py (import pynvml) to read GPU energy/power"
        return report
    try:
        pynvml.nvmlInit()
    except Exception as error:
        report["reason"] = f"NVML initialization failed: {error}"
        return report

    was_training = detector.training
    try:
        try:
            handle, mapping = _nvml_handle_for_cuda(pynvml, device)
            report["device_mapping"] = mapping
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            report["nvml_gpu_uuid"] = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
            name = pynvml.nvmlDeviceGetName(handle)
            report["gpu_name"] = name.decode() if isinstance(name, bytes) else str(name)
        except Exception as error:
            report["reason"] = f"NVML device mapping failed: {error}"
            return report
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            report["gpu_compute_process_count_at_start"] = len(processes)
            if len(processes) > 1:
                report["warnings"].append(
                    "Multiple compute processes are visible on this GPU; "
                    "device-wide energy cannot be assigned exclusively to this model"
                )
        except Exception:
            report["gpu_compute_process_count_at_start"] = None
        try:
            report["power_limit_watts"] = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1_000.0
        except Exception:
            report["power_limit_watts"] = None
        detector.to(device).eval()
        try:
            batch = next(iter(loader))
        except StopIteration as error:
            raise ValueError("Energy measurement loader is empty") from error
        images = batch["image"].to(device)
        del batch
        if images.ndim != 4 or images.shape[0] != 1 or images.dtype != torch.float32:
            raise ValueError("Energy comparison requires one FP32 BCHW image")
        report["input_shape"] = list(images.shape)
        report["input_dtype"] = str(images.dtype)

        def predict_once() -> None:
            prediction = detector.predict(images)
            torch.cuda.synchronize(device)
            del prediction

        torch.cuda.synchronize(device)
        warmup_start = time.perf_counter()
        predict_once()
        while time.perf_counter() - warmup_start < warmup_seconds:
            predict_once()
        report["warmup_seconds_actual"] = time.perf_counter() - warmup_start
        chosen_method = "power" if method == "power" else "counter"
        for repeat in range(repeats):
            if chosen_method == "counter":
                run = _nvml_counter_energy_window(
                    predict_once, pynvml, handle, duration_seconds
                )
                # Auto chooses one successful backend on the first repeat only;
                # never average counter and integrated-power repeats together.
                if repeat == 0 and method == "auto" and run["status"] != "measured":
                    report["warnings"].append("Counter unavailable: " + str(run.get("reason")))
                    chosen_method = "power"
                    run = _nvml_power_energy_window(
                        predict_once, pynvml, handle,
                        duration_seconds, sample_interval_seconds,
                    )
            else:
                run = _nvml_power_energy_window(
                    predict_once, pynvml, handle,
                    duration_seconds, sample_interval_seconds,
                )
            run["repeat"] = repeat + 1
            report["runs"].append(run)
            if run["status"] != "measured":
                report["status"] = "partial" if repeat else "unavailable"
                report["reason"] = run.get("reason", "GPU energy sensor failed")
                return report
        runs = report["runs"]
        total_images = sum(run["inferences"] for run in runs)
        total_joules = sum(run["gpu_energy_joules"] for run in runs)
        total_seconds = sum(run["duration_seconds"] for run in runs)
        values = [run["energy_joules_per_image"] for run in runs]
        mean = sum(values) / len(values)
        report.update({
            "status": "measured", "reason": None,
            "method": runs[0]["method"],
            "total_inferences": total_images,
            "total_gpu_energy_joules": total_joules,
            "total_duration_seconds": total_seconds,
            "energy_joules_per_image": total_joules / total_images,
            "average_gpu_power_watts": total_joules / total_seconds,
            "energy_joules_per_image_std": (
                math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
                if len(values) > 1 else 0.0
            ),
        })
        return report
    finally:
        detector.train(was_training)
        pynvml.nvmlShutdown()
