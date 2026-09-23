from __future__ import annotations

import threading
import time

import numpy as np
import torch

from src.quantization.core.dependencies import require_module
from src.quantization.core.onnx import GPU_ORT_PROVIDERS, prepare_onnx_input

@torch.inference_mode()
def measure_pytorch_latency(model, sample, device, warmup_runs, measured_runs):
    if sample.shape[0] != 1:
        raise ValueError("Latency benchmark requires batch size 1")
    sample = sample.to(device, non_blocking=True)
    model.eval()
    for _ in range(int(warmup_runs)):
        model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latencies = []
    for _ in range(int(measured_runs)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    mean_ms = float(np.mean(latencies))
    return {
        "device": str(device),
        "batch_size": 1,
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        "measurement_scope": "forward with input already on device",
        "mean_ms": mean_ms,
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "images_per_second": 1000.0 / mean_ms,
    }


def build_onnx_runner(session, sample):
    ort = require_module("onnxruntime")
    sample_numpy = prepare_onnx_input(session, sample)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    primary = session.get_providers()[0]
    if primary in GPU_ORT_PROVIDERS:
        binding = session.io_binding()
        value = ort.OrtValue.ortvalue_from_numpy(sample_numpy, "cuda", 0)
        binding.bind_ortvalue_input(input_name, value)
        binding.bind_output(output_name, "cuda", 0)

        def run_once():
            session.run_with_iobinding(binding)
            binding.synchronize_outputs()

        scope = "forward with input already on device (I/O Binding)"
    else:
        def run_once():
            session.run([output_name], {input_name: sample_numpy})

        scope = "session.run on CPU"
    return run_once, scope


def measure_onnx_latency(session, sample, warmup_runs, measured_runs):
    if sample.shape[0] != 1:
        raise ValueError("Latency benchmark requires batch size 1")
    run_once, scope = build_onnx_runner(session, sample)
    for _ in range(int(warmup_runs)):
        run_once()
    latencies = []
    for _ in range(int(measured_runs)):
        start = time.perf_counter()
        run_once()
        latencies.append((time.perf_counter() - start) * 1000.0)
    mean_ms = float(np.mean(latencies))
    return {
        "provider": session.get_providers()[0],
        "provider_chain": session.get_providers(),
        "batch_size": 1,
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        "measurement_scope": scope,
        "mean_ms": mean_ms,
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "images_per_second": 1000.0 / mean_ms,
    }


class NvmlMonitor:
    def __init__(self, device_id: int, interval_seconds: float) -> None:
        self.device_id = int(device_id)
        self.interval_seconds = float(interval_seconds)
        self.samples: list[tuple[float, float, float]] = []
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        pynvml = require_module("pynvml")
        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)

        def sample_loop():
            while not self._stop.is_set():
                timestamp = time.perf_counter()
                power_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                memory_mib = (
                    pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
                )
                self.samples.append((timestamp, power_w, memory_mib))
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._pynvml.nvmlShutdown()

    def summary(self, elapsed_seconds: float, runs: int) -> dict:
        if not self.samples:
            return {"available": False}
        powers = np.asarray([sample[1] for sample in self.samples])
        memories = np.asarray([sample[2] for sample in self.samples])
        energy_j = float(powers.mean() * elapsed_seconds)
        return {
            "available": True,
            "average_sampled_power_w": float(powers.mean()),
            "peak_gpu_memory_mib": float(memories.max()),
            "elapsed_seconds": float(elapsed_seconds),
            "runs": int(runs),
            "energy_j": energy_j,
            "energy_per_image_mj": energy_j * 1000.0 / max(runs, 1),
            "images_per_joule": runs / energy_j if energy_j > 0 else None,
        }


def measure_runner_efficiency(
    run_once,
    warmup_runs: int,
    min_runs: int,
    min_seconds: float,
    device_id: int,
    sample_interval_seconds: float,
):
    for _ in range(int(warmup_runs)):
        run_once()
    runs = 0
    start = time.perf_counter()
    with NvmlMonitor(device_id, sample_interval_seconds) as monitor:
        while runs < int(min_runs) or time.perf_counter() - start < float(min_seconds):
            run_once()
            runs += 1
    elapsed = time.perf_counter() - start
    return monitor.summary(elapsed, runs)
