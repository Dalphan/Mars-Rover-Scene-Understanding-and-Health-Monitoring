"""Resource benchmark mirrored from the authoritative PTQ notebook."""

from __future__ import annotations

import gc
import os
import threading
import time

import numpy as np
import torch

from src.quantization.core import (
    GPU_ORT_PROVIDERS,
    create_onnx_session,
    measure_onnx_latency,
    prepare_onnx_input,
    require_module,
    resolve_ort_providers,
)
from src.quantization.ptq_common import persist_results


def build_onnx_gpu_runner(session, sample, device_id=0):
    """Prepara un forward con input/output residenti sulla GPU."""
    if sample.shape[0] != 1:
        raise ValueError("Il benchmark di efficienza richiede batch size 1")
    primary_provider = session.get_providers()[0]
    if primary_provider not in GPU_ORT_PROVIDERS:
        raise RuntimeError(
            f"Il benchmark NVML richiede un provider GPU, trovato {primary_provider}"
        )

    ort = require_module("onnxruntime")
    sample_numpy = prepare_onnx_input(session, sample)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    io_binding = session.io_binding()
    input_ortvalue = ort.OrtValue.ortvalue_from_numpy(
        sample_numpy, "cuda", device_id
    )
    io_binding.bind_ortvalue_input(input_name, input_ortvalue)
    io_binding.bind_output(output_name, "cuda", device_id)

    def run_once():
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

    return run_once


class NvmlEfficiencyMonitor:
    """Misura energia device e picco memoria durante un loop sostenuto."""

    def __init__(self, device_id, sample_interval_seconds):
        self.pynvml = require_module("pynvml")
        self.device_id = int(device_id)
        self.sample_interval_seconds = float(sample_interval_seconds)
        if self.sample_interval_seconds <= 0:
            raise ValueError("NVML_SAMPLE_INTERVAL_SECONDS deve essere positivo")
        self.process_id = os.getpid()
        self.stop_event = threading.Event()
        self.samples = []
        self.sampling_error = None

    def _read_memory_mib(self):
        device_memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        try:
            processes = self.pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
            for process in processes:
                if process.pid == self.process_id:
                    used = process.usedGpuMemory
                    # Alcune versioni di nvidia-ml-py non esportano una
                    # costante sentinella uniforme. Un valore reale deve
                    # comunque stare tra zero e la VRAM fisica totale.
                    if (
                        isinstance(used, (int, np.integer))
                        and 0 <= int(used) <= int(device_memory.total)
                    ):
                        return int(used) / (1024 ** 2), "process"
        except (AttributeError, self.pynvml.NVMLError):
            pass
        return device_memory.used / (1024 ** 2), "device_total_fallback"

    def _sample_once(self):
        timestamp = time.perf_counter()
        power_w = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        memory_mib, memory_scope = self._read_memory_mib()
        self.samples.append((timestamp, power_w, memory_mib, memory_scope))

    def _sample_loop(self):
        try:
            while not self.stop_event.wait(self.sample_interval_seconds):
                self._sample_once()
        except Exception as error:
            self.sampling_error = repr(error)
            self.stop_event.set()

    def start(self):
        self.pynvml.nvmlInit()
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        self.samples = []
        self.sampling_error = None
        self.stop_event.clear()
        try:
            self.energy_start_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(
                self.handle
            )
            self.total_energy_counter_available = True
        except (AttributeError, self.pynvml.NVMLError):
            self.energy_start_mj = None
            self.total_energy_counter_available = False
        self._sample_once()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self):
        energy_end_mj = None
        if self.total_energy_counter_available:
            energy_end_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(
                self.handle
            )
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, 5 * self.sample_interval_seconds))
        self._sample_once()
        if self.sampling_error is not None:
            raise RuntimeError(f"Campionamento NVML fallito: {self.sampling_error}")
        if not self.samples:
            raise RuntimeError("NVML non ha prodotto campioni")

        if energy_end_mj is not None:
            total_energy_j = (energy_end_mj - self.energy_start_mj) / 1000.0
            energy_source = "nvml_total_energy_counter"
        else:
            total_energy_j = sum(
                0.5 * (left[1] + right[1]) * (right[0] - left[0])
                for left, right in zip(self.samples, self.samples[1:])
            )
            energy_source = "integrated_nvml_power_samples"
        if total_energy_j <= 0:
            raise RuntimeError(f"Energia NVML non valida: {total_energy_j}")

        return {
            "total_energy_j": float(total_energy_j),
            "energy_source": energy_source,
            "energy_scope": "whole GPU device, idle not subtracted",
            "average_sampled_power_w": float(
                np.mean([sample[1] for sample in self.samples])
            ),
            "peak_gpu_memory_mib": float(
                max(sample[2] for sample in self.samples)
            ),
            "gpu_memory_scope": (
                "process"
                if all(sample[3] == "process" for sample in self.samples)
                else "device_total_fallback"
            ),
            "nvml_samples": len(self.samples),
            "nvml_sample_interval_seconds": self.sample_interval_seconds,
        }


def measure_onnx_efficiency(session, sample, benchmark_cfg, device_id=0):
    """Restituisce energia per immagine e memoria GPU di picco."""
    min_seconds = float(benchmark_cfg.efficiency_min_seconds)
    min_runs = int(benchmark_cfg.efficiency_min_runs)
    if min_seconds <= 0 or min_runs <= 0:
        raise ValueError("Durata e numero minimo di run devono essere positivi")

    run_once = build_onnx_gpu_runner(session, sample, device_id)
    for _ in range(int(benchmark_cfg.efficiency_warmup_runs)):
        run_once()

    monitor = NvmlEfficiencyMonitor(
        device_id, benchmark_cfg.nvml_sample_interval_seconds
    )
    monitor.start()
    measured_runs = 0
    start = time.perf_counter()
    try:
        while measured_runs < min_runs or time.perf_counter() - start < min_seconds:
            run_once()
            measured_runs += 1
        elapsed_seconds = time.perf_counter() - start
    finally:
        resource_metrics = monitor.stop()

    energy_per_image_mj = (
        resource_metrics["total_energy_j"] * 1000.0 / measured_runs
    )
    return {
        **resource_metrics,
        "elapsed_seconds": float(elapsed_seconds),
        "runs": measured_runs,
        "warmup_runs": int(benchmark_cfg.efficiency_warmup_runs),
        "minimum_runs": min_runs,
        "minimum_seconds": min_seconds,
        "energy_per_image_mj": float(energy_per_image_mj),
        "images_per_joule": float(1000.0 / energy_per_image_mj),
    }


def run_baseline_efficiency(cfg, paths, results, latency_sample):
    """Measure each enabled baseline alone, after the caller releases GPU models."""
    variants = []
    if cfg.steps.run_onnx_fp32_benchmark:
        variants.append(("onnx_fp32_cuda", paths.fp32))
    if cfg.steps.run_onnx_fp16:
        variants.append(("onnx_fp16_cuda", paths.fp16))
    if not variants:
        return

    providers = resolve_ort_providers(
        bool(cfg.runtime.prefer_gpu), bool(cfg.runtime.prefer_tensorrt)
    )
    for benchmark_key, model_path in variants:
        if benchmark_key not in results["benchmarks"]:
            raise RuntimeError(f"Missing baseline benchmark: {benchmark_key}")
        session = create_onnx_session(model_path, providers)
        try:
            benchmark = results["benchmarks"][benchmark_key]
            benchmark["latency"] = measure_onnx_latency(
                session,
                latency_sample,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            )
            benchmark["efficiency"] = measure_onnx_efficiency(
                session, latency_sample, cfg.benchmark, cfg.tensorrt.device_id
            )
            benchmark["efficiency"]["session_residency"] = (
                "one ONNX session; PyTorch model on CPU"
            )
            persist_results(results, cfg, paths)
        finally:
            del session
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
