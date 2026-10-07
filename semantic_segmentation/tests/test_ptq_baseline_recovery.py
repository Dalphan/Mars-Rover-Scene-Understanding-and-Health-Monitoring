import ast
import copy
import gc
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from scripts.check_notebook_config_sync import notebook_literals
from src.quantization import ptq_efficiency, ptq_pipeline


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks/kaggle_s5mars_onnx_ptq.ipynb"


def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def cell_source(index):
    return "".join(notebook()["cells"][index]["source"])


def notebook_helpers():
    tree = ast.parse(cell_source(30))
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {
        "np": np, "os": os,
        "threading": __import__("threading"),
        "time": __import__("time"),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<helpers>", "exec"),
         namespace)
    return namespace


def recovery_cfg():
    return OmegaConf.load(ROOT / "configs/quantization/ptq.yaml")


def test_recovery_defaults_and_notebook_syntax():
    constants = notebook_literals(NOTEBOOK_PATH)
    assert constants["MODEL_NAME"] == "smp"
    assert constants["SMP_ARCHITECTURE"] == "deeplabv3"
    assert constants["SMP_ENCODER_NAME"] == "resnet34"
    assert constants["CHECKPOINT_FILENAME"] == "best.ckpt"
    assert constants["CHECKPOINT_EXPERIMENT"] == "smp_deeplabv3_resnet34"
    assert constants["TARGET_PRECISIONS"] == ("fp32", "fp16")
    for flag in (
        "RUN_SMOKE_TEST", "RUN_ONNX_EXPORT", "RUN_ONNX_FP32_BENCHMARK",
        "RUN_ONNX_FP16",
    ):
        assert constants[flag] is True
    for flag in (
        "RUN_FP32_BASELINE", "RUN_INT8_CALIBRATION_PREPARATION",
        "RUN_ONNX_INT8_QUANTIZATION", "RUN_TRT_INT8_PROBE",
        "RUN_ONNX_INT8_BENCHMARK", "RUN_FINAL_TEST_EVALUATION",
    ):
        assert constants[flag] is False
    assert constants["BATCH_SIZE"] == 1
    assert constants["IMAGE_SIZE"] == (512, 512)
    assert constants["LATENCY_WARMUP_RUNS"] == 10
    assert constants["LATENCY_MEASURED_RUNS"] == 50
    assert constants["EFFICIENCY_MIN_RUNS"] == 100
    assert constants["EFFICIENCY_MIN_SECONDS"] == 30.0
    assert constants["NVML_SAMPLE_INTERVAL_SECONDS"] == 0.02
    assert constants["ONNX_OUTPUT_DIR"].endswith("/deeplabv3_baselines_recovery")
    for index, cell in enumerate(notebook()["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
            continue
        ast.parse(source, filename=f"PTQ cell {index}")


def test_smoke_without_calibration_uses_one_validation_image(tmp_path):
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    batch = {"image": torch.zeros(1, 3, 4, 4),
             "mask": torch.zeros(1, 4, 4, dtype=torch.long)}
    calls = []

    class Model:
        def to(self, device):
            return self

        def __call__(self, images):
            return torch.zeros(images.shape[0], 9, *images.shape[-2:])

    def build_dataset(split, max_samples=None, selection="random"):
        calls.append((split, max_samples, selection))
        return [batch]

    namespace = {
        "RUN_SMOKE_TEST": True, "RUN_INT8_CALIBRATION_PREPARATION": False,
        "EVALUATION_SPLIT": "val", "TORCH_DEVICE": torch.device("cpu"),
        "NUM_CLASSES": 9, "CHECKPOINT_FILENAME": "best.ckpt",
        "CHECKPOINT_EXPERIMENT": "smp_deeplabv3_resnet34",
        "BENCHMARK_RESULTS": {"artifacts": {}},
        "download_checkpoint_from_drive": lambda: checkpoint,
        "load_trained_model": lambda path: Model(),
        "build_dataset": build_dataset, "build_dataloader": lambda dataset: dataset,
        "save_benchmark_results": lambda: None,
        "torch": torch, "Path": Path,
    }
    exec(compile(cell_source(14), "<smoke>", "exec"), namespace)
    assert calls == [("val", 1, "random")]
    assert "calibration_dataset" not in namespace
    assert "images" not in namespace and "logits" not in namespace
    assert namespace["BENCHMARK_RESULTS"]["artifacts"]["pytorch_fp32"][
        "checkpoint_filename"] == "best.ckpt"


@pytest.mark.parametrize("fp32,fp16", [(True, True), (True, False), (False, True)])
def test_notebook_baseline_efficiency_is_independent_of_int8(fp32, fp16):
    live = weakref.WeakSet()
    seen = []
    saved = []

    class Session:
        def __init__(self, path):
            self.path = path
            live.add(self)

    class Model:
        device = "cuda"

        def to(self, device):
            self.device = device
            return self

    model = Model()
    results = {"benchmarks": {
        "onnx_fp32_cuda": {"metrics": {"miou": 0.7379}},
        "onnx_fp16_cuda": {"metrics": {"miou": 0.7379}},
    }}

    def create_session(path):
        assert not live, "Another ONNX session is still resident"
        return Session(path)

    def efficiency(session, sample):
        assert len(live) == 1
        assert model.device == "cpu"
        seen.append(session.path)
        return {"energy_per_image_mj": 400.0, "peak_gpu_memory_mib": 800.0}

    namespace = {
        "RUN_ONNX_INT8_BENCHMARK": False,
        "RUN_ONNX_FP32_BENCHMARK": fp32, "RUN_ONNX_FP16": fp16,
        "BENCHMARK_RESULTS": results, "model": model, "latency_sample": object(),
        "onnx_fp32_path": "fp32.onnx", "onnx_fp16_path": "fp16.onnx",
        "create_onnx_session": create_session,
        "measure_onnx_latency": lambda *args: {"mean_ms": 10.0},
        "measure_onnx_efficiency": efficiency,
        "save_benchmark_results": lambda: saved.append(copy.deepcopy(results)),
        "gc": gc, "json": json, "logger": logging.getLogger(__name__),
        "torch": SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: None)),
    }
    for name in ("onnx_fp32_session", "onnx_fp16_session", "onnx_int8_trt_session"):
        namespace[name] = Session(name)
    tree = ast.parse(cell_source(30))
    gates = [node for node in tree.body if isinstance(node, ast.If)]
    exec(compile(ast.Module(body=gates, type_ignores=[]), "<gates>", "exec"),
         namespace)
    expected = (["fp32.onnx"] if fp32 else []) + (["fp16.onnx"] if fp16 else [])
    assert seen == expected
    assert len(saved) == len(expected)
    assert not live
    for precision, enabled in (("fp32", fp32), ("fp16", fp16)):
        result = results["benchmarks"][f"onnx_{precision}_cuda"]
        assert result["metrics"]["miou"] == 0.7379
        assert ("efficiency" in result) is enabled
        if enabled:
            assert result["efficiency"]["session_residency"].startswith("one ONNX")


@pytest.mark.parametrize("implementation", ["notebook", "module"])
@pytest.mark.parametrize("counter", [True, False])
def test_nvml_energy_and_memory_scopes(monkeypatch, implementation, counter):
    class NVMLError(Exception):
        pass

    fake_nvml = SimpleNamespace(
        NVMLError=NVMLError,
        nvmlDeviceGetTotalEnergyConsumption=lambda handle: 3000,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(
            used=16 * 2**20, total=32 * 2**20),
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [
            SimpleNamespace(pid=os.getpid(), usedGpuMemory=8 * 2**20)],
    )
    if implementation == "module":
        monkeypatch.setattr(ptq_efficiency, "require_module", lambda name: fake_nvml)
        cls = ptq_efficiency.NvmlEfficiencyMonitor
    else:
        namespace = notebook_helpers()
        namespace["pynvml"] = fake_nvml
        cls = namespace["NvmlEfficiencyMonitor"]
    monitor = cls(0, 0.02)
    monitor.handle = object()
    assert monitor._read_memory_mib() == (8.0, "process")
    fake_nvml.nvmlDeviceGetComputeRunningProcesses = lambda handle: [
        SimpleNamespace(pid=os.getpid(), usedGpuMemory=2**64 - 1)]
    assert monitor._read_memory_mib() == (16.0, "device_total_fallback")
    monitor.samples = [(0.0, 20.0, 8.0, "process"),
                       (1.0, 40.0, 16.0, "device_total_fallback")]
    monitor.thread = SimpleNamespace(join=lambda **kwargs: None)
    monitor._sample_once = lambda: None
    monitor.total_energy_counter_available = counter
    monitor.energy_start_mj = 1000
    result = monitor.stop()
    assert result["total_energy_j"] == (2.0 if counter else 30.0)
    assert result["energy_source"] == (
        "nvml_total_energy_counter" if counter else "integrated_nvml_power_samples")
    assert result["energy_scope"] == "whole GPU device, idle not subtracted"
    assert result["gpu_memory_scope"] == "device_total_fallback"
    assert result["peak_gpu_memory_mib"] == 16.0


@pytest.mark.parametrize("implementation", ["notebook", "module"])
def test_efficiency_requires_both_duration_and_run_count(monkeypatch, implementation):
    clock = [0.0]
    forwards = []

    def run_once():
        forwards.append(None)
        clock[0] += 0.25

    class Monitor:
        def __init__(self, *args):
            pass

        def start(self):
            pass

        def stop(self):
            return {"total_energy_j": 2.0, "average_sampled_power_w": 2.0,
                    "peak_gpu_memory_mib": 800.0, "gpu_memory_scope": "process",
                    "energy_source": "nvml_total_energy_counter"}

    benchmark_cfg = SimpleNamespace(
        efficiency_min_seconds=1.0, efficiency_min_runs=3,
        efficiency_warmup_runs=2, nvml_sample_interval_seconds=0.02)
    fake_time = SimpleNamespace(perf_counter=lambda: clock[0])
    if implementation == "module":
        monkeypatch.setattr(ptq_efficiency, "time", fake_time)
        monkeypatch.setattr(ptq_efficiency, "build_onnx_gpu_runner", lambda *args: run_once)
        monkeypatch.setattr(ptq_efficiency, "NvmlEfficiencyMonitor", Monitor)
        result = ptq_efficiency.measure_onnx_efficiency(None, None, benchmark_cfg)
        benchmark_cfg.efficiency_min_seconds = 0
        with pytest.raises(ValueError):
            ptq_efficiency.measure_onnx_efficiency(None, None, benchmark_cfg)
    else:
        namespace = notebook_helpers()
        namespace.update({
            "time": fake_time, "build_onnx_gpu_runner": lambda *args: run_once,
            "NvmlEfficiencyMonitor": Monitor, "EFFICIENCY_MIN_SECONDS": 1.0,
            "EFFICIENCY_MIN_RUNS": 3, "EFFICIENCY_WARMUP_RUNS": 2,
            "TRT_DEVICE_ID": 0, "NVML_SAMPLE_INTERVAL_SECONDS": 0.02,
        })
        result = namespace["measure_onnx_efficiency"](None, None)
        namespace["EFFICIENCY_MIN_SECONDS"] = 0
        with pytest.raises(ValueError):
            namespace["measure_onnx_efficiency"](None, None)
    assert len(forwards) == 6  # two warm-ups, four measured forwards
    assert result["runs"] == 4
    assert result["elapsed_seconds"] == 1.0
    assert result["energy_per_image_mj"] == 500.0
    assert result["images_per_joule"] == 2.0
    assert result["gpu_memory_scope"] == "process"


def test_python_mirror_measures_each_baseline_alone(monkeypatch):
    cfg = recovery_cfg()
    paths = SimpleNamespace(fp32="fp32.onnx", fp16="fp16.onnx")
    results = {"benchmarks": {"onnx_fp32_cuda": {"metrics": {}},
                               "onnx_fp16_cuda": {"metrics": {}}}}
    live = weakref.WeakSet()
    seen = []
    saved = []

    class Session:
        def __init__(self, path):
            assert not live
            self.path = path
            live.add(self)

    def efficiency(session, sample, benchmark_cfg, device_id):
        assert len(live) == 1
        seen.append(session.path)
        assert benchmark_cfg.efficiency_min_seconds == 30.0
        return {"energy_per_image_mj": 400.0, "peak_gpu_memory_mib": 800.0}

    monkeypatch.setattr(ptq_efficiency, "resolve_ort_providers",
                        lambda *args: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ptq_efficiency, "create_onnx_session",
                        lambda path, providers: Session(path))
    monkeypatch.setattr(ptq_efficiency, "measure_onnx_latency",
                        lambda *args: {"mean_ms": 10.0})
    monkeypatch.setattr(ptq_efficiency, "measure_onnx_efficiency", efficiency)
    monkeypatch.setattr(ptq_efficiency, "persist_results",
                        lambda result, *args: saved.append(copy.deepcopy(result)))
    monkeypatch.setattr(ptq_efficiency.torch.cuda, "is_available", lambda: False)
    ptq_efficiency.run_baseline_efficiency(cfg, paths, results, object())
    assert seen == ["fp32.onnx", "fp16.onnx"]
    assert len(saved) == 2
    assert not live


def test_pipeline_releases_retained_sessions_before_baseline(monkeypatch, tmp_path):
    cfg = recovery_cfg()
    paths = SimpleNamespace(output_dir=tmp_path, results=tmp_path / "results.json",
                            checkpoint=tmp_path / "best.ckpt")
    live = weakref.WeakSet()
    measured = []

    class Session:
        def __init__(self):
            live.add(self)

    class Model:
        device = "cuda"

        def to(self, device):
            self.device = device

    model = Model()
    monkeypatch.setattr(ptq_pipeline, "validate_config", lambda cfg: None)
    monkeypatch.setattr(ptq_pipeline, "build_paths", lambda cfg: paths)
    monkeypatch.setattr(ptq_pipeline, "setup_logger",
                        lambda *args: SimpleNamespace(info=lambda *a: None,
                                                       debug=lambda *a: None))
    monkeypatch.setattr(ptq_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(ptq_pipeline, "_resolve_checkpoint", lambda cfg, paths: paths)
    monkeypatch.setattr(ptq_pipeline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ptq_pipeline, "load_trained_model", lambda *a, **kw: model)
    monkeypatch.setattr(ptq_pipeline, "_initial_results", lambda *args: {})
    monkeypatch.setattr(ptq_pipeline, "_evaluation_inputs", lambda cfg: (None, object()))
    monkeypatch.setattr(ptq_pipeline, "run_precision_stages",
                        lambda *args: (["CUDAExecutionProvider"], Session()))
    monkeypatch.setattr(ptq_pipeline, "run_int8_stages", lambda *args: None)
    monkeypatch.setattr(ptq_pipeline, "run_final_test", lambda *args: None)

    def measure(*args):
        assert model.device == "cpu"
        assert not live
        measured.append(True)

    monkeypatch.setattr(ptq_pipeline, "run_baseline_efficiency", measure)
    ptq_pipeline.run_ptq(cfg)
    assert measured == [True]
