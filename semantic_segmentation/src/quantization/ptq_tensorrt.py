from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.quantization.core import require_module, sha256

def preload_tensorrt_shared_libraries():
    """Expose TensorRT 10 native wheels to the ONNX Runtime loader on Linux."""

    if os.name != "posix":
        # Windows wheels expose DLLs through the normal loader path.
        return None, [], []
    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "tensorrt_libs was not found; install requirements-quantization.txt"
        )
    library_dir = Path(next(iter(spec.submodule_search_locations)))
    handles, loaded_paths = [], []
    for soname in (
        "libnvinfer.so.10",
        "libnvinfer_plugin.so.10",
        "libnvonnxparser.so.10",
    ):
        candidates = sorted(
            library_dir.glob(f"{soname}*"), key=lambda path: len(path.name)
        )
        if not candidates:
            raise RuntimeError(f"Missing TensorRT native library: {soname}")
        selected = candidates[0]
        handles.append(
            ctypes.CDLL(str(selected), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        )
        loaded_paths.append(str(selected))
    return library_dir, handles, loaded_paths


def _trt_provider_options(cfg: DictConfig, model_path: Path):
    signature = sha256(model_path)[:16]
    engine_dir = Path(to_absolute_path(str(cfg.tensorrt.engine_cache_dir))) / signature
    timing_dir = Path(to_absolute_path(str(cfg.tensorrt.timing_cache_dir)))
    engine_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "device_id": int(cfg.tensorrt.device_id),
        "trt_int8_enable": bool(cfg.tensorrt.int8_enable),
        "trt_fp16_enable": bool(cfg.tensorrt.fp16_enable),
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(engine_dir),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(timing_dir),
    }
    return options, engine_dir


def create_trt_session(cfg: DictConfig, model_path: Path, profile=False):
    ort = require_module("onnxruntime")
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("TensorRTExecutionProvider is not available")
    options, engine_dir = _trt_provider_options(cfg, model_path)
    session_options = ort.SessionOptions()
    session_options.enable_profiling = bool(profile)
    providers = [
        ("TensorrtExecutionProvider", options),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return (
        ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        ),
        engine_dir,
    )


def probe_trt_int8(cfg, model_path, reference_session, sample):
    session, engine_dir = create_trt_session(cfg, model_path, profile=True)
    before = list(engine_dir.rglob("*"))
    candidate_input = np.ascontiguousarray(sample.cpu().numpy(), dtype=np.float32)
    started = time.perf_counter()
    candidate_output = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: candidate_input},
    )[0]
    first_run_seconds = time.perf_counter() - started
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    trt_events = [
        event
        for event in events
        if "TensorrtExecutionProvider"
        in str(event.get("args", {}).get("provider", ""))
    ]
    if not trt_events:
        raise RuntimeError(
            "TensorRT session ran only through fallback providers; INT8 probe failed"
        )
    reference_input = np.ascontiguousarray(sample.cpu().numpy(), dtype=np.float32)
    reference_output = reference_session.run(
        [reference_session.get_outputs()[0].name],
        {reference_session.get_inputs()[0].name: reference_input},
    )[0]
    error = np.abs(reference_output.astype(np.float32) - candidate_output.astype(np.float32))
    after = [item for item in engine_dir.rglob("*") if item.is_file()]
    return {
        "status": "passed",
        "providers": session.get_providers(),
        "tensorrt_profile_events": len(trt_events),
        "profile_path": str(profile_path),
        "engine_cache_path": str(engine_dir),
        "engine_cache_files_before": len([item for item in before if item.is_file()]),
        "engine_cache_files_after": len(after),
        "first_run_seconds_including_possible_engine_build": first_run_seconds,
        "max_abs_error_vs_onnx_fp32": float(error.max()),
        "mean_abs_error_vs_onnx_fp32": float(error.mean()),
        "prediction_agreement_vs_onnx_fp32": float(
            np.mean(
                np.argmax(reference_output, axis=1)
                == np.argmax(candidate_output, axis=1)
            )
        ),
    }
