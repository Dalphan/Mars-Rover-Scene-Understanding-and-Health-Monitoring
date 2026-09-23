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
