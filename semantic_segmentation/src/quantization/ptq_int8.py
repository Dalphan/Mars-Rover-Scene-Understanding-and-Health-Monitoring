from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.quantization.core import require_module, sha256
from src.quantization.ptq_graph_rewrite import (
    decompose_conv_biases,
    verify_fp32_graph_rewrite,
)
from src.quantization.ptq_int8_audit import (
    audit_qdq_initializers,
    count_op_types,
    count_qdq_wrapped_ops,
)
from src.quantization.ptq_onnx_utils import element_type_name, tensor_shape


def _quantization_options(cfg):
    options = {
        "ActivationSymmetric": bool(cfg.int8.symmetric),
        "WeightSymmetric": bool(cfg.int8.symmetric),
        "CalibTensorRangeSymmetric": bool(cfg.int8.symmetric),
        "DedicatedQDQPair": bool(cfg.int8.dedicated_qdq_pair),
        "QuantizeBias": bool(cfg.int8.quantize_bias),
    }
    method_name = str(cfg.int8.calibration_method)
    if method_name in {"Entropy", "Percentile"}:
        options["CalibStridedMinMax"] = int(cfg.int8.calibration_chunk_size)
    if method_name == "Percentile":
        percentile = float(cfg.int8.calibration_percentile)
        if not 0 < percentile <= 100:
            raise ValueError("calibration_percentile must be in (0, 100]")
        options["CalibPercentile"] = percentile
    return options


def _calibration_providers(ort, cfg):
    available = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if cfg.runtime.prefer_gpu and "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    if cfg.runtime.prefer_gpu and providers[0] == "CPUExecutionProvider":
        raise RuntimeError("CUDAExecutionProvider unavailable for calibration")
    return providers


def create_int8(fp32_path: Path, int8_path: Path, reader, cfg: DictConfig):
    """Create and audit the static INT8 ONNX artifact."""

    onnx = require_module("onnx")
    ort = require_module("onnxruntime")
    try:
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime quantization tools are unavailable") from exc

    formats = {"QDQ": QuantFormat.QDQ, "QOperator": QuantFormat.QOperator}
    methods = {
        "MinMax": CalibrationMethod.MinMax,
        "Entropy": CalibrationMethod.Entropy,
        "Percentile": CalibrationMethod.Percentile,
    }
    types = {"QInt8": QuantType.QInt8, "QUInt8": QuantType.QUInt8}
    quant_format = formats[str(cfg.int8.quant_format)]
    method = methods[str(cfg.int8.calibration_method)]
    activation_type = types[str(cfg.int8.activation_type)]
    weight_type = types[str(cfg.int8.weight_type)]
    op_types = [str(value) for value in cfg.int8.op_types]

    source_model = onnx.load(fp32_path)
    source_counts = count_op_types(source_model)
    source_path = fp32_path
    decomposition = {"enabled": False, "path": str(fp32_path)}
    if cfg.int8.decompose_conv_bias and "Conv" in op_types:
        source_path, decomposition = decompose_conv_biases(fp32_path)
        decomposition["numerical_parity_with_original"] = verify_fp32_graph_rewrite(
            fp32_path, source_path, reader
        )

    calibration_providers = _calibration_providers(ort, cfg)
    reader.rewind()
    started = time.perf_counter()
    quantize_static(
        model_input=str(source_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=method,
        per_channel=bool(cfg.int8.per_channel),
        reduce_range=bool(cfg.int8.reduce_range),
        op_types_to_quantize=op_types,
        use_external_data_format=False,
        calibration_providers=calibration_providers,
        extra_options=_quantization_options(cfg),
    )
    reader.rewind()
    model = onnx.load(int8_path)
    onnx.checker.check_model(model, full_check=True)
    audit = audit_qdq_initializers(onnx, model)
    if audit["tensorrt_incompatible_nodes"]:
        raise RuntimeError(
            f"TensorRT-incompatible Q/DQ nodes: "
            f"{audit['tensorrt_incompatible_nodes'][:5]}"
        )

    initializers = {item.name: item for item in model.graph.initializer}
    nonzero = []
    for node in model.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"} and len(node.input) >= 3:
            item = initializers.get(node.input[2])
            if item is not None and np.any(onnx.numpy_helper.to_array(item) != 0):
                nonzero.append(item.name)
    if cfg.int8.symmetric and nonzero:
        raise RuntimeError(f"Symmetric model has non-zero zero points: {nonzero[:5]}")
    int8_initializers = sum(
        item.data_type == onnx.TensorProto.INT8 for item in model.graph.initializer
    )
    quantize_nodes = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    if not int8_initializers or not quantize_nodes:
        raise RuntimeError("Output graph contains no effective INT8 Q/DQ quantization")

    return {
        "path": str(int8_path),
        "sha256": sha256(int8_path),
        "size_mib": int8_path.stat().st_size / 2**20,
        "size_reduction_percent_vs_fp32": 100.0
        * (1.0 - int8_path.stat().st_size / fp32_path.stat().st_size),
        "quantization_seconds": time.perf_counter() - started,
        "quant_format": str(cfg.int8.quant_format),
        "calibration_method": str(cfg.int8.calibration_method),
        "activation_type": str(cfg.int8.activation_type),
        "weight_type": str(cfg.int8.weight_type),
        "symmetric": bool(cfg.int8.symmetric),
        "per_channel_weights": bool(cfg.int8.per_channel),
        "conv_bias_decomposition": decomposition,
        "op_types_requested": op_types,
        "source_op_type_counts": source_counts,
        "qdq_wrapped_op_counts": count_qdq_wrapped_ops(model, op_types),
        "calibration_providers_requested": calibration_providers,
        "quantize_linear_nodes": quantize_nodes,
        "dequantize_linear_nodes": sum(
            node.op_type == "DequantizeLinear" for node in model.graph.node
        ),
        "int8_initializers": int8_initializers,
        "nonzero_zero_points": sorted(set(nonzero)),
        "qdq_initializer_audit": audit,
        "input_type": element_type_name(onnx, model.graph.input[0]),
        "input_shape": list(tensor_shape(model.graph.input[0])),
        "output_type": element_type_name(onnx, model.graph.output[0]),
        "output_shape": list(tensor_shape(model.graph.output[0])),
        "onnx_checker": "passed",
    }
