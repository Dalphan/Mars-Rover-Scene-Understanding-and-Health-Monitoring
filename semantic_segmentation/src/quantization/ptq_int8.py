from __future__ import annotations

from src.quantization.ptq_common import element_type_name, tensor_shape

from pathlib import Path

import numpy as np

from src.quantization.core import require_module


def decompose_conv_biases(fp32_path: Path):
    onnx = require_module("onnx")
    model = onnx.load(fp32_path)
    initializers = {item.name: item for item in model.graph.initializer}
    used_names = set(initializers)
    used_names.update(output for node in model.graph.node for output in node.output)

    def unique(base):
        candidate, suffix = base, 1
        while candidate in used_names:
            candidate, suffix = f"{base}_{suffix}", suffix + 1
        used_names.add(candidate)
        return candidate

    rewritten, decomposed = [], []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 3 or not node.input[2]:
            rewritten.append(node)
            continue
        bias, weight = initializers.get(node.input[2]), initializers.get(node.input[1])
        if bias is None or weight is None or len(weight.dims) < 3:
            raise RuntimeError(f"Cannot safely decompose bias for {node.name}")
        bias_array = onnx.numpy_helper.to_array(bias)
        channels = int(weight.dims[0])
        if bias_array.ndim != 1 or bias_array.size != channels:
            raise RuntimeError(f"Unexpected Conv bias shape for {node.name}")
        original_output = node.output[0]
        conv_output = unique(f"{original_output}__without_bias")
        broadcast_name = unique(f"{node.input[2]}__broadcast")
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(
                bias_array.reshape((1, channels) + (1,) * (len(weight.dims) - 2)),
                name=broadcast_name,
            )
        )
        del node.input[2:]
        node.output[0] = conv_output
        rewritten.extend(
            [
                node,
                onnx.helper.make_node(
                    "Add",
                    [conv_output, broadcast_name],
                    [original_output],
                    name=unique(f"{node.name or original_output}__bias_add"),
                ),
            ]
        )
        decomposed.append(node.name or original_output)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    output = fp32_path.with_name(f"{fp32_path.stem}_trt_biasless_conv_source.onnx")
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, output)
    return output, {
        "enabled": True,
        "path": str(output),
        "decomposed_conv_bias_count": len(decomposed),
        "decomposed_conv_nodes": decomposed,
        "added_float_add_nodes": len(decomposed),
        "onnx_checker": "passed",
    }


def verify_fp32_graph_rewrite(reference_path, candidate_path, reader):
    ort = require_module("onnxruntime")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    reader.rewind()
    sample = reader.get_next()
    reader.rewind()
    if sample is None:
        raise RuntimeError("Calibration reader is empty during graph-rewrite gate")
    reference = ort.InferenceSession(str(reference_path), providers=providers)
    candidate = ort.InferenceSession(str(candidate_path), providers=providers)
    reference_output = reference.run(None, sample)[0]
    candidate_output = candidate.run(None, sample)[0]
    error = np.abs(reference_output - candidate_output)
    parity = {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "prediction_agreement": float(
            np.mean(
                np.argmax(reference_output, axis=1)
                == np.argmax(candidate_output, axis=1)
            )
        ),
        "allclose_rtol": 1e-5,
        "allclose_atol": 1e-5,
        "allclose": bool(
            np.allclose(reference_output, candidate_output, rtol=1e-5, atol=1e-5)
        ),
    }
    if not parity["allclose"]:
        raise RuntimeError(f"Conv-bias graph rewrite is not equivalent: {parity}")
    return parity

def count_op_types(model):
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return dict(sorted(counts.items()))


def count_qdq_wrapped_ops(model, configured_op_types):
    dequantized = {
        output
        for node in model.graph.node
        if node.op_type == "DequantizeLinear"
        for output in node.output
    }
    quantized_inputs = {
        node.input[0]
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input
    }
    counts = {str(op_type): 0 for op_type in configured_op_types}
    for node in model.graph.node:
        input_is_quantized = any(name in dequantized for name in node.input)
        output_is_quantized = any(name in quantized_inputs for name in node.output)
        if node.op_type in counts and input_is_quantized and output_is_quantized:
            counts[node.op_type] += 1
    return counts


def audit_qdq_initializers(onnx, model):
    initializers = {item.name: item for item in model.graph.initializer}
    source_types, zero_point_types, unsupported = {}, {}, []
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or not node.input:
            continue
        source_type = None
        if node.op_type == "DequantizeLinear" and node.input[0] in initializers:
            source_type = onnx.TensorProto.DataType.Name(
                initializers[node.input[0]].data_type
            )
            source_types[source_type] = source_types.get(source_type, 0) + 1
        zero_type = None
        if len(node.input) >= 3 and node.input[2] in initializers:
            zero_type = onnx.TensorProto.DataType.Name(
                initializers[node.input[2]].data_type
            )
            zero_point_types[zero_type] = zero_point_types.get(zero_type, 0) + 1
        if source_type in {"INT32", "UINT8"} or zero_type == "UINT8":
            unsupported.append(
                {
                    "node": node.name,
                    "source_type": source_type,
                    "zero_point_type": zero_type,
                }
            )
    return {
        "initializer_source_type_counts": dict(sorted(source_types.items())),
        "zero_point_type_counts": dict(sorted(zero_point_types.items())),
        "tensorrt_incompatible_nodes": unsupported,
    }

import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.quantization.core import require_module, sha256


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
