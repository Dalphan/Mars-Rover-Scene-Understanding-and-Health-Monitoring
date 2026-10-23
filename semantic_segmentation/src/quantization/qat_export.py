from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FP32IOFP16Fallback(nn.Module):
    """Keep FP32 I/O while executing fallback operators with FP16 weights."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model(images.to(torch.float16)).float()


def parity(reference, candidate):
    error = np.abs(candidate.astype(np.float32) - reference.astype(np.float32))
    return {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "prediction_agreement": float(
            np.mean(candidate.argmax(1) == reference.argmax(1))
        ),
    }

import numpy as np

from src.quantization.core import require_module

def fold_constant_int8_weights(model):
    onnx = require_module("onnx")
    from onnx import numpy_helper

    initializers = {item.name: item for item in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node, input_index))

    def constant_array(name):
        if name in initializers:
            return numpy_helper.to_array(initializers[name])
        producer = producers.get(name)
        if producer is None:
            return None
        if producer.op_type == "Constant":
            value = next((attr for attr in producer.attribute if attr.name == "value"), None)
            return None if value is None else numpy_helper.to_array(value.t)
        if producer.op_type == "Cast":
            source = constant_array(producer.input[0])
            target = next(
                (
                    onnx.helper.get_attribute_value(attr)
                    for attr in producer.attribute
                    if attr.name == "to"
                ),
                None,
            )
            if source is None or target is None:
                return None
            return source.astype(onnx.helper.tensor_dtype_to_np_dtype(target))
        return None

    def downstream_weighted_users(tensor_name):
        found = {"conv": {}, "linear": {}}
        pending, visited = [tensor_name], set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for user, input_index in consumers.get(current, []):
                key = user.name or f"{user.op_type}:{user.output[0]}"
                if user.op_type == "Conv" and input_index == 1:
                    found["conv"][key] = user
                elif user.op_type in {"MatMul", "Gemm"} and input_index == 1:
                    found["linear"][key] = user
                elif input_index == 0 and user.op_type in {
                    "Transpose",
                    "Reshape",
                    "Cast",
                    "Identity",
                }:
                    pending.extend(user.output)
        return list(found["conv"].values()), list(found["linear"].values())

    folded_outputs, quantized_initializers = set(), []
    folded_conv_names, folded_linear_names, issues = [], [], []
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or node.input[0] not in initializers:
            continue
        dq_users = consumers.get(node.output[0], [])
        if not dq_users or any(user.op_type != "DequantizeLinear" for user, _ in dq_users):
            continue
        conv_by_name, linear_by_name = {}, {}
        for dq_node, _ in dq_users:
            conv_users, linear_users = downstream_weighted_users(dq_node.output[0])
            conv_by_name.update({item.name or item.output[0]: item for item in conv_users})
            linear_by_name.update({item.name or item.output[0]: item for item in linear_users})
        conv_users, linear_users = list(conv_by_name.values()), list(linear_by_name.values())
        if not conv_users and not linear_users:
            continue
        scale = constant_array(node.input[1])
        zero = constant_array(node.input[2]) if len(node.input) > 2 else None
        if scale is None or zero is None:
            issues.append({"quantizer": node.name, "reason": "scale_or_zero_not_constant"})
            continue
        weight = numpy_helper.to_array(initializers[node.input[0]]).astype(np.float32)
        scale, zero = np.asarray(scale, dtype=np.float32), np.asarray(zero)
        if zero.dtype != np.int8 or np.any(zero != 0) or np.any(scale <= 0):
            issues.append({"quantizer": node.name, "reason": "invalid_int8_parameters"})
            continue
        axis = next(
            (
                onnx.helper.get_attribute_value(attr)
                for attr in node.attribute
                if attr.name == "axis"
            ),
            1,
        )
        axis = axis if axis >= 0 else weight.ndim + axis
        if not 0 <= axis < weight.ndim:
            issues.append({"quantizer": node.name, "reason": "invalid_axis"})
            continue
        if scale.size > 1 and (
            scale.size != weight.shape[axis] or zero.size != scale.size
        ):
            issues.append({"quantizer": node.name, "reason": "per_channel_shape_mismatch"})
            continue
        shape = [1] * weight.ndim
        if scale.size > 1:
            shape[axis] = scale.size
        quantized = np.rint(weight / scale.reshape(shape)) + zero.reshape(shape)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        quantized_initializers.append(
            numpy_helper.from_array(quantized, node.output[0])
        )
        folded_outputs.add(node.output[0])
        folded_conv_names.extend(item.name for item in conv_users)
        folded_linear_names.extend(item.name for item in linear_users)

    kept_nodes = [
        node
        for node in model.graph.node
        if node.op_type != "QuantizeLinear" or node.output[0] not in folded_outputs
    ]
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    model.graph.initializer.extend(quantized_initializers)
    used = {name for node in model.graph.node for name in node.input}
    used.update(output.name for output in model.graph.output)
    kept_initializers = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    return model, {
        "folded_weight_quantizers": len(folded_outputs),
        "int8_weight_initializers": len(quantized_initializers),
        "folded_conv_names": folded_conv_names,
        "folded_linear_names": folded_linear_names,
        "issues": issues,
    }

def _qdq_wrapped(node, producers):
    def traces_to_dequantize(tensor_name):
        visited = set()
        while tensor_name not in visited:
            visited.add(tensor_name)
            producer = producers.get(tensor_name)
            if producer is None:
                return False
            if producer.op_type == "DequantizeLinear":
                return True
            if producer.op_type not in {
                "Transpose",
                "Reshape",
                "Flatten",
                "Cast",
                "Identity",
            }:
                return False
            tensor_name = producer.input[0]
        return False

    return (
        len(node.input) >= 2
        and traces_to_dequantize(node.input[0])
        and traces_to_dequantize(node.input[1])
    )


def audit_qat_graph(onnx_model, quantizer_audit, cfg, weight_fold):
    """Check that exported Q/DQ coverage matches the PyTorch quantizer audit."""

    producers = {
        output: node for node in onnx_model.graph.node for output in node.output
    }
    conv_nodes = [node for node in onnx_model.graph.node if node.op_type == "Conv"]
    linear_nodes = [
        node for node in onnx_model.graph.node if node.op_type in {"MatMul", "Gemm"}
    ]
    q_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "QuantizeLinear"
    ]
    dq_nodes = [
        node for node in onnx_model.graph.node if node.op_type == "DequantizeLinear"
    ]
    wrapped_conv = [node for node in conv_nodes if _qdq_wrapped(node, producers)]
    fp16_conv = [node for node in conv_nodes if not _qdq_wrapped(node, producers)]
    wrapped_linear = [node for node in linear_nodes if _qdq_wrapped(node, producers)]
    fp16_linear = [
        node for node in linear_nodes if not _qdq_wrapped(node, producers)
    ]

    fallback_prefixes = [
        str(value) for value in cfg.model.fp16_fallback_module_prefixes
    ]
    unexpected_fp16 = [
        node.name
        for node in fp16_conv
        if not any(
            prefix in node.name.replace("/", ".") for prefix in fallback_prefixes
        )
    ]
    unexpected_int8 = [
        node.name
        for node in wrapped_conv
        if any(prefix in node.name.replace("/", ".") for prefix in fallback_prefixes)
    ]
    expected_linear_names = quantizer_audit["linear_int8_names"]
    missing_linear_names = [
        name
        for name in expected_linear_names
        if not any(name in node.name.replace("/", ".") for node in wrapped_linear)
    ]
    unexpected_linear = [
        node.name
        for node in wrapped_linear
        if not any(
            name in node.name.replace("/", ".") for name in expected_linear_names
        )
    ]
    if (
        not q_nodes
        or len(wrapped_conv) != int(quantizer_audit["conv2d_int8"])
        or len(fp16_conv) != int(quantizer_audit["conv2d_fp16_fallback"])
        or unexpected_fp16
        or unexpected_int8
        or len(wrapped_linear) != int(quantizer_audit["linear_int8"])
        or missing_linear_names
        or unexpected_linear
    ):
        raise RuntimeError(
            {
                "conv_qdq": len(wrapped_conv),
                "conv_fp16": [node.name for node in fp16_conv],
                "linear_qdq": [node.name for node in wrapped_linear],
                "missing_int8_linear": missing_linear_names,
                "unexpected_int8_linear": unexpected_linear,
            }
        )
    return {
        "conv": len(conv_nodes),
        "conv_qdq_wrapped_int8": len(wrapped_conv),
        "conv_fp16_fallback": len(fp16_conv),
        "fp16_fallback_conv_names": [node.name for node in fp16_conv],
        "matmul_gemm_total": len(linear_nodes),
        "matmul_gemm_qdq_wrapped_int8": len(wrapped_linear),
        "matmul_gemm_qdq_wrapped_names": [node.name for node in wrapped_linear],
        "matmul_gemm_fp16": len(fp16_linear),
        "int8_linear_module_names": expected_linear_names,
        "quantize_linear": len(q_nodes),
        "dequantize_linear": len(dq_nodes),
        "constant_weight_qdq_fold": weight_fold,
        "onnx_checker": "passed",
    }

import copy

import torch

from src.quantization.core import require_module, sha256


def export_fp16_baseline(
    student, sample, sample_np, tensor_quantizer_type, cfg, paths
):
    """Export the same post-QAT weights with all quantizers disabled."""

    if not cfg.steps.run_trt_fp16_baseline:
        return None, None
    onnx = require_module("onnx")
    ort = require_module("onnxruntime")
    fp16_model = copy.deepcopy(student).eval()
    for module in fp16_model.modules():
        if isinstance(module, tensor_quantizer_type):
            module.disable()
    still_enabled = [
        name
        for name, module in fp16_model.named_modules()
        if isinstance(module, tensor_quantizer_type) and module.is_enabled
    ]
    if still_enabled:
        raise RuntimeError({"fp16_quantizers_still_enabled": still_enabled})

    fp16_export = FP32IOFP16Fallback(fp16_model.half()).eval()
    with torch.inference_mode():
        fp16_output = fp16_export(sample).detach().cpu().numpy()
        torch.onnx.export(
            fp16_export,
            sample,
            str(paths.onnx_fp16),
            input_names=["images"],
            output_names=["logits"],
            opset_version=int(cfg.export.onnx_opset),
            dynamo=False,
            do_constant_folding=True,
        )
    fp16_onnx = onnx.load(paths.onnx_fp16)
    onnx.checker.check_model(fp16_onnx, full_check=True)
    fp16_session = ort.InferenceSession(
        str(paths.onnx_fp16),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    fp16_cuda_output = fp16_session.run(["logits"], {"images": sample_np})[0]
    export_parity = parity(fp16_output, fp16_cuda_output)
    if export_parity["prediction_agreement"] < float(
        cfg.export.export_min_prediction_agreement
    ):
        raise RuntimeError(f"FP16 ONNX export parity failed: {export_parity}")
    artifact = {
        "path": str(paths.onnx_fp16),
        "sha256": sha256(paths.onnx_fp16),
        "size_mib": paths.onnx_fp16.stat().st_size / 2**20,
        "disabled_quantizers": len(
            [
                module
                for module in fp16_model.modules()
                if isinstance(module, tensor_quantizer_type)
            ]
        ),
        "parity_with_pytorch_fp16": export_parity,
    }
    return artifact, fp16_output

import copy

import numpy as np
import torch
from omegaconf import DictConfig

from src.quantization.core import require_module, sha256
from src.quantization.qat_config import QATPaths


def _fold_and_audit_onnx(onnx, paths, quantizer_audit, cfg):
    model = onnx.load(paths.onnx_int8)
    onnx.checker.check_model(model, full_check=True)
    raw_conv_count = sum(node.op_type == "Conv" for node in model.graph.node)
    model, weight_fold = fold_constant_int8_weights(model)
    expected_folded = int(quantizer_audit["conv2d_int8"]) + int(
        quantizer_audit["linear_int8"]
    )
    if raw_conv_count != int(quantizer_audit["conv2d_total"]):
        raise RuntimeError(
            {
                "pytorch_conv": quantizer_audit["conv2d_total"],
                "onnx_conv": raw_conv_count,
            }
        )
    if (
        weight_fold["issues"]
        or weight_fold["folded_weight_quantizers"] != expected_folded
    ):
        raise RuntimeError(
            {
                "expected_int8_weight_initializers": expected_folded,
                "weight_fold": weight_fold,
            }
        )
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, paths.onnx_int8)
    return audit_qat_graph(model, quantizer_audit, cfg, weight_fold)


def export_qat_onnx(
    student,
    val_loader,
    tensor_quantizer_type,
    quantizer_audit,
    cfg: DictConfig,
    paths: QATPaths,
    device,
):
    """Export the selectively quantized QAT model and validate runtime parity."""

    onnx = require_module("onnx")
    ort = require_module("onnxruntime")
    sample = next(iter(val_loader))["image"].to(device)
    student.eval()
    for module in student.modules():
        if isinstance(module, tensor_quantizer_type) and module.is_enabled:
            module.trt_high_precision_dtype = "Half"
    with torch.inference_mode():
        fake_quant_output = student(sample).detach().cpu().numpy()

    export_model = FP32IOFP16Fallback(copy.deepcopy(student).half()).eval()
    with torch.inference_mode():
        fallback_output = export_model(sample).detach().cpu().numpy()
    fallback_parity = parity(fake_quant_output, fallback_output)
    if fallback_parity["prediction_agreement"] < float(
        cfg.export.fp16_fallback_min_prediction_agreement
    ):
        raise RuntimeError(f"FP16 fallback parity failed: {fallback_parity}")

    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            sample,
            str(paths.onnx_int8),
            input_names=["images"],
            output_names=["logits"],
            opset_version=int(cfg.export.onnx_opset),
            dynamo=False,
            do_constant_folding=True,
        )
    onnx_audit = _fold_and_audit_onnx(onnx, paths, quantizer_audit, cfg)

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is required for QAT export parity")
    cuda_session = ort.InferenceSession(
        str(paths.onnx_int8),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    sample_np = sample.detach().cpu().numpy().astype(np.float32, copy=False)
    cuda_output = cuda_session.run(["logits"], {"images": sample_np})[0]
    export_parity = parity(fallback_output, cuda_output)
    if export_parity["prediction_agreement"] < float(
        cfg.export.export_min_prediction_agreement
    ):
        raise RuntimeError(f"ONNX QAT export parity failed: {export_parity}")
    artifact = {
        "path": str(paths.onnx_int8),
        "sha256": sha256(paths.onnx_int8),
        "size_mib": paths.onnx_int8.stat().st_size / 2**20,
        "opset": int(cfg.export.onnx_opset),
        "audit": onnx_audit,
        "fp16_fallback_parity_with_fp32_fake_quant": fallback_parity,
        "parity_with_pytorch_fp16_fallback": export_parity,
    }
    fp16_artifact, fp16_output = export_fp16_baseline(
        student,
        sample,
        sample_np,
        tensor_quantizer_type,
        cfg,
        paths,
    )
    return artifact, fp16_artifact, sample, fallback_output, fp16_output
