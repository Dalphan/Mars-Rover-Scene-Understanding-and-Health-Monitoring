from __future__ import annotations

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
