from __future__ import annotations


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
