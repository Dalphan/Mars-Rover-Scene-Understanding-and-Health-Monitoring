from __future__ import annotations


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
