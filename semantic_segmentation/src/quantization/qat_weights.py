from __future__ import annotations

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


