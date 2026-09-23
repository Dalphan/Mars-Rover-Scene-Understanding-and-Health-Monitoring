from __future__ import annotations


def tensor_shape(value_info):
    return tuple(
        dimension.dim_value if dimension.HasField("dim_value") else None
        for dimension in value_info.type.tensor_type.shape.dim
    )


def element_type_name(onnx, value_info):
    return onnx.TensorProto.DataType.Name(value_info.type.tensor_type.elem_type)
