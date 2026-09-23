from __future__ import annotations

from pathlib import Path

from src.quantization.core import require_module, sha256
from src.quantization.ptq_onnx_utils import element_type_name, tensor_shape

def _deduplicate_fp16_metadata(model):
    producer_by_output = {}
    retained_nodes = []
    removed_nodes = 0
    for node in model.graph.node:
        outputs = [name for name in node.output if name]
        duplicate_outputs = [name for name in outputs if name in producer_by_output]
        if duplicate_outputs:
            previous = [producer_by_output[name] for name in duplicate_outputs]
            if (
                node.op_type == "Cast"
                and len(duplicate_outputs) == len(outputs)
                and all(
                    item.op_type == "Cast"
                    and item.SerializeToString() == node.SerializeToString()
                    for item in previous
                )
            ):
                removed_nodes += 1
                continue
            raise RuntimeError(f"Non-equivalent ONNX SSA collision: {duplicate_outputs}")
        retained_nodes.append(node)
        for output in outputs:
            producer_by_output[output] = node
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)

    retained_info = []
    info_by_name = {}
    removed_info = 0
    for value_info in model.graph.value_info:
        previous = info_by_name.get(value_info.name)
        if previous is None:
            info_by_name[value_info.name] = value_info
            retained_info.append(value_info)
        elif previous.SerializeToString() == value_info.SerializeToString():
            removed_info += 1
        else:
            raise RuntimeError(f"Conflicting ONNX metadata for {value_info.name}")
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    return removed_nodes, removed_info


def convert_fp16(fp32_path: Path, fp16_path: Path, keep_io_types: bool) -> dict:
    onnx = require_module("onnx")
    try:
        from onnxruntime.quantization.onnx_model import ONNXModel
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime FP16 tools are unavailable") from exc

    model = convert_float_to_float16(
        onnx.load(fp32_path),
        keep_io_types=bool(keep_io_types),
        disable_shape_infer=False,
        force_fp16_initializers=False,
    )
    removed_nodes, removed_info = _deduplicate_fp16_metadata(model)
    initializer_types = {item.name: item.data_type for item in model.graph.initializer}
    corrected = []
    for value_info in model.graph.value_info:
        expected = initializer_types.get(value_info.name)
        tensor_type = value_info.type.tensor_type
        if expected is not None and tensor_type.elem_type != expected:
            corrected.append(value_info.name)
            tensor_type.elem_type = expected
    ONNXModel(model).save_model_to_file(
        str(fp16_path), use_external_data_format=False
    )
    checked = onnx.load(fp16_path)
    onnx.checker.check_model(checked, full_check=True)
    fp16_initializers = sum(
        item.data_type == onnx.TensorProto.FLOAT16 for item in checked.graph.initializer
    )
    if not fp16_initializers:
        raise RuntimeError("FP16 conversion produced no FP16 initializers")
    return {
        "path": str(fp16_path),
        "sha256": sha256(fp16_path),
        "size_mib": fp16_path.stat().st_size / 2**20,
        "size_reduction_percent_vs_fp32": 100.0
        * (1.0 - fp16_path.stat().st_size / fp32_path.stat().st_size),
        "input_name": checked.graph.input[0].name,
        "input_type": element_type_name(onnx, checked.graph.input[0]),
        "input_shape": list(tensor_shape(checked.graph.input[0])),
        "output_name": checked.graph.output[0].name,
        "output_type": element_type_name(onnx, checked.graph.output[0]),
        "output_shape": list(tensor_shape(checked.graph.output[0])),
        "keep_io_types": bool(keep_io_types),
        "fp16_initializers": fp16_initializers,
        "fp32_initializers": sum(
            item.data_type == onnx.TensorProto.FLOAT
            for item in checked.graph.initializer
        ),
        "cast_nodes": sum(node.op_type == "Cast" for node in checked.graph.node),
        "deduplicated_equivalent_cast_nodes": removed_nodes,
        "deduplicated_equivalent_value_info": removed_info,
        "corrected_initializer_value_info_types": corrected,
        "onnx_checker": "passed",
    }
