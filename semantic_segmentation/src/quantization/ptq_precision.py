from __future__ import annotations

from src.quantization.ptq_common import element_type_name, persist_results, tensor_shape

from pathlib import Path

from src.quantization.core import require_module, sha256

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

from src.quantization.core import (
    compare_onnx_sessions,
    compare_pytorch_and_onnx,
    create_onnx_session,
    evaluate_onnx,
    evaluate_torch,
    export_onnx_fp32,
    measure_onnx_latency,
    measure_pytorch_latency,
    resolve_ort_providers,
)


def run_precision_stages(
    cfg,
    paths,
    model,
    device,
    results,
    evaluation_loader,
    latency_sample,
):
    if cfg.steps.run_fp32_baseline:
        results["benchmarks"]["pytorch_fp32_cuda"] = {
            "runtime": "pytorch",
            "precision": "fp32",
            "device": str(device),
            "artifact_key": "pytorch_fp32",
            "metrics": evaluate_torch(
                model,
                evaluation_loader,
                device,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_pytorch_latency(
                model,
                latency_sample,
                device,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
        }
        persist_results(results, cfg, paths)

    if cfg.steps.run_onnx_export:
        info = export_onnx_fp32(
            model,
            paths.fp32,
            cfg.dataset.image_size,
            cfg.dataset.num_classes,
            cfg.onnx.opset_version,
        )
        results["artifacts"]["onnx_fp32"] = {
            "format": "onnx",
            "precision": "fp32",
            **info,
        }
        persist_results(results, cfg, paths)
    elif not paths.fp32.is_file():
        raise FileNotFoundError(f"Existing FP32 ONNX not found: {paths.fp32}")

    providers = resolve_ort_providers(
        bool(cfg.runtime.prefer_gpu), bool(cfg.runtime.prefer_tensorrt)
    )
    fp32_session = create_onnx_session(paths.fp32, providers)
    if cfg.steps.run_onnx_fp32_benchmark:
        results["benchmarks"]["onnx_fp32_cuda"] = {
            "runtime": "onnxruntime",
            "precision": "fp32",
            "device": fp32_session.get_providers()[0],
            "providers": fp32_session.get_providers(),
            "artifact_key": "onnx_fp32",
            "numerical_parity_with_pytorch": compare_pytorch_and_onnx(
                model, fp32_session, latency_sample, device
            ),
            "metrics": evaluate_onnx(
                fp32_session,
                evaluation_loader,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_onnx_latency(
                fp32_session,
                latency_sample,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
        }
        persist_results(results, cfg, paths)

    fp16_session = None
    if cfg.steps.run_onnx_fp16:
        info = convert_fp16(paths.fp32, paths.fp16, cfg.onnx.fp16_keep_io_types)
        results["artifacts"]["onnx_fp16"] = {
            "format": "onnx",
            "precision": "fp16",
            **info,
        }
        fp16_session = create_onnx_session(paths.fp16, providers)
        results["benchmarks"]["onnx_fp16_cuda"] = {
            "runtime": "onnxruntime",
            "precision": "fp16",
            "device": fp16_session.get_providers()[0],
            "providers": fp16_session.get_providers(),
            "artifact_key": "onnx_fp16",
            "numerical_parity_with_onnx_fp32": compare_onnx_sessions(
                fp32_session, fp16_session, latency_sample
            ),
            "metrics": evaluate_onnx(
                fp16_session,
                evaluation_loader,
                cfg.dataset.num_classes,
                cfg.dataset.ignore_index,
                cfg.dataset.evaluation_split,
                cfg.dataset.class_names,
            ),
            "latency": measure_onnx_latency(
                fp16_session,
                latency_sample,
                cfg.benchmark.latency_warmup_runs,
                cfg.benchmark.latency_measured_runs,
            ),
        }
        persist_results(results, cfg, paths)
    return providers, fp32_session
