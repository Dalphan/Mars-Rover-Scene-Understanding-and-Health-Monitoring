from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.metrics.segmentation_metrics import (
    create_confusion_matrix,
    update_confusion_matrix,
)
from src.quantization.core.dependencies import require_module, sha256
from src.quantization.core.metrics import quantization_metrics

LOGGER = logging.getLogger(__name__)
GPU_ORT_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}

def resolve_ort_providers(prefer_gpu: bool, prefer_tensorrt: bool):
    ort = require_module("onnxruntime")
    available = set(ort.get_available_providers())
    providers = []
    if prefer_tensorrt and "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")
    if prefer_gpu and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    if prefer_gpu and providers[0] == "CPUExecutionProvider":
        LOGGER.warning(
            "GPU execution was requested but ORT exposes no CUDA/TensorRT "
            "provider; using CPUExecutionProvider"
        )
    return providers


def create_onnx_session(model_path, providers):
    ort = require_module("onnxruntime")
    session = ort.InferenceSession(str(model_path), providers=list(providers))
    if providers[0] in GPU_ORT_PROVIDERS and session.get_providers()[0] not in GPU_ORT_PROVIDERS:
        raise RuntimeError(
            f"Requested a GPU ORT provider, active chain={session.get_providers()}"
        )
    return session


def prepare_onnx_input(session, sample):
    input_type = session.get_inputs()[0].type
    numpy_dtypes = {"tensor(float)": np.float32, "tensor(float16)": np.float16}
    if input_type not in numpy_dtypes:
        raise ValueError(f"Unsupported ONNX input type: {input_type}")
    return np.ascontiguousarray(sample.cpu().numpy(), dtype=numpy_dtypes[input_type])


def export_onnx_fp32(
    model,
    output_path: str | Path,
    image_size,
    num_classes: int,
    opset_version: int,
):
    onnx = require_module("onnx")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    parameter = next(model.parameters())
    if parameter.dtype != torch.float32:
        raise ValueError(f"Expected an FP32 model, found {parameter.dtype}")
    example = torch.zeros(
        (1, 3, *tuple(image_size)), dtype=torch.float32, device=parameter.device
    )
    torch.onnx.export(
        model,
        (example,),
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        opset_version=int(opset_version),
        dynamo=True,
        external_data=False,
        export_params=True,
        keep_initializers_as_inputs=False,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    def tensor_shape(value_info):
        return tuple(
            dimension.dim_value if dimension.HasField("dim_value") else None
            for dimension in value_info.type.tensor_type.shape.dim
        )

    input_info, output_info = onnx_model.graph.input[0], onnx_model.graph.output[0]
    expected_input = (1, 3, *tuple(image_size))
    expected_output = (1, int(num_classes), *tuple(image_size))
    if input_info.name != "images" or tensor_shape(input_info) != expected_input:
        raise ValueError(
            f"Unexpected ONNX input: {input_info.name}, {tensor_shape(input_info)}"
        )
    if output_info.name != "logits" or tensor_shape(output_info) != expected_output:
        raise ValueError(
            f"Unexpected ONNX output: {output_info.name}, {tensor_shape(output_info)}"
        )
    opset = next(
        item.version
        for item in onnx_model.opset_import
        if item.domain in ("", "ai.onnx")
    )
    return {
        "path": str(output_path),
        "sha256": sha256(output_path),
        "size_mib": output_path.stat().st_size / 2**20,
        "opset": opset,
        "input_name": input_info.name,
        "input_shape": list(expected_input),
        "output_name": output_info.name,
        "output_shape": list(expected_output),
        "onnx_checker": "passed",
    }


@torch.inference_mode()
def compare_pytorch_and_onnx(model, session, sample, device):
    sample_numpy = prepare_onnx_input(session, sample)
    pytorch_logits = model(sample.to(device, non_blocking=True)).cpu().numpy()
    onnx_logits = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: sample_numpy},
    )[0]
    absolute_error = np.abs(pytorch_logits - onnx_logits)
    pytorch_predictions = np.argmax(pytorch_logits, axis=1)
    onnx_predictions = np.argmax(onnx_logits, axis=1)
    rtol, atol = 1e-4, 1e-5
    return {
        "output_shape": list(onnx_logits.shape),
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "prediction_agreement": float(
            np.mean(pytorch_predictions == onnx_predictions)
        ),
        "allclose_rtol": rtol,
        "allclose_atol": atol,
        "allclose": bool(np.allclose(pytorch_logits, onnx_logits, rtol=rtol, atol=atol)),
    }


def compare_onnx_sessions(reference_session, candidate_session, sample):
    reference_input = prepare_onnx_input(reference_session, sample)
    candidate_input = prepare_onnx_input(candidate_session, sample)
    reference = reference_session.run(
        [reference_session.get_outputs()[0].name],
        {reference_session.get_inputs()[0].name: reference_input},
    )[0]
    candidate = candidate_session.run(
        [candidate_session.get_outputs()[0].name],
        {candidate_session.get_inputs()[0].name: candidate_input},
    )[0]
    error = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "prediction_agreement": float(
            np.mean(np.argmax(reference, axis=1) == np.argmax(candidate, axis=1))
        ),
    }


def evaluate_onnx(
    session, loader, num_classes, ignore_index, split_name, class_names=None
):
    confmat = create_confusion_matrix(int(num_classes))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for batch in loader:
        images = prepare_onnx_input(session, batch["image"])
        logits = session.run([output_name], {input_name: images})[0]
        predictions = torch.from_numpy(
            np.argmax(logits, axis=1).astype(np.int64, copy=False)
        )
        update_confusion_matrix(
            confmat,
            predictions,
            batch["mask"],
            int(num_classes),
            int(ignore_index),
        )
    metrics = quantization_metrics(
        confmat, num_classes, ignore_index, class_names
    )
    metrics["split"] = str(split_name)
    metrics["samples"] = len(loader.dataset)
    return metrics
