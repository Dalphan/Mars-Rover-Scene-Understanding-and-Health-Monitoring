from __future__ import annotations

import copy

import numpy as np
import torch
from omegaconf import DictConfig

from src.quantization.core import require_module, sha256
from src.quantization.qat_config import QATPaths
from src.quantization.qat_fp16_export import export_fp16_baseline
from src.quantization.qat_onnx_audit import audit_qat_graph
from src.quantization.qat_parity import FP32IOFP16Fallback, parity
from src.quantization.qat_weights import fold_constant_int8_weights


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
