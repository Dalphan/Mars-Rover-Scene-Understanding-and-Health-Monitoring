from __future__ import annotations

import copy

import torch

from src.quantization.core import require_module, sha256
from src.quantization.qat_parity import FP32IOFP16Fallback, parity


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
