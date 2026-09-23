from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

from src.quantization.qat_model import freeze_batch_norm

def segformer_linear_group(name: str) -> str | None:
    if name.endswith((".mlp.dense1", ".mlp.dense2", ".mlp.fc1", ".mlp.fc2")):
        return "encoder_mlp"
    decoder = (
        ".decode_head." in name
        and name.endswith(".proj")
        and (".linear_c." in name or ".linear_projections." in name)
    )
    return "decoder_projection" if decoder else None


def prepare_selective_qat(student, calibration_loader, cfg, device, restore_path):
    try:
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.nn import TensorQuantizer
    except ImportError as exc:
        raise RuntimeError("NVIDIA ModelOpt 0.46.1 is required for QAT") from exc

    source_linear = {
        name: module
        for name, module in student.named_modules()
        if isinstance(module, nn.Linear)
    }
    selected_groups = [str(value) for value in cfg.model.int8_linear_groups]
    discovered = {
        group: sorted(
            name for name in source_linear if segformer_linear_group(name) == group
        )
        for group in selected_groups
    }
    expected_counts = {"encoder_mlp": 16, "decoder_projection": 4}
    for group in selected_groups:
        if group not in expected_counts or len(discovered[group]) != expected_counts[group]:
            raise RuntimeError(
                f"Unexpected SegFormer Linear group {group}: {len(discovered.get(group, []))}"
            )
    fallback_prefixes = [str(value) for value in cfg.model.fp16_fallback_module_prefixes]
    target_linear_names = sorted(
        name
        for names in discovered.values()
        for name in names
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in fallback_prefixes)
    )
    conv_parent = mtq.QuantModuleRegistry.get_key(nn.Conv2d)
    linear_parent = mtq.QuantModuleRegistry.get_key(nn.Linear)
    if not conv_parent or not linear_parent:
        raise RuntimeError("ModelOpt has no Conv2d/Linear quantization registry entries")
    linear_cfg = []
    for name in target_linear_names:
        linear_cfg.extend(
            [
                {
                    "quantizer_name": f"{name}.weight_quantizer",
                    "parent_class": linear_parent,
                    "enable": True,
                    "cfg": {"num_bits": 8, "axis": 0, "trt_high_precision_dtype": "Half"},
                },
                {
                    "quantizer_name": f"{name}.input_quantizer",
                    "parent_class": linear_parent,
                    "enable": True,
                    "cfg": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Half"},
                },
            ]
        )
    selective_cfg = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "parent_class": conv_parent,
                "enable": True,
                "cfg": {"num_bits": 8, "axis": 0, "trt_high_precision_dtype": "Half"},
            },
            {
                "quantizer_name": "*input_quantizer",
                "parent_class": conv_parent,
                "enable": True,
                "cfg": {"num_bits": 8, "axis": None, "trt_high_precision_dtype": "Half"},
            },
        ]
        + linear_cfg
        + [
            {"quantizer_name": f"*{prefix}.*", "enable": False}
            for prefix in fallback_prefixes
        ],
        "algorithm": "max",
    }

    @torch.inference_mode()
    def calibration_loop(model):
        model.eval()
        for batch in tqdm(calibration_loader, desc="ModelOpt calibration"):
            model(batch["image"].to(device, non_blocking=True))

    if cfg.steps.restore_saved_qat:
        student = mto.restore(student, str(restore_path), map_location=device).to(device)
    else:
        student = mtq.quantize(student, selective_cfg, forward_loop=calibration_loop)
    freeze_batch_norm(student)

    modules = dict(student.named_modules())
    conv_modules = {
        name: module for name, module in modules.items() if isinstance(module, nn.Conv2d)
    }
    linear_modules = {
        name: module for name, module in modules.items() if isinstance(module, nn.Linear)
    }
    layer_norms = {
        name: module for name, module in modules.items() if isinstance(module, nn.LayerNorm)
    }
    expected_fp16_conv = sorted(
        name
        for name in conv_modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in fallback_prefixes)
    )
    expected_int8_conv = sorted(set(conv_modules) - set(expected_fp16_conv))

    def quantized_names(modules_by_name):
        int8_names, fp16_names, partial = [], [], []
        for name, module in modules_by_name.items():
            input_q = getattr(module, "input_quantizer", None)
            weight_q = getattr(module, "weight_quantizer", None)
            input_enabled = input_q is not None and input_q.is_enabled
            weight_enabled = weight_q is not None and weight_q.is_enabled
            if input_enabled and weight_enabled:
                int8_names.append(name)
            elif not input_enabled and not weight_enabled:
                fp16_names.append(name)
            else:
                partial.append(name)
        return sorted(int8_names), sorted(fp16_names), sorted(partial)

    int8_conv, fp16_conv, partial_conv = quantized_names(conv_modules)
    int8_linear, fp16_linear, partial_linear = quantized_names(linear_modules)
    enabled = sorted(
        name
        for name, module in modules.items()
        if isinstance(module, TensorQuantizer) and module.is_enabled
    )
    expected_enabled = sorted(
        f"{name}.{quantizer}"
        for name in expected_int8_conv + target_linear_names
        for quantizer in ("input_quantizer", "weight_quantizer")
    )
    if (
        partial_conv
        or partial_linear
        or int8_conv != expected_int8_conv
        or fp16_conv != expected_fp16_conv
        or int8_linear != target_linear_names
        or set(enabled) != set(expected_enabled)
    ):
        raise RuntimeError(
            {
                "partial_conv": partial_conv,
                "partial_linear": partial_linear,
                "expected_int8_conv": expected_int8_conv,
                "actual_int8_conv": int8_conv,
                "expected_int8_linear": target_linear_names,
                "actual_int8_linear": int8_linear,
                "missing_quantizers": sorted(set(expected_enabled) - set(enabled)),
                "unexpected_quantizers": sorted(set(enabled) - set(expected_enabled)),
            }
        )
    audit = {
        "conv2d_total": len(conv_modules),
        "conv2d_int8": len(int8_conv),
        "conv2d_fp16_fallback": len(fp16_conv),
        "fp16_fallback_conv_names": fp16_conv,
        "linear_total": len(linear_modules),
        "linear_int8": len(int8_linear),
        "linear_int8_names": int8_linear,
        "linear_fp16": len(fp16_linear),
        "linear_fp16_names": fp16_linear,
        "linear_groups_int8": discovered,
        "layer_norm_total_fp16": len(layer_norms),
        "enabled_quantizer_names": enabled,
    }
    return student, audit, TensorQuantizer, mto


