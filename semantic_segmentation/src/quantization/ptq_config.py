from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass(frozen=True)
class PTQPaths:
    output_dir: Path
    checkpoint: Path
    fp32: Path
    fp16: Path
    int8: Path
    manifest: Path
    results: Path


def operator_scope(op_types) -> str:
    values = tuple(str(value) for value in op_types)
    if values == ("Conv",):
        return "conv_only"
    if set(values) == {"Conv", "MatMul", "Gemm"}:
        return "conv_matmul_gemm"
    return "expanded_ops"


def build_paths(cfg: DictConfig) -> PTQPaths:
    output_dir = Path(to_absolute_path(str(cfg.onnx.output_dir)))
    experiment = str(cfg.checkpoint.experiment)
    bias_scope = (
        "conv_bias_add_float"
        if cfg.int8.decompose_conv_bias
        else "bias_int32" if cfg.int8.quantize_bias else "bias_float"
    )
    int8_name = (
        f"{experiment}_int8_qdq_{str(cfg.int8.calibration_method).lower()}_"
        f"{cfg.data.calibration_selection}_{operator_scope(cfg.int8.op_types)}_"
        f"{bias_scope}_trt_fp16.onnx"
    )
    return PTQPaths(
        output_dir=output_dir,
        checkpoint=Path(to_absolute_path(str(cfg.checkpoint.local_dir)))
        / str(cfg.checkpoint.filename),
        fp32=output_dir / f"{experiment}_fp32.onnx",
        fp16=output_dir / f"{experiment}_fp16.onnx",
        int8=output_dir / int8_name,
        manifest=output_dir / str(cfg.artifacts.calibration_manifest_filename),
        results=output_dir / str(cfg.artifacts.results_filename),
    )


def validate_config(cfg: DictConfig) -> None:
    flags = {name: value for name, value in cfg.steps.items()}
    invalid = {name: value for name, value in flags.items() if type(value) is not bool}
    if invalid:
        raise TypeError(f"PTQ step flags must be true or false: {invalid}")
    if str(cfg.model.name) not in {"segformer_b0", "smp"}:
        raise ValueError("PTQ supports model.name=segformer_b0 or smp")
    if int(cfg.dataset.ignore_index) in range(int(cfg.dataset.num_classes)):
        raise ValueError("PTQ expects all S5Mars classes to remain semantic")
    if str(cfg.data.calibration_selection) not in {"random", "class_aware"}:
        raise ValueError("calibration_selection must be random or class_aware")
    if int(cfg.data.max_calibration_samples) <= 0:
        raise ValueError("max_calibration_samples must be positive")
    if cfg.runtime.prefer_gpu:
        if str(cfg.int8.quant_format) != "QDQ":
            raise ValueError("The TensorRT path requires int8.quant_format=QDQ")
        if str(cfg.int8.activation_type) != "QInt8" or str(cfg.int8.weight_type) != "QInt8":
            raise ValueError("The GPU path requires signed QInt8 activations and weights")
    if cfg.int8.decompose_conv_bias and cfg.int8.quantize_bias:
        raise ValueError("decompose_conv_bias requires quantize_bias=false")
    if (
        cfg.steps.run_onnx_int8_quantization
        and not cfg.steps.run_int8_calibration_preparation
    ):
        raise ValueError(
            "run_onnx_int8_quantization=true requires "
            "run_int8_calibration_preparation=true in the same run"
        )
    method = str(cfg.int8.calibration_method)
    if method in {"Entropy", "Percentile"}:
        chunk = int(cfg.int8.calibration_chunk_size)
        samples = int(cfg.data.max_calibration_samples)
        if chunk <= 0 or samples % chunk:
            raise ValueError(
                "calibration_chunk_size must divide max_calibration_samples"
            )
