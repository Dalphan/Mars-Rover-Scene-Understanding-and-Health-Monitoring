from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig


@dataclass(frozen=True)
class QATPaths:
    output_dir: Path
    checkpoint_dir: Path
    source_checkpoint: Path
    best_qat: Path
    last_qat: Path
    history: Path
    results: Path
    onnx_int8: Path
    engine_int8: Path
    onnx_fp16: Path
    engine_fp16: Path


def build_paths(cfg: DictConfig) -> QATPaths:
    root = Path(to_absolute_path(str(cfg.output.root)))
    output_dir = root / str(cfg.model.experiment) / str(cfg.model.variant)
    checkpoint_dir = output_dir / "checkpoints"
    stem = f"{cfg.model.experiment}_{cfg.model.variant}"
    return QATPaths(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        source_checkpoint=Path(to_absolute_path(str(cfg.checkpoint.local_dir)))
        / str(cfg.checkpoint.filename),
        best_qat=checkpoint_dir / "best_modelopt_qat.pth",
        last_qat=checkpoint_dir / "last_modelopt_qat.pth",
        history=output_dir / "history.json",
        results=output_dir / "results.json",
        onnx_int8=output_dir / f"{stem}.onnx",
        engine_int8=output_dir / f"{stem}.engine",
        onnx_fp16=output_dir / f"{stem}_qat_weights_fp16.onnx",
        engine_fp16=output_dir / f"{stem}_qat_weights_fp16.engine",
    )


def validate_config(cfg: DictConfig) -> None:
    flags = {name: value for name, value in cfg.steps.items()}
    invalid = {name: value for name, value in flags.items() if type(value) is not bool}
    if invalid:
        raise TypeError(f"QAT flags must be true or false: {invalid}")
    if str(cfg.model.name) != "segformer_b0":
        raise ValueError("The synchronized QAT pipeline supports SegFormer-B0 only")
    if int(cfg.dataset.ignore_index) in range(int(cfg.dataset.num_classes)):
        raise ValueError("QAT expects all S5Mars classes to remain semantic")
    supervised = float(cfg.training.supervised_weight)
    distillation = float(cfg.training.distillation_weight)
    if abs(supervised + distillation - 1.0) >= 1e-12:
        raise ValueError("supervised_weight + distillation_weight must equal 1")
    if int(cfg.training.gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if int(cfg.training.early_stopping_patience) <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if cfg.steps.restore_saved_qat and not cfg.checkpoint.filename:
        raise ValueError("restore_saved_qat requires a source checkpoint")
