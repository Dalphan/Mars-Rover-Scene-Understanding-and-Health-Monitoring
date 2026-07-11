from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PTQConfig:
    calibration_size: int = 128
    calibration_seed: int = 42
    calibration_strategy: str = "random"
    calibration_batch_size: int = 4
    benchmark_batch_size: int = 1
    warmup_iterations: int = 100
    measurement_iterations: int = 1000

    def validate(self) -> None:
        if self.calibration_size <= 0:
            raise ValueError("calibration_size must be positive")
        if self.calibration_batch_size <= 0:
            raise ValueError("calibration_batch_size must be positive")
        if self.calibration_strategy not in {"random", "class_coverage"}:
            raise ValueError(
                "calibration_strategy must be one of: random, class_coverage"
            )
        if self.benchmark_batch_size != 1:
            raise ValueError("the PTQ study defines benchmark_batch_size=1")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be positive")


@dataclass(frozen=True)
class QATConfig:
    mode: str = "none"
    source_checkpoint: str | None = None
    calibration_size: int = 128
    calibration_seed: int = 42
    epochs: int = 5
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    freeze: str = "none"

    def validate(self) -> None:
        if self.mode not in {"none", "qat_int8"}:
            raise ValueError("mode must be one of: none, qat_int8")
        if self.mode == "none":
            return
        if not self.source_checkpoint:
            raise ValueError("source_checkpoint is required for qat_int8")
        if self.calibration_size <= 0:
            raise ValueError("calibration_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.freeze not in {"none", "encoder", "classifier"}:
            raise ValueError("freeze must be one of: none, encoder, classifier")
