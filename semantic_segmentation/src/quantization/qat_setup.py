from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from torch.utils.data import Subset

from src.quantization.core import build_dataloader, build_dataset, ensure_checkpoint
from src.quantization.qat_config import QATPaths
from src.quantization.qat_model import CombinedLoss, evaluate_torch_model, load_fp32_model
from src.quantization.qat_storage import download_output_artifact


@dataclass
class QATExperiment:
    """Runtime state shared by the small QAT pipeline stages."""

    paths: QATPaths
    train_loader: Any
    val_loader: Any
    calibration_loader: Any
    teacher: Any
    student: Any
    criterion: Any
    results: dict


def _restore_previous_results(paths: QATPaths, cfg, logger):
    if not cfg.steps.restore_saved_qat:
        return None
    download_output_artifact(paths.best_qat.name, paths.best_qat, cfg)
    try:
        download_output_artifact(paths.results.name, paths.results, cfg)
        return json.loads(paths.results.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Previous QAT results were not found; creating a fresh report")
        return None


def _build_loaders(cfg):
    train_dataset = build_dataset(
        cfg.dataset,
        cfg.dataset.train_split,
        cfg.training.max_train_samples,
        seed=cfg.seed,
    )
    val_dataset = build_dataset(
        cfg.dataset,
        cfg.dataset.val_split,
        cfg.training.max_val_samples,
        seed=cfg.seed,
    )
    train_loader = build_dataloader(
        train_dataset,
        cfg.training.train_batch_size,
        cfg.training.num_workers,
        shuffle=True,
        seed=cfg.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        cfg.training.eval_batch_size,
        cfg.training.num_workers,
        seed=cfg.seed,
    )
    rng = np.random.default_rng(int(cfg.seed))
    calibration_indices = rng.permutation(len(train_dataset))[
        : min(int(cfg.training.calibration_samples), len(train_dataset))
    ]
    calibration_dataset = Subset(train_dataset, calibration_indices.tolist())
    calibration_loader = build_dataloader(
        calibration_dataset,
        cfg.training.calibration_batch_size,
        cfg.training.num_workers,
        seed=cfg.seed,
    )
    return train_loader, val_loader, calibration_loader, len(calibration_dataset)


def _fresh_results(cfg, calibration_samples: int) -> dict:
    return {
        "experiment": {
            "method": "nvidia_modelopt_qat",
            "model": str(cfg.model.experiment),
            "quantization": (
                "INT8 Conv2d + encoder MLP Linear + decoder projection Linear; "
                "attention and other operators FP16"
            ),
            "variant": str(cfg.model.variant),
            "fp16_fallback_module_prefixes": list(
                cfg.model.fp16_fallback_module_prefixes
            ),
            "calibration_samples": calibration_samples,
            "no_oversampling": True,
            "no_augmentation": True,
            "test_is_final_holdout": True,
        },
        "training": {},
        "artifacts": {},
        "benchmarks": {},
        "comparisons": {},
    }


def prepare_qat_experiment(
    cfg, paths: QATPaths, device, logger
) -> QATExperiment:
    """Resolve artifacts, data and models before the QAT stages start."""

    source_checkpoint = ensure_checkpoint(cfg.checkpoint)
    if source_checkpoint.resolve() != paths.source_checkpoint.resolve():
        paths = replace(paths, source_checkpoint=source_checkpoint)
    restored_results = _restore_previous_results(paths, cfg, logger)
    train_loader, val_loader, calibration_loader, calibration_samples = (
        _build_loaders(cfg)
    )

    teacher = load_fp32_model(source_checkpoint, cfg).to(device).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(teacher).to(device)
    student.requires_grad_(True)
    results_template = _fresh_results(cfg, calibration_samples)
    results = restored_results or results_template
    results.setdefault("experiment", {}).update(results_template["experiment"])
    for section in ("training", "artifacts", "benchmarks", "comparisons"):
        results.setdefault(section, {})
    results["benchmarks"]["pytorch_fp32_cuda"] = {
        "runtime": "pytorch",
        "precision": "fp32",
        "metrics": evaluate_torch_model(
            teacher, val_loader, device, cfg, cfg.dataset.val_split
        ),
    }
    return QATExperiment(
        paths=paths,
        train_loader=train_loader,
        val_loader=val_loader,
        calibration_loader=calibration_loader,
        teacher=teacher,
        student=student,
        criterion=CombinedLoss(cfg),
        results=results,
    )
