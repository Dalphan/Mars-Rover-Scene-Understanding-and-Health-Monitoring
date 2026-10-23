from __future__ import annotations

from src.quantization.qat_runtime import TensorRTRunner, evaluate_tensorrt
from src.quantization.qat_training import (
    CombinedLoss,
    evaluate_torch_model,
    load_fp32_model,
    prepare_selective_qat,
    train_qat,
)

import copy
import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from torch.utils.data import Subset

from src.quantization.core import build_dataloader, build_dataset, ensure_checkpoint
from src.quantization.qat_config import QATPaths
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

from src.quantization.qat_storage import save_results


def run_qat_stage(experiment, cfg, device):
    """Prepare selective quantizers, run QAT and update training reports."""

    student, quantizer_audit, tensor_quantizer_type, modelopt = (
        prepare_selective_qat(
            experiment.student,
            experiment.calibration_loader,
            cfg,
            device,
            experiment.paths.best_qat,
        )
    )
    results = experiment.results
    results["experiment"]["modelopt_quantizer_audit"] = quantizer_audit
    initialized_metrics = None
    if not cfg.steps.restore_saved_qat:
        initialized_metrics = evaluate_torch_model(
            student, experiment.val_loader, device, cfg, cfg.dataset.val_split
        )
        results["benchmarks"]["pytorch_modelopt_initialized_fake_quant"] = {
            "runtime": "pytorch",
            "precision": "fake_quant_int8_conv_mlp_decoder_linear",
            "metrics": initialized_metrics,
        }
    save_results(results, cfg, experiment.paths)

    student, history, best_miou, _ = train_qat(
        experiment.teacher,
        student,
        experiment.train_loader,
        experiment.val_loader,
        experiment.criterion,
        modelopt,
        experiment.paths,
        cfg,
        device,
    )
    experiment.student = student
    qat_metrics = evaluate_torch_model(
        student, experiment.val_loader, device, cfg, cfg.dataset.val_split
    )
    results["benchmarks"]["pytorch_modelopt_qat_fake_quant"] = {
        "runtime": "pytorch",
        "precision": "fake_quant_int8_conv_mlp_decoder_linear_qat",
        "metrics": qat_metrics,
    }
    if cfg.steps.restore_saved_qat:
        results["training"]["runtime_comparison_restore"] = {
            "restored_without_retraining": True,
            "best_val_miou_recomputed": best_miou,
            "best_checkpoint": str(experiment.paths.best_qat),
        }
    else:
        results["training"] = {
            "epochs_completed": len(history),
            "best_epoch": max(history, key=lambda row: row["val_miou"])["epoch"],
            "best_val_miou": best_miou,
            "history": history,
            "best_checkpoint": str(experiment.paths.best_qat),
            "pre_post_qat_validation": {
                "split": str(cfg.dataset.val_split),
                "before_qat": initialized_metrics,
                "after_qat": qat_metrics,
                "delta_after_minus_before": {
                    "pixel_accuracy": (
                        qat_metrics["pixel_accuracy"]
                        - initialized_metrics["pixel_accuracy"]
                    ),
                    "miou": qat_metrics["miou"] - initialized_metrics["miou"],
                },
            },
        }
    save_results(results, cfg, experiment.paths)
    return tensor_quantizer_type, quantizer_audit

from pathlib import Path

from src.quantization.core import build_dataloader, build_dataset
from src.quantization.qat_storage import (
    build_drive_service,
    resolve_drive_folder,
    save_results,
    upload_file,
)


def _run_final_test(experiment, trt_runner, cfg, device):
    if not cfg.steps.run_final_test:
        return
    test_dataset = build_dataset(
        cfg.dataset,
        cfg.dataset.test_split,
        cfg.training.max_test_samples,
        seed=cfg.seed,
    )
    test_loader = build_dataloader(
        test_dataset,
        cfg.training.eval_batch_size,
        cfg.training.num_workers,
        seed=cfg.seed,
    )
    experiment.teacher.to(device).eval()
    experiment.student.to(device).eval()
    final_test = {
        "protocol": "final hold-out; never use for QAT configuration selection",
        "split": str(cfg.dataset.test_split),
        "samples": len(test_dataset),
        "pytorch_fp32_cuda": evaluate_torch_model(
            experiment.teacher, test_loader, device, cfg, cfg.dataset.test_split
        ),
        "pytorch_modelopt_qat_fake_quant": evaluate_torch_model(
            experiment.student, test_loader, device, cfg, cfg.dataset.test_split
        ),
        "onnx_modelopt_qat_int8_tensorrt": evaluate_tensorrt(
            trt_runner, test_loader, cfg, cfg.dataset.test_split
        ),
    }
    results = experiment.results
    results["comparisons"]["vs_segformer_qat_conv_only"][
        "test_miou_delta_current_minus_conv_only"
    ] = float(
        final_test["onnx_modelopt_qat_int8_tensorrt"]["miou"]
        - float(cfg.reference.segformer_qat_conv_only.test_miou)
    )
    if cfg.steps.run_trt_fp16_baseline:
        fp16_runner = TensorRTRunner(
            experiment.paths.engine_fp16, cfg.tensorrt.device_id
        )
        final_test["onnx_modelopt_qat_weights_fp16_tensorrt"] = evaluate_tensorrt(
            fp16_runner, test_loader, cfg, cfg.dataset.test_split
        )
        del fp16_runner
        results["comparisons"]["int8_vs_fp16_tensorrt"][
            "test_miou_delta_int8_minus_fp16"
        ] = float(
            final_test["onnx_modelopt_qat_int8_tensorrt"]["miou"]
            - final_test["onnx_modelopt_qat_weights_fp16_tensorrt"]["miou"]
        )
    results["test_metrics"] = final_test


def _upload_final_artifacts(experiment, cfg, logger):
    if not cfg.steps.upload_final_artifacts:
        return
    try:
        service = build_drive_service()
        folder_id = resolve_drive_folder(service, cfg)
        candidates = (
            experiment.paths.best_qat,
            experiment.paths.last_qat,
            experiment.paths.history,
            experiment.paths.results,
            experiment.paths.onnx_int8,
            experiment.paths.onnx_fp16
            if cfg.steps.run_trt_fp16_baseline
            else None,
        )
        for path in candidates:
            if path is not None and Path(path).is_file():
                upload_file(Path(path), cfg, service, folder_id)
    except Exception as error:
        logger.warning("Final Drive upload deferred: %s", error)


def finalize_qat(experiment, trt_runner, cfg, device, logger):
    """Evaluate the untouched test split, persist reports and upload artifacts."""

    _run_final_test(experiment, trt_runner, cfg, device)
    save_results(experiment.results, cfg, experiment.paths)
    _upload_final_artifacts(experiment, cfg, logger)
