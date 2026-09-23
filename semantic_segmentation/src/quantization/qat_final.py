from __future__ import annotations

from pathlib import Path

from src.quantization.core import build_dataloader, build_dataset
from src.quantization.qat_model import evaluate_torch_model
from src.quantization.qat_storage import (
    build_drive_service,
    resolve_drive_folder,
    save_results,
    upload_file,
)
from src.quantization.qat_tensorrt import TensorRTRunner, evaluate_tensorrt


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
