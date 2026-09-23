from __future__ import annotations

import gc

import torch

from src.quantization.core import (
    build_dataloader,
    build_dataset,
    create_onnx_session,
    evaluate_onnx,
    evaluate_torch,
)
from src.quantization.ptq_reporting import persist_results
from src.quantization.ptq_tensorrt import create_trt_session, probe_trt_int8


def run_final_test(
    cfg,
    paths,
    model,
    device,
    providers,
    int8_session,
    results,
) -> None:
    if cfg.steps.run_final_test_evaluation:
        required_artifacts = (paths.fp32, paths.fp16, paths.int8)
        missing_artifacts = [
            str(path) for path in required_artifacts if not path.is_file()
        ]
        if missing_artifacts:
            raise FileNotFoundError(
                "Final PTQ test requires the selected FP32, FP16, and INT8 "
                f"artifacts; missing={missing_artifacts}"
            )
        test_dataset = build_dataset(
            cfg.dataset,
            cfg.dataset.test_split,
            cfg.data.max_test_samples,
            seed=cfg.seed,
        )
        test_loader = build_dataloader(
            test_dataset, cfg.data.batch_size, cfg.data.num_workers
        )
        test_sample = next(iter(test_loader))["image"]
        final_test = {
            "protocol": (
                "final hold-out; never use for model, calibration, or "
                "operator selection"
            ),
            "split": str(cfg.dataset.test_split),
            "samples": len(test_dataset),
        }

        model.to(device)
        final_test["pytorch_fp32_cuda"] = evaluate_torch(
            model,
            test_loader,
            device,
            cfg.dataset.num_classes,
            cfg.dataset.ignore_index,
            cfg.dataset.test_split,
            cfg.dataset.class_names,
        )
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        test_fp32_session = create_onnx_session(paths.fp32, providers)
        final_test["onnx_fp32_cuda"] = evaluate_onnx(
            test_fp32_session,
            test_loader,
            cfg.dataset.num_classes,
            cfg.dataset.ignore_index,
            cfg.dataset.test_split,
            cfg.dataset.class_names,
        )
        test_fp16_session = create_onnx_session(paths.fp16, providers)
        final_test["onnx_fp16_cuda"] = evaluate_onnx(
            test_fp16_session,
            test_loader,
            cfg.dataset.num_classes,
            cfg.dataset.ignore_index,
            cfg.dataset.test_split,
            cfg.dataset.class_names,
        )
        del test_fp16_session
        gc.collect()

        final_test["tensorrt_int8_probe"] = probe_trt_int8(
            cfg, paths.int8, test_fp32_session, test_sample
        )
        test_int8_session = int8_session or create_trt_session(
            cfg, paths.int8
        )[0]
        final_test["onnx_int8_tensorrt"] = evaluate_onnx(
            test_int8_session,
            test_loader,
            cfg.dataset.num_classes,
            cfg.dataset.ignore_index,
            cfg.dataset.test_split,
            cfg.dataset.class_names,
        )
        results["test_metrics"] = final_test
        persist_results(results, cfg, paths)
