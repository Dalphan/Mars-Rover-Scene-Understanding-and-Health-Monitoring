from __future__ import annotations

from src.quantization.ptq_calibration import run_int8_stages
from src.quantization.ptq_common import PTQPaths, build_paths, validate_config
from src.quantization.ptq_precision import run_precision_stages
from src.quantization.ptq_runtime import preload_tensorrt_shared_libraries, run_final_test

import torch
from omegaconf import DictConfig, OmegaConf

from src.quantization.core import (
    build_dataloader,
    build_dataset,
    ensure_checkpoint,
    load_trained_model,
    set_seed,
    sha256,
)
from src.utils.logging_utils import setup_logger


def _resolve_checkpoint(cfg, paths: PTQPaths) -> PTQPaths:
    checkpoint_path = ensure_checkpoint(
        cfg.checkpoint, experiment=str(cfg.checkpoint.experiment)
    )
    if checkpoint_path.resolve() == paths.checkpoint.resolve():
        return paths
    return PTQPaths(
        paths.output_dir,
        checkpoint_path,
        paths.fp32,
        paths.fp16,
        paths.int8,
        paths.manifest,
        paths.results,
    )


def _initial_results(cfg, paths: PTQPaths) -> dict:
    return {
        "pipeline": "onnx_ptq",
        "source_notebook": str(cfg.sync.source_notebook),
        "experiment": {
            "model": str(cfg.checkpoint.experiment),
            "input_shape": [1, 3, *list(cfg.dataset.image_size)],
            "evaluation_split": str(cfg.dataset.evaluation_split),
            "ignore_index": int(cfg.dataset.ignore_index),
        },
        "artifacts": {
            "pytorch_fp32": {
                "format": "pytorch_checkpoint",
                "precision": "fp32",
                "path": str(paths.checkpoint),
                "size_mib": paths.checkpoint.stat().st_size / 2**20,
                "sha256": sha256(paths.checkpoint),
            }
        },
        "benchmarks": {},
        "calibration": {},
    }


def _evaluation_inputs(cfg):
    needs_evaluation = any(
        (
            cfg.steps.run_fp32_baseline,
            cfg.steps.run_onnx_fp32_benchmark,
            cfg.steps.run_onnx_fp16,
            cfg.steps.run_trt_int8_probe,
            cfg.steps.run_onnx_int8_benchmark,
        )
    )
    if not needs_evaluation:
        return None, None
    dataset = build_dataset(
        cfg.dataset,
        cfg.dataset.evaluation_split,
        cfg.data.max_evaluation_samples,
        seed=cfg.seed,
    )
    loader = build_dataloader(dataset, cfg.data.batch_size, cfg.data.num_workers)
    return loader, next(iter(loader))["image"]


def run_ptq(cfg: DictConfig) -> None:
    validate_config(cfg)
    paths = build_paths(cfg)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("s5mars_onnx_ptq", paths.output_dir, "ptq.log", "INFO")
    logger.info("PTQ config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Notebook synchronization source: %s", cfg.sync.source_notebook)
    set_seed(int(cfg.seed))

    downstream_enabled = any(
        value for name, value in cfg.steps.items() if name != "run_smoke_test"
    )
    if not cfg.steps.run_smoke_test and downstream_enabled:
        raise ValueError("run_smoke_test=false requires all downstream PTQ steps=false")
    if not cfg.steps.run_smoke_test:
        logger.info("PTQ configuration validated; execution disabled")
        return

    tensorrt_handles = []
    if any(
        (
            cfg.steps.run_trt_int8_probe,
            cfg.steps.run_onnx_int8_benchmark,
            cfg.steps.run_final_test_evaluation,
        )
    ):
        _, tensorrt_handles, loaded_paths = preload_tensorrt_shared_libraries()
        logger.info("Preloaded TensorRT native libraries: %s", loaded_paths)

    paths = _resolve_checkpoint(cfg, paths)
    device = torch.device(
        "cuda" if cfg.runtime.prefer_gpu and torch.cuda.is_available() else "cpu"
    )
    model = load_trained_model(paths.checkpoint, cfg.model, cfg.dataset, device=device)
    results = _initial_results(cfg, paths)
    evaluation_loader, latency_sample = _evaluation_inputs(cfg)

    providers, fp32_session = run_precision_stages(
        cfg,
        paths,
        model,
        device,
        results,
        evaluation_loader,
        latency_sample,
    )
    int8_session = run_int8_stages(
        cfg,
        paths,
        results,
        fp32_session,
        evaluation_loader,
        latency_sample,
    )
    run_final_test(
        cfg,
        paths,
        model,
        device,
        providers,
        int8_session,
        results,
    )
    logger.debug("TensorRT library handles retained: %d", len(tensorrt_handles))
    logger.info("PTQ completed; results=%s", paths.results)
