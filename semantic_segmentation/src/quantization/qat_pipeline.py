from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from src.quantization.core import set_seed
from src.quantization.qat_config import build_paths, validate_config
from src.quantization.qat_deployment import run_qat_deployment
from src.quantization.qat_final import finalize_qat
from src.quantization.qat_setup import prepare_qat_experiment
from src.quantization.qat_stage import run_qat_stage
from src.utils.logging_utils import setup_logger


def run_qat(cfg: DictConfig) -> None:
    """Run the notebook-synchronized QAT stages from a Hydra config."""

    validate_config(cfg)
    paths = build_paths(cfg)
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("s5mars_modelopt_qat", paths.output_dir, "qat.log", "INFO")
    logger.info("QAT config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Notebook synchronization source: %s", cfg.sync.source_notebook)
    if not torch.cuda.is_available():
        raise RuntimeError("The synchronized ModelOpt/TensorRT QAT pipeline requires CUDA")

    set_seed(int(cfg.seed))
    device = torch.device(f"cuda:{int(cfg.tensorrt.device_id)}")
    experiment = prepare_qat_experiment(cfg, paths, device, logger)
    tensor_quantizer_type, quantizer_audit = run_qat_stage(
        experiment, cfg, device
    )
    trt_runner = run_qat_deployment(
        experiment, tensor_quantizer_type, quantizer_audit, cfg, device
    )
    finalize_qat(experiment, trt_runner, cfg, device, logger)
    logger.info("QAT completed; results=%s", experiment.paths.results)
