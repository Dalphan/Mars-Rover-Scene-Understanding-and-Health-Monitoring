from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.train.checkpoint_stage import prepare_evaluation_checkpoint
from src.train.evaluation_stage import run_evaluation_stage
from src.train.experiment import build_training_experiment
from src.train.run_context import build_run_context, resolve_device, validate_config
from src.train.training_stage import run_training_stage
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def run_training(cfg: DictConfig) -> None:
    """Run the notebook-synchronized segmentation training stages."""

    validate_config(cfg)
    context = build_run_context(cfg)
    # These dataset values override reusable model-preset fallbacks at runtime.
    with open_dict(cfg):
        cfg.model.num_classes = context.num_classes
        cfg.model.ignore_index = int(cfg.ignore_index)

    context.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        "mars_semantic_segmentation_training",
        context.output_dir,
        cfg.logging.filename if cfg.logging.log_to_file else "",
        str(cfg.logging.level),
    )
    logger.info("Training config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Notebook synchronization source: %s", cfg.sync.source_notebook)
    torch.multiprocessing.set_sharing_strategy(
        str(cfg.runtime.torch_multiprocessing_sharing_strategy)
    )
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.runtime.device))
    logger.info("Using device=%s", device)

    experiment = build_training_experiment(cfg, context, device, logger)
    training_state = run_training_stage(experiment, cfg)
    checkpoint_path = prepare_evaluation_checkpoint(
        experiment, training_state, cfg
    )
    run_evaluation_stage(experiment, cfg, checkpoint_path)
