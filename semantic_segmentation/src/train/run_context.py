from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.data.dataloaders import (
    resolve_dataset_config,
    validate_cross_dataset_configuration,
)

@dataclass(frozen=True)
class RunContext:
    dataset_name: str
    dataset_cfg: object
    cross_dataset_name: str | None
    cross_dataset_cfg: object | None
    class_names: dict[int, str]
    num_classes: int
    model_run_name: str
    sampling_tag: str
    augmentation_tag: str
    experiment_name: str
    output_dir: Path
    checkpoint_name: str
    checkpoint_path: Path
    drive_folder_id: str | None


def _model_run_name(cfg: DictConfig) -> str:
    name = str(cfg.model.name)
    if name == "segformer_b0":
        return name
    if name == "smp":
        return f"smp_{cfg.model.architecture}_{cfg.model.encoder_name}"
    if name == "lcnet":
        return str(cfg.model.variant)
    raise ValueError("model.name must be one of: segformer_b0, smp, lcnet")


def validate_config(cfg: DictConfig) -> None:
    boolean_options = {
        "run_cross_dataset_evaluation": cfg.run_cross_dataset_evaluation,
        "oversampling.enabled": cfg.oversampling.enabled,
        "augmentation.enabled": cfg.augmentation.enabled,
        "execution.skip_train": cfg.execution.skip_train,
        "execution.load_checkpoint": cfg.execution.load_checkpoint,
        "execution.load_best": cfg.execution.load_best,
        "checkpoint.save_best": cfg.checkpoint.save_best,
    }
    invalid = {
        name: value for name, value in boolean_options.items() if type(value) is not bool
    }
    if invalid:
        raise TypeError(f"These configuration options must be true or false: {invalid}")
    if cfg.execution.skip_train and not cfg.execution.load_checkpoint:
        raise ValueError("execution.skip_train=true requires load_checkpoint=true")
    if (
        not cfg.execution.skip_train
        and cfg.execution.load_checkpoint
        and cfg.execution.load_best
        and not cfg.checkpoint.save_best
    ):
        raise ValueError(
            "checkpoint.save_best=false is incompatible with "
            "execution.load_best=true during training"
        )
    if str(cfg.optimizer.name).lower() != "adamw":
        raise ValueError("Only optimizer.name=adamw is synchronized")
    if str(cfg.scheduler.name).lower() != "cosine":
        raise ValueError("Only scheduler.name=cosine is synchronized")
    if int(cfg.epochs) <= 0 or int(cfg.batch_size) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if len(cfg.image_size) != 2 or min(int(value) for value in cfg.image_size) <= 0:
        raise ValueError("image_size must contain two positive integers")

    _, dataset_cfg = resolve_dataset_config(cfg)
    class_names = {int(key): str(value) for key, value in dataset_cfg.class_names.items()}
    if set(class_names) != set(range(int(dataset_cfg.num_classes))):
        raise ValueError("The selected dataset taxonomy is not contiguous from class 0")
    if int(cfg.ignore_index) in class_names:
        raise ValueError(
            "The synchronized pipeline treats every annotated class as semantic; "
            "ignore_index must remain outside the class range"
        )
    validate_cross_dataset_configuration(cfg)

    strategy = str(cfg.runtime.torch_multiprocessing_sharing_strategy)
    if strategy not in torch.multiprocessing.get_all_sharing_strategies():
        raise ValueError(f"Unsupported torch sharing strategy: {strategy}")
    if cfg.runtime.enforce_torch_version:
        actual = torch.__version__.split("+", maxsplit=1)[0]
        expected = str(cfg.runtime.expected_torch_version)
        if actual != expected:
            raise RuntimeError(f"Expected PyTorch {expected}, got {torch.__version__}")


def build_run_context(cfg: DictConfig) -> RunContext:
    dataset_name, dataset_cfg = resolve_dataset_config(cfg)
    cross = validate_cross_dataset_configuration(cfg)
    cross_name, cross_cfg = cross if cross is not None else (None, None)
    class_names = {int(key): str(value) for key, value in dataset_cfg.class_names.items()}
    model_run_name = _model_run_name(cfg)
    sampling_tag = "oversampling" if cfg.oversampling.enabled else "standard_sampling"
    augmentation_tag = "marsbench_aug" if cfg.augmentation.enabled else "no_aug"
    experiment_name = (
        f"{model_run_name}_{sampling_tag}_{cfg.criterion.weight_type}_"
        f"{augmentation_tag}"
    )
    run_name = f"{experiment_name}_{dataset_name}"
    configured_output = cfg.checkpoint.output_dir
    if configured_output:
        output_dir = Path(to_absolute_path(str(configured_output)))
    else:
        output_dir = Path(to_absolute_path(str(cfg.checkpoint.output_root))) / run_name
    checkpoint_name = "best.ckpt" if cfg.execution.load_best else "last.ckpt"
    folder_id = cfg.checkpoint.drive_folder_id_override
    if not folder_id:
        folder_id = cfg.checkpoint.drive_folder_ids.get(run_name)
    return RunContext(
        dataset_name=dataset_name,
        dataset_cfg=dataset_cfg,
        cross_dataset_name=cross_name,
        cross_dataset_cfg=cross_cfg,
        class_names=class_names,
        num_classes=int(dataset_cfg.num_classes),
        model_run_name=model_run_name,
        sampling_tag=sampling_tag,
        augmentation_tag=augmentation_tag,
        experiment_name=experiment_name,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_path=output_dir / checkpoint_name,
        drive_folder_id=str(folder_id) if folder_id else None,
    )


def resolve_device(configured: str) -> torch.device:
    configured = str(configured).lower()
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(configured)
