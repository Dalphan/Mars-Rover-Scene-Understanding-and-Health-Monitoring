from __future__ import annotations

from src.data.dataloaders import build_loader
from src.train.evaluate_segmentation import evaluate
from src.train.reporting import save_json, save_prediction_grid
from src.utils.memory import release_host_memory


def _evaluate(experiment, loader, cfg, description: str):
    return evaluate(
        experiment.model,
        loader,
        experiment.criterion,
        experiment.device,
        experiment.context.num_classes,
        int(cfg.ignore_index),
        list(cfg.image_mean),
        list(cfg.image_std),
        desc=description,
        limit_batches=cfg.limits.test_batches,
        log_memory=bool(cfg.memory.log_usage),
        logger=experiment.logger,
    )


def _source_report(experiment, cfg, stats):
    context = experiment.context
    return {
        "dataset_name": context.dataset_name,
        "dataset_family": str(context.dataset_cfg.family),
        "dataset_display_name": str(context.dataset_cfg.display_name),
        "repo_id": str(context.dataset_cfg.repo_id),
        "model_run_name": context.model_run_name,
        "experiment_name": context.experiment_name,
        "sampling_tag": context.sampling_tag,
        "criterion_weight_type": str(cfg.criterion.weight_type),
        "augmentation_tag": context.augmentation_tag,
        "train_augmentation_enabled": bool(cfg.augmentation.enabled),
        "oversampling_enabled": bool(cfg.oversampling.enabled),
        "cross_dataset_evaluation_enabled": bool(
            cfg.run_cross_dataset_evaluation
        ),
        "class_names": context.class_names,
        **stats,
    }


def _cross_report(experiment, cfg, stats, loaded_checkpoint_path):
    context = experiment.context
    return {
        "evaluation_protocol": "zero_shot_cross_dataset",
        "checkpoint_selection": "source_validation_miou",
        "target_test_used_for_tuning": False,
        "source_dataset_name": context.dataset_name,
        "source_dataset_display_name": str(context.dataset_cfg.display_name),
        "source_repo_id": str(context.dataset_cfg.repo_id),
        "target_dataset_name": context.cross_dataset_name,
        "target_dataset_display_name": str(
            context.cross_dataset_cfg.display_name
        ),
        "target_repo_id": str(context.cross_dataset_cfg.repo_id),
        "target_split": str(cfg.splits.test),
        "checkpoint": (
            str(loaded_checkpoint_path)
            if loaded_checkpoint_path
            else "in_memory_model"
        ),
        "model_run_name": context.model_run_name,
        "experiment_name": context.experiment_name,
        "sampling_tag": context.sampling_tag,
        "criterion_weight_type": str(cfg.criterion.weight_type),
        "augmentation_tag": context.augmentation_tag,
        "oversampling_enabled": bool(cfg.oversampling.enabled),
        "class_names": context.class_names,
        **stats,
    }


def run_evaluation_stage(experiment, cfg, loaded_checkpoint_path):
    """Evaluate source test, optional zero-shot target test and prediction grid."""

    context = experiment.context
    test_stats = _evaluate(experiment, experiment.test_loader, cfg, "Test")
    test_report = _source_report(experiment, cfg, test_stats)
    save_json(test_report, context.output_dir / "test_metrics.json")
    experiment.logger.info("Source test metrics: %s", test_report)

    if cfg.run_cross_dataset_evaluation:
        cross_dataset, cross_loader = build_loader(
            cfg,
            str(cfg.splits.test),
            shuffle=False,
            dataset_name=context.cross_dataset_name,
            logger=experiment.logger,
        )
        description = (
            f"Zero-shot {context.dataset_name} -> {context.cross_dataset_name}"
        )
        cross_stats = _evaluate(experiment, cross_loader, cfg, description)
        cross_report = _cross_report(
            experiment, cfg, cross_stats, loaded_checkpoint_path
        )
        cross_path = context.output_dir / (
            f"cross_dataset_test_{context.dataset_name}_to_"
            f"{context.cross_dataset_name}.json"
        )
        save_json(cross_report, cross_path)
        experiment.logger.info("Cross-dataset metrics: %s", cross_report)
        del cross_loader, cross_dataset
        release_host_memory()

    if cfg.predictions.enabled:
        prediction_path = save_prediction_grid(
            experiment.model,
            experiment.test_dataset,
            experiment.device,
            cfg,
            context,
        )
        experiment.logger.info("Prediction grid saved to %s", prediction_path)
    experiment.logger.info("Outputs written to %s", context.output_dir)
