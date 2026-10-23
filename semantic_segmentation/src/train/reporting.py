from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from src.data.transforms import prepare_batch
from src.train.run_context import RunContext

def checkpoint_metadata(cfg: DictConfig, context: RunContext) -> dict:
    dataset_cfg = context.dataset_cfg
    return {
        "dataset_name": context.dataset_name,
        "dataset_family": str(dataset_cfg.family),
        "dataset_display_name": str(dataset_cfg.display_name),
        "repo_id": str(dataset_cfg.repo_id),
        "run_cross_dataset_evaluation": bool(
            cfg.run_cross_dataset_evaluation
        ),
        "cross_dataset_name": context.cross_dataset_name,
        "train_split": str(cfg.splits.train),
        "val_split": str(cfg.splits.val),
        "test_split": str(cfg.splits.test),
        "class_names": context.class_names,
        "expected_split_sizes": dict(dataset_cfg.expected_split_sizes),
        "expected_torch_version": str(cfg.runtime.expected_torch_version),
        "actual_torch_version": torch.__version__,
        "seed": int(cfg.seed),
        "model_name": str(cfg.model.name),
        "model_run_name": context.model_run_name,
        "pretrained_name": getattr(cfg.model, "pretrained_name", None),
        "smp_architecture": getattr(cfg.model, "architecture", None),
        "smp_encoder_name": getattr(cfg.model, "encoder_name", None),
        "smp_encoder_weights": getattr(cfg.model, "encoder_weights", None),
        "num_classes": context.num_classes,
        "ignore_index": int(cfg.ignore_index),
        "image_size": list(cfg.image_size),
        "epochs": int(cfg.epochs),
        "batch_size": int(cfg.batch_size),
        "num_workers": cfg.dataloader.train_num_workers,
        "eval_num_workers": cfg.dataloader.eval_num_workers,
        "dataloader_pin_memory": bool(cfg.dataloader.pin_memory),
        "dataloader_prefetch_factor": int(cfg.dataloader.prefetch_factor),
        "dataloader_persistent_workers": bool(
            cfg.dataloader.persistent_workers
        ),
        "torch_multiprocessing_sharing_strategy": str(
            cfg.runtime.torch_multiprocessing_sharing_strategy
        ),
        "dataloader_transport": "numpy_uint8",
        "log_memory_usage": bool(cfg.memory.log_usage),
        "memory_log_interval_batches": int(
            cfg.memory.log_interval_batches or 0
        ),
        "freeze": str(cfg.freeze),
        "lr": float(cfg.optimizer.lr),
        "weight_decay": float(cfg.optimizer.weight_decay),
        "criterion_name": str(cfg.criterion.name),
        "criterion_alpha": float(cfg.criterion.alpha),
        "criterion_weight_type": str(cfg.criterion.weight_type),
        "criterion_smooth": float(cfg.criterion.smooth),
        "augmentation_tag": context.augmentation_tag,
        "train_augmentation_enabled": bool(cfg.augmentation.enabled),
        "train_horizontal_flip_prob": float(
            cfg.augmentation.horizontal_flip_prob
        ),
        "train_vertical_flip_prob": float(cfg.augmentation.vertical_flip_prob),
        "train_random_rotate90_prob": float(
            cfg.augmentation.random_rotate90_prob
        ),
        "sampling_tag": context.sampling_tag,
        "oversampling_enabled": bool(cfg.oversampling.enabled),
        "oversampling_power": float(cfg.oversampling.power),
        "oversampling_max_weight": float(cfg.oversampling.max_weight),
        "oversampling_num_samples_multiplier": float(
            cfg.oversampling.num_samples_multiplier
        ),
        "oversampling_replacement": bool(cfg.oversampling.replacement),
        "oversampling_excluded_class_ids": list(
            cfg.oversampling.excluded_class_ids
        ),
    }


def checkpoint_expectations(context: RunContext) -> dict:
    return {
        "dataset_name": context.dataset_name,
        "repo_id": str(context.dataset_cfg.repo_id),
        "model_run_name": context.model_run_name,
        "num_classes": context.num_classes,
        "sampling_tag": context.sampling_tag,
        "criterion_weight_type": None,
        "augmentation_tag": context.augmentation_tag,
    }


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


@torch.no_grad()
def save_prediction_grid(
    model,
    dataset,
    device,
    cfg: DictConfig,
    context: RunContext,
) -> Path | None:
    count = min(int(cfg.predictions.num_samples), len(dataset))
    if count <= 0:
        return None
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    base_palette = [
        (30, 30, 30),
        (166, 118, 29),
        (0, 109, 119),
        (131, 56, 236),
        (230, 57, 70),
        (255, 183, 3),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
    ]
    palette = {
        class_id: base_palette[class_id % len(base_palette)]
        for class_id in context.class_names
    }

    def colorize(mask):
        color = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, rgb in palette.items():
            color[mask == class_id] = rgb
        return color

    model.eval()
    fig, axes = plt.subplots(count, 4, figsize=(18, 4 * count))
    if count == 1:
        axes = np.expand_dims(axes, 0)
    for row in range(count):
        sample = dataset[row]
        images, _ = prepare_batch(
            {
                "image": np.expand_dims(sample["image"], axis=0),
                "mask": np.expand_dims(sample["mask"], axis=0),
            },
            device,
            list(cfg.image_mean),
            list(cfg.image_std),
        )
        pred = torch.argmax(model(images), dim=1).squeeze(0).cpu().numpy()
        image = sample["image"].astype(np.float32) / 255.0
        ground_truth = colorize(sample["mask"])
        prediction = colorize(pred)
        overlay = (
            (1.0 - float(cfg.predictions.alpha)) * (image * 255)
            + float(cfg.predictions.alpha) * prediction
        ).astype(np.uint8)
        for axis, value, title in zip(
            axes[row],
            (image, ground_truth, prediction, overlay),
            ("Image", "Ground truth", "Prediction", "Prediction overlay"),
        ):
            axis.imshow(value)
            axis.set_title(title)
            axis.axis("off")
        del images

    handles = [
        Patch(
            facecolor=tuple(channel / 255 for channel in palette[class_id]),
            edgecolor="black",
            label=f"{class_id}: {context.class_names[class_id]}",
        )
        for class_id in sorted(context.class_names)
    ]
    fig.legend(handles=handles, loc="center right", title="Classes", frameon=True)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    path = context.output_dir / "prediction_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
