from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch

from .curiosity_wheel_dataset import CuriosityWheelDataset
from .preprocessing import PreprocessingConfig, WheelPreprocessor


def audit_preprocessing(
    root: str | Path,
    config: PreprocessingConfig,
    *,
    split: str = "train",
    indices: Sequence[int] | None = None,
    camera_poses: Sequence[str] | None = None,
    variants: int = 2,
    seed: int = 42,
) -> plt.Figure:
    """Plot raw samples beside reproducible stochastic preprocessing variants."""
    if variants < 1:
        raise ValueError("variants must be at least 1")

    dataset = CuriosityWheelDataset(root, split, camera_poses=camera_poses)
    selected = list(indices) if indices is not None else list(range(min(4, len(dataset))))
    if not selected:
        raise ValueError("indices must select at least one sample")
    if any(index < 0 or index >= len(dataset) for index in selected):
        raise IndexError("audit sample index is outside the dataset")

    preprocessor = WheelPreprocessor(config)
    raw_preprocessor = WheelPreprocessor()
    figure, axes = plt.subplots(
        len(selected),
        variants + 1,
        figsize=(4.5 * (variants + 1), 3.5 * len(selected)),
        squeeze=False,
    )

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for row_index, sample_index in enumerate(selected):
            sample = dataset[sample_index]
            raw_image = sample["image"]
            target_mask = sample["target_mask"]
            anomaly_mask = sample["anomaly_mask"]
            image_id = sample["metadata"]["image_id"]

            axes[row_index, 0].imshow(
                raw_preprocessor.image_for_display(raw_image).permute(1, 2, 0)
            )
            axes[row_index, 0].set_title(f"Originale\n{image_id}")

            for variant in range(variants):
                processed, _, _ = preprocessor(
                    raw_image.clone(),
                    target_mask.clone(),
                    anomaly_mask.clone(),
                    camera_pose=sample["metadata"].get("camera_pose"),
                )
                axes[row_index, variant + 1].imshow(
                    preprocessor.image_for_display(processed).permute(1, 2, 0)
                )
                axes[row_index, variant + 1].set_title(f"Augmentata {variant + 1}")

            for axis in axes[row_index]:
                axis.axis("off")

    figure.suptitle(f"Audit preprocessing — split {split}")
    figure.tight_layout()
    return figure
