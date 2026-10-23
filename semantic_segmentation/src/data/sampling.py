from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


@dataclass(frozen=True)
class SamplingAudit:
    class_image_counts: np.ndarray
    image_classes: tuple[tuple[int, ...], ...]
    normalized_sample_weights: np.ndarray


def compute_image_level_sampling_weights(
    class_labels: Sequence[Sequence[int | str]],
    class_names: Mapping[int | str, str],
    num_classes: int,
    power: float,
    max_weight: float,
    excluded_class_ids: Sequence[int] = (),
) -> SamplingAudit:
    """Mirror the notebook's bounded inverse-rarity image weighting."""

    power = float(power)
    max_weight = float(max_weight)
    if power < 0:
        raise ValueError("oversampling power must be >= 0")
    if max_weight < 1:
        raise ValueError("oversampling max_weight must be >= 1")

    names = {int(key): str(value) for key, value in class_names.items()}
    if set(names) != set(range(int(num_classes))):
        raise ValueError("class_names must define every class ID exactly once")
    name_to_id = {name.strip().casefold(): class_id for class_id, name in names.items()}
    excluded = {int(class_id) for class_id in excluded_class_ids}
    invalid_excluded = excluded - set(range(int(num_classes)))
    if invalid_excluded:
        raise ValueError(f"Invalid excluded class IDs: {sorted(invalid_excluded)}")

    def resolve_class_id(label: int | str) -> int:
        if isinstance(label, (int, np.integer)):
            return int(label)
        if isinstance(label, str):
            normalized = label.strip().casefold()
            if normalized in name_to_id:
                return name_to_id[normalized]
            try:
                return int(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown class label {label!r}; expected one of "
                    f"{list(names.values())}"
                ) from exc
        raise TypeError(f"Unsupported class label type: {type(label).__name__}")

    image_classes: list[tuple[int, ...]] = []
    class_image_counts = np.zeros(int(num_classes), dtype=np.int64)
    for raw_labels in class_labels:
        labels = sorted({resolve_class_id(label) for label in raw_labels})
        invalid = set(labels) - set(range(int(num_classes)))
        if invalid:
            raise ValueError(f"Invalid class IDs in class_labels: {sorted(invalid)}")
        eligible = tuple(class_id for class_id in labels if class_id not in excluded)
        image_classes.append(eligible)
        if eligible:
            class_image_counts[list(eligible)] += 1

    present_counts = class_image_counts[class_image_counts > 0]
    if present_counts.size == 0:
        raise ValueError("No eligible classes found for oversampling")
    reference_count = float(present_counts.max())
    class_rarity = np.ones(int(num_classes), dtype=np.float64)
    present = class_image_counts > 0
    class_rarity[present] = (reference_count / class_image_counts[present]) ** power

    weights = np.asarray(
        [
            max((class_rarity[class_id] for class_id in labels), default=1.0)
            for labels in image_classes
        ],
        dtype=np.float64,
    )
    weights = np.minimum(weights, max_weight)
    weights /= weights.mean()
    return SamplingAudit(
        class_image_counts=class_image_counts,
        image_classes=tuple(image_classes),
        normalized_sample_weights=weights,
    )


def build_image_level_sampler(
    dataset,
    class_names: Mapping[int | str, str],
    num_classes: int,
    power: float,
    max_weight: float,
    num_samples_multiplier: float,
    replacement: bool,
    excluded_class_ids: Sequence[int],
    seed: int,
    logger: logging.Logger | None = None,
) -> WeightedRandomSampler:
    logger = logger or logging.getLogger(__name__)
    if float(num_samples_multiplier) <= 0:
        raise ValueError("oversampling num_samples_multiplier must be > 0")
    if type(replacement) is not bool:
        raise TypeError("oversampling replacement must be true or false")
    if not replacement:
        raise ValueError("Oversampling requires replacement=true")
    if dataset.class_labels is None:
        raise RuntimeError("Class metadata is unavailable for oversampling")

    audit = compute_image_level_sampling_weights(
        class_labels=dataset.class_labels,
        class_names=class_names,
        num_classes=num_classes,
        power=power,
        max_weight=max_weight,
        excluded_class_ids=excluded_class_ids,
    )
    num_samples = max(1, round(len(dataset) * float(num_samples_multiplier)))
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            audit.normalized_sample_weights, dtype=torch.double
        ),
        num_samples=num_samples,
        replacement=True,
        generator=torch.Generator().manual_seed(int(seed)),
    )

    probabilities = (
        audit.normalized_sample_weights / audit.normalized_sample_weights.sum()
    )
    excluded = {int(value) for value in excluded_class_ids}
    names = {int(key): str(value) for key, value in class_names.items()}
    logger.info(
        "Oversampling enabled: draws=%d power=%.3f max_weight=%.3f "
        "normalized_weight_range=[%.3f, %.3f]",
        num_samples,
        float(power),
        float(max_weight),
        float(audit.normalized_sample_weights.min()),
        float(audit.normalized_sample_weights.max()),
    )
    for class_id in range(int(num_classes)):
        count = int(audit.class_image_counts[class_id])
        if class_id in excluded or count == 0:
            continue
        before = count / len(dataset)
        after = sum(
            probabilities[index]
            for index, labels in enumerate(audit.image_classes)
            if class_id in labels
        )
        logger.info(
            "Class presence %d:%s before=%.2f%% expected_after=%.2f%%",
            class_id,
            names[class_id],
            before * 100,
            after * 100,
        )
    return sampler
