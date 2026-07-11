from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Hashable


def _validate_size(dataset_size: int, sample_size: int) -> None:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > dataset_size:
        raise ValueError(
            f"sample_size ({sample_size}) cannot exceed dataset_size ({dataset_size})"
        )


def _seeded_permutation(dataset_size: int, seed: int) -> list[int]:
    indices = list(range(dataset_size))
    random.Random(seed).shuffle(indices)
    return indices


def select_calibration_indices(
    dataset_size: int,
    sample_size: int,
    *,
    seed: int,
) -> list[int]:
    """Select a deterministic random calibration subset."""
    _validate_size(dataset_size, sample_size)
    return _seeded_permutation(dataset_size, seed)[:sample_size]


def select_nested_calibration_indices(
    dataset_size: int,
    sample_sizes: Sequence[int],
    *,
    seed: int,
) -> dict[int, list[int]]:
    """Return prefix-nested calibration subsets for controlled size ablations."""
    if not sample_sizes:
        raise ValueError("sample_sizes must not be empty")
    normalized = sorted(set(int(size) for size in sample_sizes))
    for size in normalized:
        _validate_size(dataset_size, size)
    permutation = _seeded_permutation(dataset_size, seed)
    return {size: permutation[:size] for size in normalized}


def select_class_coverage_indices(
    labels_by_index: Sequence[Iterable[Hashable]],
    sample_size: int,
    *,
    seed: int,
) -> list[int]:
    """Greedily cover observed labels, then fill the subset deterministically.

    This sampler relies only on image-level label metadata. It does not alter the
    evaluation metrics and never reads validation or test data by itself.
    """
    dataset_size = len(labels_by_index)
    _validate_size(dataset_size, sample_size)
    label_sets = [set(labels) for labels in labels_by_index]
    permutation = _seeded_permutation(dataset_size, seed)
    rank = {index: position for position, index in enumerate(permutation)}

    uncovered = set().union(*label_sets) if label_sets else set()
    remaining = set(range(dataset_size))
    selected: list[int] = []

    while uncovered and remaining and len(selected) < sample_size:
        best = max(
            remaining,
            key=lambda index: (
                len(label_sets[index] & uncovered),
                -rank[index],
            ),
        )
        if not (label_sets[best] & uncovered):
            break
        selected.append(best)
        remaining.remove(best)
        uncovered.difference_update(label_sets[best])

    for index in permutation:
        if len(selected) >= sample_size:
            break
        if index in remaining:
            selected.append(index)
            remaining.remove(index)
    return selected
