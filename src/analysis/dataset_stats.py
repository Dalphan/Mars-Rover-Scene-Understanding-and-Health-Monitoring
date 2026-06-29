from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def _limit(length: int, max_samples: int | None) -> int:
    return length if max_samples is None else min(length, int(max_samples))


def _class_name(class_names: dict | Any, class_id: int) -> str:
    return str(class_names.get(class_id, class_names.get(str(class_id), f"class_{class_id}")))


def summarize_hf_dataset(dataset, logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    logger.info("Summarizing dataset")
    hf_dataset = getattr(dataset, "dataset", None)
    summary = {
        "repo_id": getattr(dataset, "repo_id", None),
        "split": getattr(dataset, "split", None),
        "num_samples": len(dataset),
        "columns": list(getattr(hf_dataset, "column_names", [])),
        "features": str(getattr(hf_dataset, "features", "")),
    }
    return summary


def compute_image_level_class_distribution(
    dataset,
    class_names,
    max_samples=None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    sample_count = _limit(len(dataset), max_samples)
    logger.info("Computing image-level class distribution for %d samples", sample_count)
    counts: dict[int, int] = {}
    for index in tqdm(range(sample_count), desc="image classes"):
        mask = dataset[index]["mask"].numpy()
        for class_id in np.unique(mask):
            class_id = int(class_id)
            counts[class_id] = counts.get(class_id, 0) + 1
    rows = [
        {
            "class_id": class_id,
            "class_name": _class_name(class_names, class_id),
            "num_images": count,
            "percentage_images": count / sample_count if sample_count else 0.0,
        }
        for class_id, count in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def compute_mask_pixel_distribution(
    dataset,
    num_classes,
    class_names,
    ignore_index=0,
    max_samples=None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    sample_count = _limit(len(dataset), max_samples)
    logger.info("Computing mask pixel distribution for %d samples", sample_count)
    counts = np.zeros(int(num_classes), dtype=np.int64)
    total_pixels = 0
    for index in tqdm(range(sample_count), desc="mask pixels"):
        mask = dataset[index]["mask"].numpy().reshape(-1)
        total_pixels += int(mask.size)
        counts += np.bincount(mask, minlength=int(num_classes))[: int(num_classes)]

    ignored_pixels = int(counts[int(ignore_index)]) if 0 <= int(ignore_index) < len(counts) else 0
    valid_pixels = total_pixels - ignored_pixels
    rows = []
    for class_id, pixel_count in enumerate(counts.tolist()):
        is_ignored = class_id == int(ignore_index)
        rows.append(
            {
                "class_id": class_id,
                "class_name": _class_name(class_names, class_id),
                "pixel_count": int(pixel_count),
                "is_ignore_index": is_ignored,
                "percentage_total_pixels": pixel_count / total_pixels if total_pixels else 0.0,
                "percentage_valid_pixels": (
                    0.0 if is_ignored else pixel_count / valid_pixels if valid_pixels else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_ignore_pixel_ratio(
    dataset,
    ignore_index=0,
    max_samples=None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    sample_count = _limit(len(dataset), max_samples)
    logger.info("Computing ignore pixel ratio for %d samples", sample_count)
    ignored_pixels = 0
    total_pixels = 0
    for index in tqdm(range(sample_count), desc="ignore ratio"):
        mask = dataset[index]["mask"].numpy()
        total_pixels += int(mask.size)
        ignored_pixels += int((mask == int(ignore_index)).sum())
    return {
        "ignore_index": int(ignore_index),
        "num_samples": sample_count,
        "ignored_pixels": ignored_pixels,
        "total_pixels": total_pixels,
        "ignore_pixel_ratio": ignored_pixels / total_pixels if total_pixels else 0.0,
    }


def run_full_analysis(dataset, cfg, logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    logger.info("Starting full dataset analysis")
    results: dict[str, Any] = {"summary": summarize_hf_dataset(dataset, logger)}
    max_samples = cfg.analysis.max_samples
    class_names = dict(cfg.dataset.class_names)

    if cfg.analysis.compute_image_level_class_distribution:
        results["image_level_class_distribution"] = compute_image_level_class_distribution(
            dataset, class_names, max_samples, logger
        )
    if cfg.analysis.compute_mask_pixel_distribution:
        results["mask_pixel_distribution"] = compute_mask_pixel_distribution(
            dataset,
            cfg.dataset.num_classes,
            class_names,
            cfg.dataset.ignore_index,
            max_samples,
            logger,
        )
    if cfg.analysis.compute_ignore_pixel_ratio:
        results["ignore_pixel_ratio"] = compute_ignore_pixel_ratio(
            dataset, cfg.dataset.ignore_index, max_samples, logger
        )

    logger.info("Finished full dataset analysis")
    return results
