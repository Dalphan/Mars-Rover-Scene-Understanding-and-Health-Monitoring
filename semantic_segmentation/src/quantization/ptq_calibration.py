from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.quantization.core import save_json

class CalibrationReader:
    def __init__(self, dataset, input_name: str) -> None:
        self.dataset = dataset
        self.input_name = str(input_name)
        self.start_index = 0
        self.end_index = len(dataset)
        self.next_index = 0

    def __len__(self):
        return len(self.dataset)

    def set_range(self, start_index, end_index):
        start_index, end_index = int(start_index), int(end_index)
        if not 0 <= start_index < end_index <= len(self.dataset):
            raise ValueError(f"Invalid calibration range {start_index}:{end_index}")
        self.start_index, self.end_index = start_index, end_index
        self.next_index = start_index

    def get_next(self):
        if self.next_index >= self.end_index:
            return None
        sample = self.dataset[self.next_index]
        self.next_index += 1
        images = np.ascontiguousarray(
            sample["image"].unsqueeze(0).numpy(), dtype=np.float32
        )
        return {self.input_name: images}

    def rewind(self):
        self.start_index = 0
        self.end_index = len(self.dataset)
        self.next_index = 0


def verify_calibration_reader(reader, session, cfg: DictConfig) -> dict:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_input = (1, 3, *tuple(cfg.dataset.image_size))
    expected_output = (
        1,
        int(cfg.dataset.num_classes),
        *tuple(cfg.dataset.image_size),
    )
    sample_count = inference_runs = 0
    observed_min, observed_max = float("inf"), float("-inf")
    reader.rewind()
    while True:
        model_inputs = reader.get_next()
        if model_inputs is None:
            break
        images = model_inputs[input_name]
        if images.dtype != np.float32 or tuple(images.shape) != expected_input:
            raise ValueError(
                f"Invalid calibration tensor dtype/shape: {images.dtype}, {images.shape}"
            )
        if not np.isfinite(images).all():
            raise ValueError("Calibration input contains NaN or infinity")
        sample_count += 1
        observed_min = min(observed_min, float(images.min()))
        observed_max = max(observed_max, float(images.max()))
        if inference_runs < int(cfg.data.calibration_dry_run_samples):
            logits = session.run([output_name], model_inputs)[0]
            if tuple(logits.shape) != expected_output or not np.isfinite(logits).all():
                raise ValueError("Calibration dry-run returned invalid logits")
            inference_runs += 1
    if sample_count != len(reader):
        raise ValueError(f"Calibration reader yielded {sample_count}/{len(reader)} samples")
    reader.rewind()
    return {
        "status": "passed",
        "samples": sample_count,
        "input_name": input_name,
        "input_dtype": "float32",
        "input_shape": list(expected_input),
        "normalized_value_min": observed_min,
        "normalized_value_max": observed_max,
        "dry_run_samples": inference_runs,
        "dry_run_output_shape": list(expected_output),
        "all_inputs_finite": True,
        "all_dry_run_outputs_finite": True,
        "rewind_ok": True,
    }


def class_presence(dataset, class_names) -> tuple[dict, list, list]:
    names = {int(key): str(value) for key, value in class_names.items()}
    name_to_id = {name: class_id for class_id, name in names.items()}
    counts = {name: 0 for name in names.values()}
    unknown = set()
    for labels in dataset.dataset["class_labels"]:
        ids = set()
        for value in labels:
            if isinstance(value, str) and value in name_to_id:
                ids.add(name_to_id[value])
                continue
            try:
                class_id = int(value)
            except (TypeError, ValueError):
                unknown.add(str(value))
                continue
            if class_id in names:
                ids.add(class_id)
            else:
                unknown.add(str(value))
        for class_id in ids:
            counts[names[class_id]] += 1
    return counts, [name for name, count in counts.items() if count == 0], sorted(unknown)


def create_calibration_manifest(dataset, reader, verification, cfg, path: Path):
    counts, missing, unknown = class_presence(dataset, cfg.dataset.class_names)
    selected = dataset.selected_source_indices
    unique = len(set(selected)) == len(selected)
    in_range = all(0 <= index < dataset.source_sample_count for index in selected)
    if not unique or not in_range:
        raise ValueError("Calibration source indices are invalid")
    manifest = {
        "dataset_repo": str(cfg.dataset.repo_id),
        "split": str(cfg.dataset.calibration_split),
        "source_dataset_fingerprint": dataset.source_fingerprint,
        "source_sample_count": dataset.source_sample_count,
        "subset_fingerprint": dataset.subset_fingerprint,
        "selection": str(cfg.data.calibration_selection),
        "seed": int(cfg.seed),
        "sample_count": len(dataset),
        "batch_size": 1,
        "input_name": reader.input_name,
        "input_dtype": "float32",
        "input_shape": [1, 3, *list(cfg.dataset.image_size)],
        "normalization_mean": list(cfg.dataset.image_mean),
        "normalization_std": list(cfg.dataset.image_std),
        "masks_used_for_calibration": False,
        "class_presence_unit": "images, not pixels",
        "class_presence_counts": counts,
        "missing_classes": missing,
        "unknown_class_labels": unknown,
        "selected_indices_unique": unique,
        "selected_indices_in_source_range": in_range,
        "selected_source_indices": selected,
        "verification": verification,
    }
    save_json(manifest, path)
    return manifest
