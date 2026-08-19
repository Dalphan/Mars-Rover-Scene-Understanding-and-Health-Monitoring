"""Pure-Python contract and deterministic split builder for dataset v1."""

from __future__ import annotations

import hashlib
import json
import itertools
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


SCHEMA_VERSION = 1
UNIT_TYPES = {"standalone_clean", "paired_clean_hole"}
SPLIT_ORDER = ("train", "validation", "test")
QUOTA_DIMENSIONS = {
    "wheel_camera_severity": ("target_wheel", "camera_pose", "severity"),
    "wheel_camera_sector": ("target_wheel", "camera_pose", "image_sector"),
    "wheel_camera_surface": ("target_wheel", "camera_pose", "surface"),
}


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _validate_distribution(name: str, values: dict, expected_keys: set[str]) -> None:
    if set(values) != expected_keys:
        raise ValueError(f"{name} keys must be exactly {sorted(expected_keys)}")
    if any(float(value) < 0.0 for value in values.values()):
        raise ValueError(f"{name} contains a negative probability")
    if abs(sum(map(float, values.values())) - 1.0) > 1e-9:
        raise ValueError(f"{name} probabilities must sum to one")


def validate_dataset_composition_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset-composition schema")
    if not str(config.get("dataset_id", "")).strip():
        raise ValueError("dataset_id is required")
    if int(config.get("master_seed", -1)) < 0:
        raise ValueError("master_seed must be non-negative")
    pool = config.get("pool", {})
    standalone = int(pool.get("standalone_clean_units", -1))
    paired = int(pool.get("paired_clean_hole_units", -1))
    if standalone < 0 or paired < 0:
        raise ValueError("Pool unit counts must be non-negative")
    if pool.get("artifact_policy") != "reference_relative_paths_without_copy":
        raise ValueError("Dataset v1 must not copy or move pool artifacts while splitting")
    composition = config.get("composition", {})
    total_images = int(composition.get("total_images", -1))
    clean_images = int(composition.get("clean_images", -1))
    anomaly_images = int(composition.get("anomaly_images", -1))
    if total_images != standalone + 2 * paired:
        raise ValueError("Total images must equal standalone units plus two images per pair")
    if clean_images != standalone + paired or anomaly_images != paired:
        raise ValueError("Clean/anomaly image counts do not match the pool composition")
    if clean_images + anomaly_images != total_images:
        raise ValueError("Condition counts do not sum to total_images")
    hole = composition.get("hole", {})
    _validate_distribution("hole.severity", hole.get("severity", {}), {"small", "medium", "large"})
    _validate_distribution("hole.surface", hole.get("surface", {}), {"tread", "shoulder"})
    if hole.get("constraints", {}).get("large_surface") != "tread":
        raise ValueError("Large holes must remain constrained to tread")
    domain = composition.get("domain_randomization", {})
    _validate_distribution("domain_randomization.camera_pose", domain.get("camera_pose", {}), {"A_overhead", "C_leading_three_quarter", "C_trailing_three_quarter", "D_upper_detail"})
    _validate_distribution("domain_randomization.lighting", domain.get("lighting", {}), {"mars_dusty_refined", "mars_clear_refined"})
    _validate_distribution("domain_randomization.surface_wear", domain.get("surface_wear", {}), {"surface_current", "wear_light", "wear_evident"})
    split_images = split_standalone = split_paired = 0
    split_fractions = 0.0
    for split in SPLIT_ORDER:
        row = config.get("splits", {}).get(split, {})
        fraction = float(row.get("fraction", -1.0))
        image_count = int(row.get("image_count", -1))
        clean_units = int(row.get("standalone_clean_units", -1))
        pair_units = int(row.get("paired_clean_hole_units", -1))
        if fraction < 0.0 or min(image_count, clean_units, pair_units) < 0 or image_count != clean_units + 2 * pair_units:
            raise ValueError(f"Invalid counts for split {split}")
        if abs(image_count - total_images * fraction) > 1e-9:
            raise ValueError(f"Fraction and image count disagree for split {split}")
        split_fractions += fraction
        split_images += image_count
        split_standalone += clean_units
        split_paired += pair_units
    if (split_images, split_standalone, split_paired) != (total_images, standalone, paired):
        raise ValueError("Split totals do not match pool totals")
    if abs(split_fractions - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")
    if int(config["splits"]["train"]["paired_clean_hole_units"]) != 0:
        raise ValueError("Training split must be clean-only")
    policy = config.get("split_policy", {})
    if policy.get("train_clean_only") is not True or policy.get("pair_lock_indivisible") is not True:
        raise ValueError("Clean-only training and indivisible pairs are required")
    if policy.get("artifact_materialization") != "index_only":
        raise ValueError("Splits must be index-only")
    quotas = config.get("evaluation_minimum_quotas", {})
    for split in ("validation", "test"):
        if set(quotas.get(split, {})) != set(QUOTA_DIMENSIONS):
            raise ValueError(f"Quota dimensions are incomplete for {split}")
        if any(int(value) < 1 for value in quotas[split].values()):
            raise ValueError(f"Minimum quotas must be positive for {split}")
    categories = config.get("evaluation_categories", {})
    expected_category_fields = {field for fields in QUOTA_DIMENSIONS.values() for field in fields}
    if set(categories) != expected_category_fields:
        raise ValueError("evaluation_categories must define every quota field")
    if any(not isinstance(values, list) or not values or len(values) != len(set(values)) for values in categories.values()):
        raise ValueError("evaluation_categories values must be non-empty unique lists")
    return {"dataset_id": config["dataset_id"], "total_images": total_images, "standalone_units": standalone, "paired_units": paired}


def validate_pool_rows(rows: Iterable[dict]) -> list[dict]:
    result = list(rows)
    ids: set[str] = set()
    locks: set[str] = set()
    required = {field for fields in QUOTA_DIMENSIONS.values() for field in fields}
    for row in result:
        unit_id = str(row.get("unit_id", ""))
        unit_type = row.get("unit_type")
        if not unit_id or unit_id in ids:
            raise ValueError(f"Missing or duplicate unit_id: {unit_id!r}")
        if unit_type not in UNIT_TYPES:
            raise ValueError(f"Unknown unit_type for {unit_id}")
        ids.add(unit_id)
        members = row.get("members")
        if not isinstance(members, list):
            raise ValueError(f"members must be a list for {unit_id}")
        conditions = [member.get("condition") for member in members]
        if unit_type == "standalone_clean":
            if conditions != ["clean"] or row.get("pair_lock_id") not in (None, ""):
                raise ValueError(f"Invalid standalone clean unit {unit_id}")
        else:
            lock = str(row.get("pair_lock_id", ""))
            if not lock or lock in locks or conditions != ["clean", "hole"]:
                raise ValueError(f"Invalid paired unit {unit_id}")
            locks.add(lock)
            if not required <= set(row.get("sampling", {})):
                raise ValueError(f"Paired unit {unit_id} lacks stratification fields")
            if row["sampling"]["severity"] == "large" and row["sampling"]["surface"] != "tread":
                raise ValueError(f"Large shoulder hole is forbidden: {unit_id}")
    return result


def _rank(seed: int, label: str, unit_id: str) -> str:
    return hashlib.sha256(f"{seed}:{label}:{unit_id}".encode("utf-8")).hexdigest()


def _quota_keys(row: dict) -> dict[str, tuple[str, ...]]:
    sampling = row["sampling"]
    return {name: tuple(str(sampling[field]) for field in fields) for name, fields in QUOTA_DIMENSIONS.items()}


def _expected_cells(config: dict) -> dict[str, set[tuple[str, ...]]]:
    categories = config["evaluation_categories"]
    return {
        name: set(itertools.product(*(map(str, categories[field]) for field in fields)))
        for name, fields in QUOTA_DIMENSIONS.items()
    }


def _select_quota_aware(rows: list[dict], count: int, minimums: dict, expected_cells: dict, seed: int, label: str) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"Not enough paired units for {label}")
    required = {name: {cell: int(minimums[name]) for cell in cells} for name, cells in expected_cells.items()}
    observed = {name: Counter() for name in QUOTA_DIMENSIONS}
    remaining = list(rows)
    selected: list[dict] = []
    while any(observed[name][cell] < need for name, cells in required.items() for cell, need in cells.items()):
        if len(selected) >= count or not remaining:
            raise ValueError(f"Unable to satisfy minimum quotas for {label}")
        def score(row: dict) -> tuple[int, str]:
            keys = _quota_keys(row)
            benefit = sum(observed[name][keys[name]] < required[name][keys[name]] for name in QUOTA_DIMENSIONS)
            return benefit, _rank(seed, label, row["unit_id"])
        best = max(remaining, key=score)
        if score(best)[0] == 0:
            raise ValueError(f"Pool lacks required quota cells for {label}")
        remaining.remove(best)
        selected.append(best)
        for name, cell in _quota_keys(best).items():
            observed[name][cell] += 1
    remaining.sort(key=lambda row: _rank(seed, f"{label}:fill", row["unit_id"]))
    selected.extend(remaining[: count - len(selected)])
    return selected


def _check_quotas(rows: list[dict], minimums: dict, label: str, expected_cells: dict) -> None:
    counts = {name: Counter(_quota_keys(row)[name] for row in rows) for name in QUOTA_DIMENSIONS}
    for name, cells in expected_cells.items():
        missing = {cell: int(minimums[name]) - counts[name][cell] for cell in cells if counts[name][cell] < int(minimums[name])}
        if missing:
            raise ValueError(f"{label} fails {name} quotas: {missing}")


def build_dataset_splits(config: dict, rows: Iterable[dict]) -> dict[str, list[dict]]:
    contract = validate_dataset_composition_config(config)
    pool = validate_pool_rows(rows)
    standalone = [row for row in pool if row["unit_type"] == "standalone_clean"]
    paired = [row for row in pool if row["unit_type"] == "paired_clean_hole"]
    if len(standalone) != contract["standalone_units"] or len(paired) != contract["paired_units"]:
        raise ValueError("Pool counts do not match dataset composition")
    seed = int(config["master_seed"])
    expected_cells = _expected_cells(config)
    standalone.sort(key=lambda row: _rank(seed, "standalone", row["unit_id"]))
    val_count = int(config["splits"]["validation"]["paired_clean_hole_units"])
    validation_pairs = _select_quota_aware(paired, val_count, config["evaluation_minimum_quotas"]["validation"], expected_cells, seed, "validation")
    validation_ids = {row["unit_id"] for row in validation_pairs}
    test_pairs = [row for row in paired if row["unit_id"] not in validation_ids]
    if len(test_pairs) != int(config["splits"]["test"]["paired_clean_hole_units"]):
        raise ValueError("Test pair count mismatch")
    _check_quotas(validation_pairs, config["evaluation_minimum_quotas"]["validation"], "validation", expected_cells)
    _check_quotas(test_pairs, config["evaluation_minimum_quotas"]["test"], "test", expected_cells)
    train_count = int(config["splits"]["train"]["standalone_clean_units"])
    val_clean_count = int(config["splits"]["validation"]["standalone_clean_units"])
    splits = {
        "train": standalone[:train_count],
        "validation": standalone[train_count:train_count + val_clean_count] + validation_pairs,
        "test": test_pairs,
    }
    for split in SPLIT_ORDER:
        splits[split].sort(key=lambda row: (row["unit_type"], row["unit_id"]))
    return splits


def expand_split_items(units: Iterable[dict], split: str) -> list[dict]:
    return [{"schema_version": SCHEMA_VERSION, "split": split, "unit_id": unit["unit_id"], "unit_type": unit["unit_type"], "pair_lock_id": unit.get("pair_lock_id"), **member} for unit in units for member in unit["members"]]


def write_split_indexes(dataset_dir: Path, config: dict, splits: dict[str, list[dict]]) -> dict:
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": SCHEMA_VERSION, "dataset_id": config["dataset_id"], "splits": {}}
    for split in SPLIT_ORDER:
        items = expand_split_items(splits[split], split)
        payload = b"".join(canonical_json_bytes(item) for item in items)
        path = split_dir / f"{split}.jsonl"
        path.write_bytes(payload)
        summary["splits"][split] = {"images": len(items), "units": len(splits[split]), "sha256": hashlib.sha256(payload).hexdigest(), "path": path.relative_to(dataset_dir).as_posix()}
    (dataset_dir / "dataset.json").write_bytes(canonical_json_bytes(summary))
    return summary
