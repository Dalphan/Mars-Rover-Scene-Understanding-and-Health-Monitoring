"""Fail-closed validator for the materialized 10k pre-render plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.anomaly_batch import validate_anomaly_batch_config
from src.wheel_preparation.clean_batch import validate_clean_batch_config
from src.wheel_preparation.dataset_splits import QUOTA_DIMENSIONS


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(plan_dir: Path) -> dict:
    summary = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    errors = []
    units_path = plan_dir / summary["artifacts"]["units"]["path"]
    if _sha256(units_path) != summary["artifacts"]["units"]["sha256"]:
        errors.append("units.jsonl checksum mismatch")
    units = _read_jsonl(units_path)
    if len(units) != 8750 or sum(2 if row["unit_type"] == "paired_clean_hole" else 1 for row in units) != 10000:
        errors.append("Unit/image total mismatch")
    if len({row["unit_id"] for row in units}) != len(units) or len({row["sample_index"] for row in units}) != len(units):
        errors.append("Duplicate unit ID or sample index")

    split_rows = {}
    for split in ("train", "validation", "test"):
        artifact = summary["artifacts"]["splits"][split]
        path = plan_dir / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            errors.append(f"{split} split checksum mismatch")
        split_rows[split] = _read_jsonl(path)
    flattened = [row for split in ("train", "validation", "test") for row in split_rows[split]]
    if Counter(row["unit_id"] for row in flattened) != Counter(row["unit_id"] for row in units):
        errors.append("Split rows are not an exact partition of units")
    expected_split = {
        "train": Counter({"standalone_clean": 7000}),
        "validation": Counter({"standalone_clean": 500, "paired_clean_hole": 250}),
        "test": Counter({"paired_clean_hole": 1000}),
    }
    for split, rows in split_rows.items():
        if Counter(row["unit_type"] for row in rows) != expected_split[split]:
            errors.append(f"{split} composition mismatch")
        if any(row["planned_split"] != split for row in rows):
            errors.append(f"{split} contains a mismatched planned_split")

    image_domain = {}
    for field in ("target_wheel", "camera_pose", "lighting_preset", "surface_wear", "healthy_roll_degrees"):
        counts = Counter()
        for row in units:
            counts[str(row["sampling"][field])] += 2 if row["unit_type"] == "paired_clean_hole" else 1
        image_domain[field] = counts
    expected = {
        "camera_pose": Counter({"A_overhead": 4000, "C_leading_three_quarter": 2750, "C_trailing_three_quarter": 2750, "D_upper_detail": 500}),
        "lighting_preset": Counter({"mars_dusty_refined": 7500, "mars_clear_refined": 2500}),
        "surface_wear": Counter({"surface_current": 2000, "wear_light": 4500, "wear_evident": 3500}),
        "healthy_roll_degrees": Counter({str(value): 1250 for value in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)}),
    }
    for field, target in expected.items():
        if image_domain[field] != target:
            errors.append(f"Image-weighted {field} distribution mismatch")
    if sorted(image_domain["target_wheel"].values()) != [1666, 1666, 1667, 1667, 1667, 1667]:
        errors.append("Wheel distribution is not maximally balanced")

    paired = [row for row in units if row["unit_type"] == "paired_clean_hole"]
    for field, target in {
        "severity": Counter({"small": 625, "medium": 500, "large": 125}),
        "surface": Counter({"tread": 1000, "shoulder": 250}),
        "image_sector": Counter({"leading": 417, "upper": 417, "trailing": 416}),
        "profile_family": Counter({"jagged_slit": 563, "branched_tear": 500, "peeled_window": 187}),
    }.items():
        if Counter(row["sampling"][field] for row in paired) != target:
            errors.append(f"Paired {field} distribution mismatch")
    if any(row["sampling"]["surface"] != "tread" for row in paired if row["sampling"]["severity"] == "large"):
        errors.append("Large shoulder hole in plan")

    quota_result = {}
    for split, minimum in (("validation", 1), ("test", 2)):
        rows = [row["sampling"] for row in split_rows[split] if row["unit_type"] == "paired_clean_hole"]
        quota_result[split] = {}
        for name, fields in QUOTA_DIMENSIONS.items():
            counts = Counter(tuple(row[field] for field in fields) for row in rows)
            expected_cells = 72 if name != "wheel_camera_surface" else 48
            observed_minimum = min(counts.values()) if counts else 0
            quota_result[split][name] = {"cells": len(counts), "minimum": observed_minimum}
            if len(counts) != expected_cells or observed_minimum < minimum:
                errors.append(f"{split} fails {name} quota")

    existing_images = sum(2 if row["unit_type"] == "paired_clean_hole" else 1 for row in units if row["render_status"] == "existing_prebulk")
    pending_images = sum(2 if row["unit_type"] == "paired_clean_hole" else 1 for row in units if row["render_status"] == "pending")
    if (existing_images, pending_images) != (200, 9800):
        errors.append("Existing/pending image counts mismatch")

    for key, validator in (("remaining_clean_config", validate_clean_batch_config), ("remaining_anomaly_config", validate_anomaly_batch_config)):
        artifact = summary["artifacts"][key]
        path = REPO_ROOT / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            errors.append(f"{key} checksum mismatch")
        validator(json.loads(path.read_text(encoding="utf-8")))
    return {"ok": not errors, "errors": errors, "units": len(units), "images": 10000, "existing_images": existing_images, "pending_images": pending_images, "quota_audit": quota_result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.plan_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
