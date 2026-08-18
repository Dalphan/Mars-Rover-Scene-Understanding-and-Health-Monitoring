"""Materialize the complete deterministic 10k dataset plan without rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.anomaly_batch import build_anomaly_plan, validate_anomaly_batch_config
from src.wheel_preparation.clean_batch import build_plan, validate_clean_batch_config
from src.wheel_preparation.dataset_splits import QUOTA_DIMENSIONS
from src.wheel_preparation.domain_randomization import sample_domain_randomization


WHEELS = ["wheel_front_left", "wheel_front_right", "wheel_middle_left", "wheel_middle_right", "wheel_rear_left", "wheel_rear_right"]
POSES = ["A_overhead", "C_leading_three_quarter", "C_trailing_three_quarter", "D_upper_detail"]
LIGHTS = ["mars_dusty_refined", "mars_clear_refined"]
WEARS = ["surface_current", "wear_light", "wear_evident"]
ROLLS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
SEVERITIES = ["small", "medium", "large"]
SURFACES = ["tread", "shoulder"]
SECTORS = ["leading", "upper", "trailing"]
PROFILES = ["jagged_slit", "branched_tear", "peeled_window"]
DOMAIN_FIELDS = ["target_wheel", "camera_pose", "lighting_preset", "surface_wear", "healthy_roll_degrees"]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(rows: list[dict], field: str) -> Counter:
    return Counter(row[field] for row in rows)


def _subtract(target: dict, fixed: Counter, label: str) -> dict:
    result = {key: int(value) - int(fixed[key]) for key, value in target.items()}
    if any(value < 0 for value in result.values()):
        raise ValueError(f"Fixed rows exceed {label} target: {result}")
    return result


def _values(counts: dict, seed: int, label: str) -> list:
    values = [value for value, count in counts.items() for _ in range(int(count))]
    random.Random(seed ^ int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")).shuffle(values)
    return values


def _independent_slots(targets: dict[str, dict], count: int, seed: int, label: str) -> list[dict]:
    columns = {}
    for field, target in targets.items():
        if sum(map(int, target.values())) != count:
            raise ValueError(f"{label}.{field} does not sum to {count}")
        columns[field] = _values(target, seed, f"{label}:{field}")
    return [{field: columns[field][index] for field in columns} for index in range(count)]


def _domain_key(row: dict) -> tuple:
    return tuple(row[field] for field in DOMAIN_FIELDS)


def _select_indices(slots: list[dict], domain: dict, *, start: int, excluded: set[int]) -> list[dict]:
    pending: dict[tuple, deque[dict]] = defaultdict(deque)
    for slot in slots:
        pending[_domain_key(slot)].append(slot)
    remaining = len(slots)
    index = int(start)
    maximum_scan = start + 2_000_000
    selected = []
    while remaining and index < maximum_scan:
        if index not in excluded:
            sample = sample_domain_randomization(domain, index, attempt=0)
            queue = pending.get(_domain_key(sample))
            if queue:
                slot = queue.popleft()
                selected.append({**slot, "sample_index": index, "domain_sample": sample})
                excluded.add(index)
                remaining -= 1
        index += 1
    if remaining:
        missing = {str(key): len(queue) for key, queue in pending.items() if queue}
        raise ValueError(f"Unable to match {remaining} deterministic domain slots; first missing={list(missing.items())[:8]}")
    return selected


def _mandatory_anomaly_rows(split: str) -> list[dict]:
    rows = []
    if split == "validation":
        pattern = [
            ("small", "shoulder", "leading"),
            ("medium", "tread", "upper"),
            ("large", "tread", "trailing"),
        ]
    else:
        pattern = [
            ("small", "shoulder", "leading"),
            ("medium", "tread", "upper"),
            ("large", "tread", "trailing"),
            ("small", "tread", "upper"),
            ("medium", "shoulder", "trailing"),
            ("large", "tread", "leading"),
        ]
    for wheel in WHEELS:
        for pose in POSES:
            for severity, surface, sector in pattern:
                rows.append({"target_wheel": wheel, "camera_pose": pose, "severity": severity, "surface": surface, "image_sector": sector})
    return rows


def _anomaly_slots(split: str, targets: dict, fixed: list[dict], seed: int) -> list[dict]:
    mandatory = _mandatory_anomaly_rows(split)
    generated_count = int(targets["count"]) - len(fixed)
    if len(mandatory) > generated_count:
        raise ValueError(f"Mandatory quota rows exceed generated {split} rows")

    fixed_joint = Counter((row["severity"], row["surface"]) for row in fixed)
    mandatory_joint = Counter((row["severity"], row["surface"]) for row in mandatory)
    joint_remaining = {
        key: int(value) - fixed_joint[key] - mandatory_joint[key]
        for key, value in targets["joint"].items()
    }
    fixed_sector = _counts(fixed, "image_sector")
    mandatory_sector = _counts(mandatory, "image_sector")
    sector_remaining = {
        key: int(value) - fixed_sector[key] - mandatory_sector[key]
        for key, value in targets["image_sector"].items()
    }
    if any(value < 0 for value in joint_remaining.values()) or any(value < 0 for value in sector_remaining.values()):
        raise ValueError(f"Mandatory/fixed anomaly rows exceed {split} targets")
    filler_count = generated_count - len(mandatory)
    joints = _values(joint_remaining, seed, f"{split}:joint")
    sectors = _values(sector_remaining, seed, f"{split}:sector")
    if len(joints) != filler_count or len(sectors) != filler_count:
        raise ValueError(f"Anomaly filler mismatch for {split}")
    fillers = [
        {"severity": joints[index][0], "surface": joints[index][1], "image_sector": sectors[index]}
        for index in range(filler_count)
    ]
    generated = mandatory + fillers

    # Profiles have no quota-cell role and can be distributed over all generated rows.
    profile_remaining = _subtract(targets["profile_family"], _counts(fixed, "profile_family"), f"{split}.profile_family")
    profiles = _values(profile_remaining, seed, f"{split}:profile")
    random.Random(seed ^ 0xA110).shuffle(generated)
    for row, profile in zip(generated, profiles, strict=True):
        row["profile_family"] = profile
    return generated


def _domain_slots_for_anomalies(anomaly_slots: list[dict], targets: dict[str, dict], seed: int, label: str) -> list[dict]:
    """Attach exact domain marginals without losing mandatory wheel/camera quota cells."""
    count = len(anomaly_slots)
    result = [dict(row) for row in anomaly_slots]
    for field, target in targets.items():
        constrained = Counter(row[field] for row in result if field in row)
        remaining = _subtract(target, constrained, f"{label}.{field}.mandatory")
        values = _values(remaining, seed, f"{label}:{field}")
        open_rows = [row for row in result if field not in row]
        if len(values) != len(open_rows):
            raise ValueError(f"Domain assignment mismatch for {label}.{field}")
        for row, value in zip(open_rows, values, strict=True):
            row[field] = value
    if any(set(DOMAIN_FIELDS) - set(row) for row in result) or len(result) != count:
        raise ValueError(f"Incomplete domain assignment for {label}")
    return result


def _quota_minima(rows: list[dict]) -> dict:
    result = {}
    for name, fields in QUOTA_DIMENSIONS.items():
        counts = Counter(tuple(row[field] for field in fields) for row in rows)
        result[name] = {"cells": len(counts), "minimum": min(counts.values()), "maximum": max(counts.values())}
    return result


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = b"".join(_canonical(row) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def plan(output_dir: Path) -> dict:
    composition = _read_json(REPO_ROOT / "configs/blender/dataset_composition_v1.json")
    domain = _read_json(REPO_ROOT / composition["source_contracts"]["domain_randomization"])
    anomaly = _read_json(REPO_ROOT / composition["source_contracts"]["hole_anomaly"])
    clean_template = _read_json(REPO_ROOT / "configs/blender/prebulk_200_clean.json")
    pair_template = _read_json(REPO_ROOT / "configs/blender/prebulk_200_anomaly.json")
    clean_run = REPO_ROOT / "outputs/anomaly_detection_2/prebulk_200/clean_batch/clean_150_v1"
    pair_run = REPO_ROOT / "outputs/anomaly_detection_2/prebulk_200/anomaly_batch/pairs_25_v1"
    fixed_clean_manifest = _read_jsonl(clean_run / "manifest.jsonl")
    fixed_pair_manifest = _read_jsonl(pair_run / "manifest.jsonl")
    fixed_clean = [{**row["sampling"], "sample_index": row["sample_index"], "sample_id": row["sample_id"]} for row in fixed_clean_manifest]
    fixed_pairs = [{**row["sampling"], **{key: row["anomaly"][key] for key in ("severity", "surface", "image_sector", "profile_family")}, "sample_index": row["sample_index"], "pair_id": row["pair_id"], "pair_lock_id": row["pair_lock_id"]} for row in fixed_pair_manifest]
    excluded = {row["sample_index"] for row in fixed_clean + fixed_pairs}
    seed = int(composition["master_seed"])

    pair_domain_targets = {
        "validation": {
            "target_wheel": dict(zip(WHEELS, [42, 42, 42, 42, 41, 41], strict=True)),
            "camera_pose": dict(zip(POSES, [96, 68, 68, 18], strict=True)),
            "lighting_preset": dict(zip(LIGHTS, [188, 62], strict=True)),
            "surface_wear": dict(zip(WEARS, [50, 112, 88], strict=True)),
            "healthy_roll_degrees": dict(zip(ROLLS, [32, 32, 31, 31, 31, 31, 31, 31], strict=True)),
        },
        "test": {
            "target_wheel": dict(zip(WHEELS, [167, 167, 166, 166, 167, 167], strict=True)),
            "camera_pose": dict(zip(POSES, [404, 276, 276, 44], strict=True)),
            "lighting_preset": dict(zip(LIGHTS, [750, 250], strict=True)),
            "surface_wear": dict(zip(WEARS, [200, 451, 349], strict=True)),
            "healthy_roll_degrees": dict(zip(ROLLS, [125, 125, 125, 125, 125, 125, 125, 125], strict=True)),
        },
    }
    pair_anomaly_targets = {
        "validation": {
            "count": 250,
            "joint": {("small", "tread"): 97, ("small", "shoulder"): 28, ("medium", "tread"): 78, ("medium", "shoulder"): 22, ("large", "tread"): 25},
            "image_sector": dict(zip(SECTORS, [84, 83, 83], strict=True)),
            "profile_family": dict(zip(PROFILES, [113, 100, 37], strict=True)),
        },
        "test": {
            "count": 1000,
            "joint": {("small", "tread"): 389, ("small", "shoulder"): 111, ("medium", "tread"): 311, ("medium", "shoulder"): 89, ("large", "tread"): 100},
            "image_sector": dict(zip(SECTORS, [333, 334, 333], strict=True)),
            "profile_family": dict(zip(PROFILES, [450, 400, 150], strict=True)),
        },
    }

    generated_pair_rows = []
    for split in ("validation", "test"):
        fixed = [] if split == "validation" else fixed_pairs
        domain_targets = {field: _subtract(target, _counts(fixed, field), f"{split}.{field}") for field, target in pair_domain_targets[split].items()}
        anomaly_slots = _anomaly_slots(split, pair_anomaly_targets[split], fixed, seed)
        slots = _domain_slots_for_anomalies(anomaly_slots, domain_targets, seed, f"pair:{split}:domain")
        random.Random(seed ^ (0x501 if split == "validation" else 0x502)).shuffle(slots)
        slots = [{**row, "planned_split": split} for row in slots]
        generated_pair_rows.extend(slots)
    selected_pairs = _select_indices(generated_pair_rows, domain, start=200_000, excluded=excluded)

    # The final image-weighted domain totals become exact after adding two images per pair.
    pair_total = fixed_pairs + selected_pairs
    standalone_targets = {
        "target_wheel": dict(zip(WHEELS, [1249, 1249, 1251, 1251, 1250, 1250], strict=True)),
        "camera_pose": dict(zip(POSES, [3000, 2062, 2062, 376], strict=True)),
        "lighting_preset": dict(zip(LIGHTS, [5624, 1876], strict=True)),
        "surface_wear": dict(zip(WEARS, [1500, 3374, 2626], strict=True)),
        "healthy_roll_degrees": dict(zip(ROLLS, [936, 936, 938, 938, 938, 938, 938, 938], strict=True)),
    }
    clean_split_targets = {
        "validation": {
            "target_wheel": dict(zip(WHEELS, [84, 84, 83, 83, 83, 83], strict=True)),
            "camera_pose": dict(zip(POSES, [200, 138, 137, 25], strict=True)),
            "lighting_preset": dict(zip(LIGHTS, [375, 125], strict=True)),
            "surface_wear": dict(zip(WEARS, [100, 225, 175], strict=True)),
            "healthy_roll_degrees": dict(zip(ROLLS, [63, 63, 63, 63, 62, 62, 62, 62], strict=True)),
        }
    }
    clean_split_targets["train"] = {
        field: {key: standalone_targets[field][key] - clean_split_targets["validation"][field][key] for key in standalone_targets[field]}
        for field in DOMAIN_FIELDS
    }
    generated_clean_slots = []
    for split in ("train", "validation"):
        fixed = fixed_clean if split == "train" else []
        targets = {field: _subtract(target, _counts(fixed, field), f"clean.{split}.{field}") for field, target in clean_split_targets[split].items()}
        count = (7000 if split == "train" else 500) - len(fixed)
        generated_clean_slots.extend({**row, "planned_split": split} for row in _independent_slots(targets, count, seed, f"clean:{split}"))
    selected_clean = _select_indices(generated_clean_slots, domain, start=20_000, excluded=excluded)

    selected_clean.sort(key=lambda row: row["sample_index"])
    selected_pairs.sort(key=lambda row: row["sample_index"])
    clean_config = {**clean_template, "sample_indices": [row["sample_index"] for row in selected_clean], "output": {"root": "outputs/anomaly_detection_2/bulk/clean_batch"}}
    pair_config = {
        **pair_template,
        "sample_indices": [row["sample_index"] for row in selected_pairs],
        "semantic_assignments": [
            {"sample_index": row["sample_index"], **{key: row[key] for key in ("severity", "surface", "image_sector", "profile_family")}}
            for row in selected_pairs
        ],
        "execution": {**pair_template["execution"], "chunk_size_pairs": 50},
        "output": {"root": "outputs/anomaly_detection_2/bulk/anomaly_batch"},
    }
    clean_config_path = REPO_ROOT / "configs/blender/bulk_remaining_clean.json"
    pair_config_path = REPO_ROOT / "configs/blender/bulk_remaining_anomaly.json"
    _write_json(clean_config_path, clean_config)
    _write_json(pair_config_path, pair_config)
    validate_clean_batch_config(clean_config)
    validate_anomaly_batch_config(pair_config)
    built_clean = build_plan(clean_config, domain)
    built_pairs = build_anomaly_plan(pair_config, domain, anomaly)

    units = []
    for row in fixed_clean:
        units.append({"unit_id": f"clean_{row['sample_id']}", "unit_type": "standalone_clean", "sample_index": row["sample_index"], "planned_split": "train", "render_status": "existing_prebulk", "sampling": {field: row[field] for field in DOMAIN_FIELDS}})
    for row in selected_clean:
        sample = row["domain_sample"]
        units.append({"unit_id": f"clean_{sample['sample_id']}", "unit_type": "standalone_clean", "sample_index": row["sample_index"], "planned_split": row["planned_split"], "render_status": "pending", "sampling": {field: sample[field] for field in DOMAIN_FIELDS}})
    for row in fixed_pairs:
        units.append({"unit_id": f"pair_{row['pair_id']}", "unit_type": "paired_clean_hole", "sample_index": row["sample_index"], "planned_split": "test", "render_status": "existing_prebulk", "pair_lock_id": row["pair_lock_id"], "sampling": {field: row[field] for field in DOMAIN_FIELDS + ["severity", "surface", "image_sector", "profile_family"]}})
    for selected, built in zip(selected_pairs, built_pairs, strict=True):
        sample, descriptor = built["domain_sample"], built["anomaly"]
        units.append({"unit_id": f"pair_{built['pair_id']}", "unit_type": "paired_clean_hole", "sample_index": built["sample_index"], "planned_split": selected["planned_split"], "render_status": "pending", "pair_lock_id": built["pair_lock_id"], "sampling": {**{field: sample[field] for field in DOMAIN_FIELDS}, **{field: descriptor[field] for field in ("severity", "surface", "image_sector", "profile_family")}}})
    units.sort(key=lambda row: (row["planned_split"], row["unit_type"], row["sample_index"]))

    paired_by_split = {split: [row["sampling"] for row in units if row["unit_type"] == "paired_clean_hole" and row["planned_split"] == split] for split in ("validation", "test")}
    quota_audit = {split: _quota_minima(rows) for split, rows in paired_by_split.items()}
    if any(quota_audit["validation"][name]["minimum"] < 1 or quota_audit["test"][name]["minimum"] < 2 for name in QUOTA_DIMENSIONS):
        raise ValueError(f"Planned split quota failure: {quota_audit}")

    unit_sha = _write_jsonl(output_dir / "units.jsonl", units)
    split_hashes = {}
    for split in ("train", "validation", "test"):
        split_hashes[split] = _write_jsonl(output_dir / "splits" / f"{split}.jsonl", [row for row in units if row["planned_split"] == split])

    image_domain = {}
    for field in DOMAIN_FIELDS:
        count = Counter()
        for row in units:
            count[row["sampling"][field]] += 2 if row["unit_type"] == "paired_clean_hole" else 1
        image_domain[field] = dict(count)
    pair_anomaly = {field: dict(Counter(row["sampling"][field] for row in units if row["unit_type"] == "paired_clean_hole")) for field in ("severity", "surface", "image_sector", "profile_family")}
    summary = {
        "schema_version": 1,
        "status": "planned",
        "dataset_id": composition["dataset_id"],
        "units": len(units),
        "images": sum(2 if row["unit_type"] == "paired_clean_hole" else 1 for row in units),
        "existing_prebulk_images": sum((2 if row["unit_type"] == "paired_clean_hole" else 1) for row in units if row["render_status"] == "existing_prebulk"),
        "pending_images": sum((2 if row["unit_type"] == "paired_clean_hole" else 1) for row in units if row["render_status"] == "pending"),
        "pending_clean_units": len(selected_clean),
        "pending_pair_units": len(selected_pairs),
        "split_units": {split: dict(Counter(row["unit_type"] for row in units if row["planned_split"] == split)) for split in ("train", "validation", "test")},
        "image_weighted_domain_distributions": image_domain,
        "paired_anomaly_distributions": pair_anomaly,
        "quota_audit": quota_audit,
        "artifacts": {
            "units": {"path": "units.jsonl", "sha256": unit_sha},
            "splits": {split: {"path": f"splits/{split}.jsonl", "sha256": digest} for split, digest in split_hashes.items()},
            "remaining_clean_config": {"path": clean_config_path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(clean_config_path)},
            "remaining_anomaly_config": {"path": pair_config_path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(pair_config_path)},
        },
        "source_contract_sha256": _sha256(REPO_ROOT / "configs/blender/dataset_composition_v1.json"),
        "materialized_plan_counts": {"clean": len(built_clean), "pairs": len(built_pairs)},
    }
    _write_json(output_dir / "plan.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to((REPO_ROOT / "outputs").resolve()):
        raise ValueError("Output must stay below outputs/")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Planning output already exists: {output}")
    if output.exists():
        for path in sorted(output.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output.mkdir(parents=True, exist_ok=True)
    summary = plan(output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
