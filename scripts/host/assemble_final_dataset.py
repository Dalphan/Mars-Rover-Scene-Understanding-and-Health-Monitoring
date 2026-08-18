"""Assemble the canonical 10k dataset and a no-copy hard-link training view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.dataset_splits import (
    SPLIT_ORDER,
    canonical_json_bytes,
    validate_dataset_composition_config,
    validate_pool_rows,
    write_split_indexes,
)


def _read_jsonl(path: Path) -> list[dict]:
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"Truncated JSONL: {path}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError(f"Path escapes repository: {value}")
    return path


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _artifact(run_dir: Path, row: dict, key: str) -> dict:
    artifact = row["artifacts"][key]
    path = (run_dir / artifact["path"]).resolve()
    return {"path": _repo_relative(path), "sha256": artifact["sha256"]}


def _clean_unit(run_dir: Path, row: dict) -> dict:
    return {
        "schema_version": 1,
        "unit_id": f"clean_{row['sample_id']}",
        "unit_type": "standalone_clean",
        "pair_lock_id": None,
        "sample_index": int(row["sample_index"]),
        "sampling": row["sampling"],
        "members": [{
            "condition": "clean",
            "rgb": _artifact(run_dir, row, "rgb"),
            "target_wheel_mask": _artifact(run_dir, row, "target_wheel_mask"),
        }],
        "source_run": _repo_relative(run_dir),
    }


def _paired_unit(run_dir: Path, row: dict) -> dict:
    post = row["gates"]["mask_and_photometric"]
    photometric_status = post.get("photometric_status")
    if photometric_status is None:
        photometric_status = (
            "normal_contrast"
            if post.get("gates", {}).get("changed_fraction") is True
            and post.get("gates", {}).get("median_delta") is True
            else "low_contrast"
        )
    return {
        "schema_version": 1,
        "unit_id": f"pair_{row['pair_id']}",
        "unit_type": "paired_clean_hole",
        "pair_lock_id": row["pair_lock_id"],
        "sample_index": int(row["sample_index"]),
        "sampling": {
            **row["sampling"],
            "severity": row["anomaly"]["severity"],
            "surface": row["anomaly"]["surface"],
            "image_sector": row["anomaly"]["image_sector"],
            "profile_family": row["anomaly"]["profile_family"],
        },
        "photometric_status": photometric_status,
        "members": [
            {
                "condition": "clean",
                "rgb": _artifact(run_dir, row, "clean_rgb"),
                "target_wheel_mask": _artifact(run_dir, row, "target_wheel_mask"),
            },
            {
                "condition": "hole",
                "rgb": _artifact(run_dir, row, "anomaly_rgb"),
                "target_wheel_mask": _artifact(run_dir, row, "target_wheel_mask"),
                "anomaly_mask": _artifact(run_dir, row, "anomaly_mask"),
            },
        ],
        "source_run": _repo_relative(run_dir),
    }


def _load_complete_run(path: Path) -> list[dict]:
    run = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if run.get("status") != "complete":
        raise ValueError(f"Source run is not complete: {path}")
    rows = _read_jsonl(path / "manifest.jsonl")
    counts = run.get("counts", {})
    expected = counts.get("completed", counts.get("completed_pairs"))
    if int(expected if expected is not None else -1) != len(rows):
        raise ValueError(f"Source run count mismatch: {path}")
    return rows


def collect_units(config: dict) -> list[dict]:
    units: list[dict] = []
    for manifest_value in config["preassembled_pool_manifests"]:
        units.extend(_read_jsonl(_repo_path(manifest_value)))
    for value in config["clean_runs"]:
        run_dir = _repo_path(value)
        units.extend(_clean_unit(run_dir, row) for row in _load_complete_run(run_dir))
    for value in config["anomaly_runs"]:
        run_dir = _repo_path(value)
        units.extend(_paired_unit(run_dir, row) for row in _load_complete_run(run_dir))
    return validate_pool_rows(units)


def reconcile_with_plan(units: list[dict], planning: list[dict]) -> list[dict]:
    source = {row["unit_id"]: row for row in units}
    planned = {row["unit_id"]: row for row in planning}
    if len(source) != len(units) or len(planned) != len(planning) or set(source) != set(planned):
        missing = sorted(set(planned) - set(source))
        extra = sorted(set(source) - set(planned))
        raise ValueError(f"Pool/plan identity mismatch: missing={missing[:5]} extra={extra[:5]}")
    semantic_fields = (
        "target_wheel", "camera_pose", "healthy_roll_degrees", "lighting_preset",
        "surface_wear", "severity", "surface", "image_sector", "profile_family",
    )
    ordered = []
    for plan_row in planning:
        row = source[plan_row["unit_id"]]
        if int(row["sample_index"]) != int(plan_row["sample_index"]):
            raise ValueError(f"Sample index mismatch: {row['unit_id']}")
        if row.get("pair_lock_id") != plan_row.get("pair_lock_id"):
            if row["unit_type"] != "paired_clean_hole" or not row.get("pair_lock_id") or not plan_row.get("pair_lock_id"):
                raise ValueError(f"Invalid pair lock mismatch: {row['unit_id']}")
            row = {**row, "planned_pair_lock_id": plan_row["pair_lock_id"]}
        for field in semantic_fields:
            if field in plan_row["sampling"] and row["sampling"].get(field) != plan_row["sampling"][field]:
                raise ValueError(f"Planned semantic mismatch {field}: {row['unit_id']}")
        ordered.append({**row, "planned_split": plan_row["planned_split"]})
    return ordered


def verify_artifacts(units: list[dict]) -> dict[str, str]:
    cache: dict[str, str] = {}
    for unit in units:
        for member in unit["members"]:
            for key in ("rgb", "target_wheel_mask", "anomaly_mask"):
                if key not in member:
                    continue
                artifact = member[key]
                path = _repo_path(artifact["path"])
                if not path.is_file():
                    raise ValueError(f"Missing source artifact: {path}")
                if artifact["path"] not in cache:
                    cache[artifact["path"]] = _sha256(path)
                actual = cache[artifact["path"]]
                if actual != artifact["sha256"]:
                    raise ValueError(f"Source artifact SHA mismatch: {path}")
    return cache


def _hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"Hard-link identity check failed: {destination}")


def materialize_view(staging: Path, units: list[dict]) -> tuple[list[dict], Counter]:
    image_rows: list[dict] = []
    counts: Counter = Counter()
    for unit in units:
        split = unit["planned_split"]
        for member in unit["members"]:
            condition = member["condition"]
            image_id = unit["unit_id"] if unit["unit_type"] == "standalone_clean" else f"{unit['unit_id']}_{condition}"
            rgb_relative = Path("images") / split / condition / f"{image_id}.png"
            target_relative = Path("masks/target_wheel") / split / condition / f"{image_id}.png"
            _hardlink(_repo_path(member["rgb"]["path"]), staging / rgb_relative)
            _hardlink(_repo_path(member["target_wheel_mask"]["path"]), staging / target_relative)
            artifacts = {
                "rgb": {"path": rgb_relative.as_posix(), "sha256": member["rgb"]["sha256"]},
                "target_wheel_mask": {"path": target_relative.as_posix(), "sha256": member["target_wheel_mask"]["sha256"]},
            }
            if "anomaly_mask" in member:
                anomaly_relative = Path("masks/anomaly") / split / condition / f"{image_id}.png"
                _hardlink(_repo_path(member["anomaly_mask"]["path"]), staging / anomaly_relative)
                artifacts["anomaly_mask"] = {"path": anomaly_relative.as_posix(), "sha256": member["anomaly_mask"]["sha256"]}
            image_rows.append({
                "schema_version": 1, "image_id": image_id, "unit_id": unit["unit_id"],
                "unit_type": unit["unit_type"], "pair_lock_id": unit.get("pair_lock_id"),
                "split": split, "condition": condition, "sample_index": unit["sample_index"],
                "photometric_status": unit.get("photometric_status"), "sampling": unit["sampling"],
                "artifacts": artifacts,
            })
            counts[(split, condition)] += 1
    return image_rows, counts


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def assemble(config_path: Path, dataset_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 1 or config.get("materialization") != "hardlink":
        raise ValueError("Unsupported final-dataset assembly config")
    composition_path = _repo_path(config["composition_config"])
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    contract = validate_dataset_composition_config(composition)
    if config["dataset_id"] != contract["dataset_id"] or dataset_dir.name != contract["dataset_id"]:
        raise ValueError("Dataset identity mismatch")
    planning_path = _repo_path(config["planning_units"])
    planning = _read_jsonl(planning_path)
    units = reconcile_with_plan(collect_units(config), planning)
    if len(units) != 8750 or sum(len(row["members"]) for row in units) != 10000:
        raise ValueError("Final pool does not contain exactly 8,750 units / 10,000 images")
    if config.get("verify_source_sha256") is not True:
        raise ValueError("Final assembly requires source SHA verification")
    verified = verify_artifacts(units)
    final_paths = [dataset_dir / "images", dataset_dir / "masks", dataset_dir / "splits", dataset_dir / "dataset.json", dataset_dir / "image_manifest.jsonl", dataset_dir / "pool/manifest.jsonl"]
    if any(path.exists() for path in final_paths):
        raise FileExistsError("Final dataset outputs already exist; refusing implicit overwrite")
    staging = dataset_dir / "_final_assembly_staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    committed_paths: list[Path] = []
    try:
        image_rows, counts = materialize_view(staging, units)
        image_manifest_sha = _write_jsonl(staging / "image_manifest.jsonl", image_rows)
        pool_manifest_sha = _write_jsonl(staging / "pool_manifest.jsonl", units)
        split_units = {split: [row for row in units if row["planned_split"] == split] for split in SPLIT_ORDER}
        split_summary = write_split_indexes(staging, composition, split_units)
        expected_counts = {
            ("train", "clean"): 7000,
            ("validation", "clean"): 750,
            ("validation", "hole"): 250,
            ("test", "clean"): 1000,
            ("test", "hole"): 1000,
        }
        if dict(counts) != expected_counts:
            raise ValueError(f"Materialized condition counts mismatch: {dict(counts)}")
        summary = {
            "schema_version": 1,
            "dataset_id": config["dataset_id"],
            "status": "complete",
            "materialization": "hardlink",
            "units": len(units),
            "images": len(image_rows),
            "source_artifacts_verified": len(verified),
            "condition_counts": {f"{split}/{condition}": value for (split, condition), value in sorted(counts.items())},
            "pair_lock_rebased_units": sum("planned_pair_lock_id" in row for row in units),
            "pool_manifest": {"path": "pool/manifest.jsonl", "sha256": pool_manifest_sha},
            "image_manifest": {"path": "image_manifest.jsonl", "sha256": image_manifest_sha},
            "planning_units": {"path": _repo_relative(planning_path), "sha256": _sha256(planning_path)},
            "composition_config": {"path": _repo_relative(composition_path), "sha256": _sha256(composition_path)},
            "splits": split_summary["splits"],
            "warning": "Hard-linked files share storage with immutable run outputs; do not modify raster contents in place.",
        }
        (staging / "dataset.json").write_bytes(canonical_json_bytes(summary))
        (staging / "README.md").write_text(
            "# Curiosity wheel-hole dataset v1\n\nRaster files are hard links to validated run outputs. "
            "Do not edit them in place. Use `image_manifest.jsonl` for per-image metadata and `splits/*.jsonl` for fixed splits.\n",
            encoding="utf-8",
        )
        (dataset_dir / "pool").mkdir(exist_ok=True)
        pool_destination = dataset_dir / "pool/manifest.jsonl"
        os.replace(staging / "pool_manifest.jsonl", pool_destination)
        committed_paths.append(pool_destination)
        for name in ("images", "masks", "splits", "image_manifest.jsonl", "dataset.json", "README.md"):
            destination = dataset_dir / name
            os.replace(staging / name, destination)
            committed_paths.append(destination)
        staging.rmdir()
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        for path in reversed(committed_paths):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    expected_root = (REPO_ROOT / "outputs/anomaly_detection_2/datasets").resolve()
    if not dataset_dir.is_relative_to(expected_root):
        raise ValueError("Dataset directory must stay below outputs/anomaly_detection_2/datasets")
    print(json.dumps(assemble(args.config.resolve(), dataset_dir), indent=2))


if __name__ == "__main__":
    main()
