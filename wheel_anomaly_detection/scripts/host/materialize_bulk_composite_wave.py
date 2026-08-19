"""Materialize a clean or paired wave from multiple planned split slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.anomaly_batch import validate_anomaly_batch_config
from src.wheel_preparation.clean_batch import validate_clean_batch_config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_selection(value: str) -> tuple[str, int, int]:
    try:
        split, offset, count = value.split(":", 2)
        offset, count = int(offset), int(count)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("selection must be SPLIT:OFFSET:COUNT") from error
    if split not in {"train", "validation", "test"} or offset < 0 or count < 1:
        raise argparse.ArgumentTypeError("selection contains an invalid split, offset or count")
    return split, offset, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--kind", choices=("clean", "anomaly"), required=True)
    parser.add_argument("--selection", action="append", type=_parse_selection, required=True)
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    plan_dir = args.plan_dir.resolve()
    units_path = plan_dir / "units.jsonl"
    units = _read_jsonl(units_path)
    unit_type = "standalone_clean" if args.kind == "clean" else "paired_clean_hole"
    selected = []
    selection_summary = []
    for split, offset, count in args.selection:
        candidates = sorted(
            (row for row in units if row["planned_split"] == split and row["unit_type"] == unit_type and row["render_status"] == "pending"),
            key=lambda row: int(row["sample_index"]),
        )
        rows = candidates[offset : offset + count]
        if len(rows) != count:
            raise ValueError(f"Selection {split}:{offset}:{count} resolved only {len(rows)} rows")
        selected.extend(rows)
        selection_summary.append({"split": split, "offset": offset, "count": count})
    indices = sorted(int(row["sample_index"]) for row in selected)
    if len(indices) != len(set(indices)):
        raise ValueError("Composite selections overlap")

    full_config = json.loads(args.full_config.read_text(encoding="utf-8"))
    allowed = set(map(int, full_config["sample_indices"]))
    if not set(indices) <= allowed:
        raise ValueError("Composite wave contains an index outside the full pending config")
    config = {**full_config, "sample_indices": indices}
    if args.kind == "clean":
        config["output"] = {"root": "outputs/anomaly_detection_2/bulk/waves/clean"}
        validate_clean_batch_config(config)
        image_count = len(indices)
    else:
        assignments = {int(row["sample_index"]): row for row in full_config["semantic_assignments"]}
        config["semantic_assignments"] = [assignments[index] for index in indices]
        config["output"] = {"root": "outputs/anomaly_detection_2/bulk/waves/anomaly"}
        validate_anomaly_batch_config(config)
        image_count = 2 * len(indices)

    output_config = args.output_config.resolve()
    if not output_config.is_relative_to((REPO_ROOT / "configs" / "blender").resolve()):
        raise ValueError("Wave config must stay below configs/blender")
    if output_config.exists():
        raise FileExistsError(f"Wave config already exists: {output_config}")
    _atomic_json(output_config, config)
    manifest = {
        "schema_version": 1,
        "wave_id": args.wave_id,
        "kind": args.kind,
        "unit_type": unit_type,
        "selections": selection_summary,
        "unit_count": len(indices),
        "image_count": image_count,
        "source_plan": {"path": units_path.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(units_path)},
        "config": {"path": output_config.relative_to(REPO_ROOT).as_posix(), "sha256": _sha256(output_config)},
        "sample_indices": indices,
        "unit_ids": [row["unit_id"] for row in sorted(selected, key=lambda row: int(row["sample_index"]))],
    }
    wave_manifest = plan_dir / "waves" / f"{args.wave_id}.json"
    if wave_manifest.exists():
        output_config.unlink(missing_ok=True)
        raise FileExistsError(f"Wave manifest already exists: {wave_manifest}")
    _atomic_json(wave_manifest, manifest)
    print(json.dumps({"wave_id": args.wave_id, "units": len(indices), "images": image_count, "config": manifest["config"], "selections": selection_summary}, indent=2))


if __name__ == "__main__":
    main()
