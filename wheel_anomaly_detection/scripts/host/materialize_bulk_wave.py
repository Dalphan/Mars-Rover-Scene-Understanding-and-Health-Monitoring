"""Materialize a deterministic render wave from the immutable bulk unit plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--full-config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--unit-type", choices=("standalone_clean",), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--wave-id", required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.count < 1:
        raise ValueError("offset must be non-negative and count positive")
    plan_dir = args.plan_dir.resolve()
    units_path = plan_dir / "units.jsonl"
    candidates = sorted(
        (
            row for row in _read_jsonl(units_path)
            if row["planned_split"] == args.split
            and row["unit_type"] == args.unit_type
            and row["render_status"] == "pending"
        ),
        key=lambda row: int(row["sample_index"]),
    )
    selected = candidates[args.offset : args.offset + args.count]
    if len(selected) != args.count:
        raise ValueError(f"Requested {args.count} rows but only {len(selected)} are available")
    full_config = json.loads(args.full_config.read_text(encoding="utf-8"))
    allowed = set(map(int, full_config["sample_indices"]))
    indices = sorted(int(row["sample_index"]) for row in selected)
    if not set(indices) <= allowed:
        raise ValueError("Wave contains an index outside the full pending config")
    config = {
        **full_config,
        "sample_indices": indices,
        "output": {"root": "outputs/anomaly_detection_2/bulk/waves/clean"},
    }
    validate_clean_batch_config(config)
    output_config = args.output_config.resolve()
    if not output_config.is_relative_to((REPO_ROOT / "configs" / "blender").resolve()):
        raise ValueError("Wave config must stay below configs/blender")
    if output_config.exists():
        raise FileExistsError(f"Wave config already exists: {output_config}")
    _atomic_json(output_config, config)
    manifest = {
        "schema_version": 1,
        "wave_id": args.wave_id,
        "split": args.split,
        "unit_type": args.unit_type,
        "offset": args.offset,
        "count": len(indices),
        "image_count": len(indices),
        "source_plan": {
            "path": units_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(units_path),
        },
        "config": {
            "path": output_config.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(output_config),
        },
        "sample_indices": indices,
        "unit_ids": [row["unit_id"] for row in selected],
    }
    wave_manifest = plan_dir / "waves" / f"{args.wave_id}.json"
    if wave_manifest.exists():
        output_config.unlink(missing_ok=True)
        raise FileExistsError(f"Wave manifest already exists: {wave_manifest}")
    _atomic_json(wave_manifest, manifest)
    print(json.dumps({"wave_id": args.wave_id, "images": len(indices), "config": manifest["config"], "manifest": wave_manifest.relative_to(REPO_ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
