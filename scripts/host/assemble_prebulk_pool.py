"""Assemble validated clean and paired runs into a reusable partial dataset pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _artifact(run_dir: Path, row: dict, key: str) -> dict:
    artifact = row["artifacts"][key]
    path = run_dir / artifact["path"]
    if not path.is_file() or _sha256(path) != artifact["sha256"]:
        raise ValueError(f"Invalid artifact {key}: {path}")
    return {"path": _relative(path), "sha256": artifact["sha256"]}


def assemble(clean_run: Path, anomaly_run: Path) -> list[dict]:
    clean_meta = json.loads((clean_run / "run.json").read_text(encoding="utf-8"))
    anomaly_meta = json.loads((anomaly_run / "run.json").read_text(encoding="utf-8"))
    if clean_meta.get("status") != "complete" or anomaly_meta.get("status") != "complete":
        raise ValueError("Both source runs must be complete")
    rows = []
    for row in _read_jsonl(clean_run / "manifest.jsonl"):
        rows.append({
            "schema_version": 1,
            "unit_id": f"clean_{row['sample_id']}",
            "unit_type": "standalone_clean",
            "pair_lock_id": None,
            "sample_index": int(row["sample_index"]),
            "sampling": row["sampling"],
            "members": [{
                "condition": "clean",
                "rgb": _artifact(clean_run, row, "rgb"),
                "target_wheel_mask": _artifact(clean_run, row, "target_wheel_mask"),
            }],
            "source_run": _relative(clean_run),
        })
    for row in _read_jsonl(anomaly_run / "manifest.jsonl"):
        rows.append({
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
            },
            "members": [
                {
                    "condition": "clean",
                    "rgb": _artifact(anomaly_run, row, "clean_rgb"),
                    "target_wheel_mask": _artifact(anomaly_run, row, "target_wheel_mask"),
                },
                {
                    "condition": "hole",
                    "rgb": _artifact(anomaly_run, row, "anomaly_rgb"),
                    "target_wheel_mask": _artifact(anomaly_run, row, "target_wheel_mask"),
                    "anomaly_mask": _artifact(anomaly_run, row, "anomaly_mask"),
                },
            ],
            "source_run": _relative(anomaly_run),
        })
    unit_ids = [row["unit_id"] for row in rows]
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate pool unit IDs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--anomaly-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    clean_run, anomaly_run, output_dir = args.clean_run.resolve(), args.anomaly_run.resolve(), args.output_dir.resolve()
    if not output_dir.is_relative_to((REPO_ROOT / "outputs").resolve()):
        raise ValueError("Output must stay below outputs/")
    rows = assemble(clean_run, anomaly_run)
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    manifest = output_dir / "manifest.jsonl"
    manifest.write_bytes(payload)
    standalone = sum(row["unit_type"] == "standalone_clean" for row in rows)
    paired = sum(row["unit_type"] == "paired_clean_hole" for row in rows)
    summary = {
        "schema_version": 1,
        "status": "prebulk_complete",
        "units": len(rows),
        "images": standalone + 2 * paired,
        "standalone_clean_units": standalone,
        "paired_clean_hole_units": paired,
        "manifest": {"path": "manifest.jsonl", "sha256": hashlib.sha256(payload).hexdigest()},
        "source_runs": {"clean": _relative(clean_run), "anomaly": _relative(anomaly_run)},
    }
    (output_dir / "prebulk.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
