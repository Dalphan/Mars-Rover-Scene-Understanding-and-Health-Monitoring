"""Fail-closed validation for a deterministic clean Blender batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.clean_batch import sha256_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        raise ValueError(f"Missing JSONL file: {path.name}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"JSONL file has a truncated final line: {path.name}")
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank JSONL line {number} in {path.name}")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSONL line {number} in {path.name}") from error
    return rows


def _inside_run(run_dir: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"Artifact path must be relative: {relative}")
    resolved = (run_dir / value).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Artifact path escapes run directory: {relative}")
    return resolved


def validate_artifact_row(run_dir: Path, row: dict, resolution: tuple[int, int]) -> None:
    if row.get("condition") != "clean":
        raise ValueError(f"Sample {row.get('sample_id')} is not labeled clean")
    gates = row.get("gates", {})
    if any(gates.get(key) is not True for key in ("framing", "terrain_coverage", "terrain_contact")):
        raise ValueError(f"Sample {row.get('sample_id')} has failed gates")
    if gates.get("terrain_contact_details", {}).get("ok") is not True:
        raise ValueError(f"Sample {row.get('sample_id')} has invalid terrain-contact details")
    matrices = row.get("resolved", {})
    for key in ("camera_matrix_world", "wheel_matrix_world", "rover_matrix_world"):
        values = matrices.get(key, [])
        if len(values) != 16 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Sample {row.get('sample_id')} has invalid {key}")

    artifacts = row.get("artifacts", {})
    expected = {
        "rgb": f"rgb/{row['sample_id']}.png",
        "target_wheel_mask": f"target_wheel_mask/{row['sample_id']}.png",
    }
    for kind, relative in expected.items():
        record = artifacts.get(kind, {})
        if record.get("path") != relative:
            raise ValueError(f"Unexpected {kind} path for {row['sample_id']}")
        path = _inside_run(run_dir, relative)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty {kind} for {row['sample_id']}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {relative}")
        with Image.open(path) as image:
            image.load()
            if image.size != resolution:
                raise ValueError(f"Unexpected dimensions for {relative}: {image.size}")
            if kind == "rgb":
                if image.mode != "RGB":
                    raise ValueError(f"RGB output must use RGB mode: {relative} is {image.mode}")
            else:
                values = set(image.convert("L").tobytes())
                if values != {0, 255}:
                    raise ValueError(f"Target mask must be binary, non-empty and non-full: {relative} has {sorted(values)}")

    timings = row.get("render", {})
    for key in ("setup_seconds", "render_seconds", "write_seconds", "total_seconds"):
        value = float(timings.get(key, -1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid timing {key} for {row['sample_id']}")


def validate_manifest_matches_plan(row: dict, planned: dict) -> None:
    if row.get("sample_id") != planned.get("sample_id") or int(row.get("sample_index", -1)) != int(planned["sample_index"]):
        raise ValueError("Manifest sample identity does not match plan")
    if row.get("semantic_sample_id") != planned.get("sample_id"):
        raise ValueError(f"Semantic sample ID mismatch for {planned['sample_id']}")
    seeds = row.get("seeds", {})
    expected_seeds = {
        "sample": int(planned["sample_seed"]),
        "wear": int(planned["wear_seed"]),
    }
    for key, expected in expected_seeds.items():
        if seeds.get(key) != expected:
            raise ValueError(f"Manifest seed {key} differs from plan for {planned['sample_id']}")
    camera_attempt_seed = int(seeds.get("camera_attempt", -1))
    expected_pair_lock = hashlib.sha256(
        f"domain-pair:{int(planned['sample_seed'])}:{camera_attempt_seed}".encode("ascii")
    ).hexdigest()[:20]
    if seeds.get("pair_lock_id") != expected_pair_lock:
        raise ValueError(f"Manifest pair-lock is inconsistent for {planned['sample_id']}")
    sampling = row.get("sampling", {})
    for key in ("target_wheel", "camera_pose", "lighting_preset", "surface_wear", "lighting_jitter"):
        if sampling.get(key) != planned[key]:
            raise ValueError(f"Manifest sampling field {key} differs from plan for {planned['sample_id']}")
    if float(sampling.get("healthy_roll_degrees", float("nan"))) != float(planned["healthy_roll_degrees"]):
        raise ValueError(f"Manifest roll differs from plan for {planned['sample_id']}")


def validate_artifact_inventory(run_dir: Path, expected_ids: list[str]) -> None:
    expected = {f"{sample_id}.png" for sample_id in expected_ids}
    actual_rgb = {path.name for path in (run_dir / "rgb").glob("*.png")}
    actual_masks = {path.name for path in (run_dir / "target_wheel_mask").glob("*.png")}
    if actual_rgb != expected or actual_masks != expected:
        raise ValueError("Artifact directories contain missing or orphan PNGs")


def validate_clean_batch_run(run_dir: Path, *, require_complete: bool = True) -> dict:
    run_dir = run_dir.resolve()
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ValueError("Missing run.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if int(run.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported run schema")
    if require_complete and run.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run.get('status')}")

    plan_path = run_dir / "plan.jsonl"
    manifest_path = run_dir / "manifest.jsonl"
    plan_rows = _read_jsonl(plan_path)
    manifest_rows = _read_jsonl(manifest_path)
    if sha256_bytes(plan_path.read_bytes()) != run.get("plan", {}).get("sha256"):
        raise ValueError("plan.jsonl SHA-256 mismatch")
    selection = run.get("selection")
    if selection is None and "range" in run:
        selection = {"range": run["range"]}
    if not isinstance(selection, dict) or ("range" in selection) == ("sample_indices" in selection):
        raise ValueError("Run selection must contain exactly one of range or sample_indices")
    if "sample_indices" in selection:
        expected_indices = [int(value) for value in selection["sample_indices"]]
    else:
        sample_range = selection["range"]
        expected_indices = list(range(int(sample_range["start_index"]), int(sample_range["start_index"]) + int(sample_range["count"])))
    expected_count = len(expected_indices)
    if len(plan_rows) != expected_count:
        raise ValueError(f"Plan count mismatch: {len(plan_rows)} != {expected_count}")
    if len(manifest_rows) != expected_count:
        raise ValueError(f"Manifest count mismatch: {len(manifest_rows)} != {expected_count}")
    expected_ids = [row["sample_id"] for row in plan_rows]
    actual_ids = [row["sample_id"] for row in manifest_rows]
    if len(set(expected_ids)) != len(expected_ids) or actual_ids != expected_ids:
        raise ValueError("Manifest IDs must be unique and match plan order exactly")
    if [int(row["sample_index"]) for row in plan_rows] != expected_indices:
        raise ValueError("Plan sample indices do not match the configured selection")

    for row, planned in zip(manifest_rows, plan_rows, strict=True):
        validate_manifest_matches_plan(row, planned)

    resolution = tuple(map(int, run["render"]["resolution"]))
    for row in manifest_rows:
        validate_artifact_row(run_dir, row, resolution)

    validate_artifact_inventory(run_dir, expected_ids)
    actual_rgb = {path.name for path in (run_dir / "rgb").glob("*.png")}
    actual_masks = {path.name for path in (run_dir / "target_wheel_mask").glob("*.png")}

    matrix_path = run_dir / "matrix_audit.json"
    if not matrix_path.is_file():
        raise ValueError("Missing matrix_audit.json")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("ok") is not True or int(matrix.get("combination_count", 0)) != 24 or int(matrix.get("accepted_count", 0)) != 24:
        raise ValueError("The 6x4 wheel/camera matrix audit did not pass")

    source = (REPO_ROOT / run["source"]["path"]).resolve()
    if not source.is_file() or _sha256(source) != run["source"]["sha256"]:
        raise ValueError("Immutable source Blend is missing or changed")
    configs = run.get("configs", {})
    required_configs = {"batch", "domain_randomization", "lighting", "surface_wear", "camera_poses", "pose_sampling"}
    if set(configs) != required_configs:
        raise ValueError("Run metadata does not contain the complete configuration fingerprint set")
    for name, record in configs.items():
        path = (REPO_ROOT / record["path"]).resolve()
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"Configuration {name} is missing or changed")
    runtime = run.get("runtime", {})
    if not str(runtime.get("blender_version", "")).startswith("5.2."):
        raise ValueError("Run metadata does not record Blender 5.2")
    if runtime.get("engine") != "BLENDER_EEVEE" or not runtime.get("device", {}).get("renderer"):
        raise ValueError("Run metadata is missing Eevee device information")
    return {
        "ok": True,
        "sample_count": len(manifest_rows),
        "rgb_count": len(actual_rgb),
        "target_wheel_mask_count": len(actual_masks),
        "matrix_combinations": int(matrix["combination_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_clean_batch_run(args.run_dir, require_complete=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
