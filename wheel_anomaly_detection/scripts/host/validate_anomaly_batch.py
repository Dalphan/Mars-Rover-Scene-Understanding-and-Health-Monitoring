"""Fail-closed validator for paired clean/hole Blender batches."""

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
    if not path.is_file():
        raise ValueError(f"Missing JSONL: {path.name}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"JSONL has an uncommitted tail: {path.name}")
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed {path.name} line {number}") from error
    return rows


def _inside_run(run_dir: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Artifact path is not relative: {relative}")
    resolved = (run_dir / value).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError(f"Artifact path escapes run: {relative}")
    return resolved


def _components(pixels: set[tuple[int, int]]) -> int:
    remaining = set(pixels)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            x, y = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbour = (x + dx, y + dy)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
    return count


def validate_pair_artifacts(run_dir: Path, row: dict, resolution: tuple[int, int]) -> None:
    expected = {"clean_rgb", "anomaly_rgb", "target_wheel_mask", "anomaly_mask"}
    if set(row.get("artifacts", {})) != expected:
        raise ValueError(f"Artifact keys mismatch for {row.get('pair_id')}")
    images = {}
    for key, artifact in row["artifacts"].items():
        path = _inside_run(run_dir, artifact["path"])
        if not path.is_file() or _sha256(path) != artifact.get("sha256"):
            raise ValueError(f"Missing/corrupt {key} for {row['pair_id']}")
        try:
            image = Image.open(path)
            image.load()
        except Exception as error:
            raise ValueError(f"Undecodable {key} for {row['pair_id']}") from error
        if image.size != resolution:
            raise ValueError(f"Wrong {key} resolution for {row['pair_id']}")
        expected_mode = "L" if key.endswith("mask") else "RGB"
        if image.mode != expected_mode:
            raise ValueError(f"Wrong {key} mode for {row['pair_id']}: {image.mode}")
        images[key] = image
    target_bytes = images["target_wheel_mask"].tobytes()
    anomaly_bytes = images["anomaly_mask"].tobytes()
    target_values = set(target_bytes)
    anomaly_values = set(anomaly_bytes)
    if not target_values <= {0, 255} or target_values != {0, 255}:
        raise ValueError(f"Invalid target mask for {row['pair_id']}")
    if not anomaly_values <= {0, 255} or anomaly_values != {0, 255}:
        raise ValueError(f"Invalid anomaly mask for {row['pair_id']}")
    width = resolution[0]
    pixels = {(index % width, index // width) for index, value in enumerate(anomaly_bytes) if value == 255}
    if _components(pixels) != 1:
        raise ValueError(f"Anomaly mask is not one 8-connected component for {row['pair_id']}")


def validate_photometric_record(post: dict, pair_id: str) -> None:
    post_gates = post.get("gates", {})
    required_post_gates = (
        "binary", "single_component", "minimum_area", "minimum_short_side",
        "inside_target_roi", "local_difference_energy", "maximum_effect_area",
        "nonzero_effect",
    )
    if post.get("ok") is not True or not all(post_gates.get(name) is True for name in required_post_gates):
        raise ValueError(f"Mask/photometric gate false for {pair_id}")
    photometric_status = post.get("photometric_status")
    if photometric_status == "normal_contrast":
        if post_gates.get("changed_fraction") is not True or post_gates.get("median_delta") is not True:
            raise ValueError(f"Inconsistent normal-contrast status for {pair_id}")
    elif photometric_status == "low_contrast":
        if post_gates.get("changed_fraction") is True and post_gates.get("median_delta") is True:
            raise ValueError(f"Inconsistent low-contrast status for {pair_id}")
    else:
        raise ValueError(f"Invalid photometric status for {pair_id}")


def validate_manifest_matches_plan(row: dict, planned: dict) -> None:
    for key in ("pair_id", "pair_index", "sample_index", "members", "pair_lock_id"):
        if row.get(key) != planned.get(key):
            raise ValueError(f"Manifest/plan mismatch {key} for {planned['pair_id']}")
    if row.get("condition") != "paired_clean_hole":
        raise ValueError(f"Wrong condition for {planned['pair_id']}")
    sampling = row.get("sampling", {})
    domain = planned["domain_sample"]
    for key in (
        "target_wheel", "camera_pose", "healthy_roll_degrees", "lighting_preset",
        "surface_wear", "lighting_jitter",
    ):
        if sampling.get(key) != domain.get(key):
            raise ValueError(f"Pair-locked field changed: {key} for {planned['pair_id']}")
    if row.get("anomaly") != planned.get("anomaly"):
        raise ValueError(f"Anomaly descriptor changed for {planned['pair_id']}")
    hole_material = row.get("resolved", {}).get("hole_material", {})
    expected_t3_max = {"small": 0.002, "medium": 0.003, "large": 0.004}.get(
        planned.get("anomaly", {}).get("severity")
    )
    t3_values = (
        hole_material.get("open_wall_depth_profile"),
        hole_material.get("open_wall_fold_arc_count"),
        hole_material.get("effective_open_wall_depth_min_m"),
        hole_material.get("effective_open_wall_depth_max_m"),
        hole_material.get("effective_open_wall_depth_m"),
    )
    if (
        expected_t3_max is None
        or t3_values[0] != "T3"
        or int(t3_values[1] or 0) != 2
        or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in t3_values[2:])
        or abs(float(t3_values[2]) - 0.00075) > 1e-12
        or abs(float(t3_values[3]) - expected_t3_max) > 1e-12
        or abs(float(t3_values[4]) - expected_t3_max) > 1e-12
    ):
        raise ValueError(f"T3 wall-depth contract invalid for {planned['pair_id']}")
    gates = row.get("gates", {})
    if not all(gates.get(key) is True for key in ("framing", "terrain_coverage", "terrain_contact", "anomaly_visibility", "through_opening")):
        raise ValueError(f"Geometric gate false for {planned['pair_id']}")
    if gates.get("terrain_contact_details", {}).get("ok") is not True:
        raise ValueError(f"Terrain-contact details invalid for {planned['pair_id']}")
    if gates.get("through_opening_details", {}).get("ok") is not True:
        raise ValueError(f"Through-opening details invalid for {planned['pair_id']}")
    if gates.get("opening_mask_details", {}).get("ok") is not True:
        raise ValueError(f"Projected opening-mask details invalid for {planned['pair_id']}")
    post = gates.get("mask_and_photometric", {})
    validate_photometric_record(post, planned["pair_id"])
    for matrix_key in ("camera_matrix_world", "wheel_matrix_world", "rover_matrix_world"):
        values = row.get("resolved", {}).get(matrix_key)
        if not isinstance(values, list) or len(values) != 16 or not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Invalid {matrix_key} for {planned['pair_id']}")


def _inventory(run_dir: Path, pair_ids: list[str]) -> None:
    expected = {
        "rgb/clean": {f"{pair_id}.png" for pair_id in pair_ids},
        "rgb/anomaly": {f"{pair_id}.png" for pair_id in pair_ids},
        "target_wheel_mask": {f"{pair_id}.png" for pair_id in pair_ids},
        "anomaly_mask": {f"{pair_id}.png" for pair_id in pair_ids},
    }
    for relative, names in expected.items():
        directory = run_dir / relative
        actual = {path.name for path in directory.iterdir() if path.is_file()} if directory.is_dir() else set()
        if actual != names:
            raise ValueError(f"Artifact inventory mismatch in {relative}: missing={sorted(names-actual)} extra={sorted(actual-names)}")


def validate_anomaly_batch_run(run_dir: Path, *, require_complete: bool = True) -> dict:
    run_dir = run_dir.resolve()
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        raise ValueError("Missing run.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if require_complete and run.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run.get('status')}")
    plan = _read_jsonl(run_dir / "plan.jsonl")
    manifest = _read_jsonl(run_dir / "manifest.jsonl")
    if sha256_bytes((run_dir / "plan.jsonl").read_bytes()) != run.get("plan", {}).get("sha256"):
        raise ValueError("Plan SHA-256 mismatch")
    if require_complete and len(plan) != len(manifest):
        raise ValueError(f"Incomplete manifest: {len(manifest)}/{len(plan)}")
    expected = {row["pair_id"]: row for row in plan}
    if len(expected) != len(plan):
        raise ValueError("Duplicate pair ID in plan")
    previous_pair_index = -1
    committed = set()
    resolution = tuple(map(int, run["render"]["resolution"]))
    for row in manifest:
        pair_id = row.get("pair_id")
        if pair_id not in expected or pair_id in committed:
            raise ValueError(f"Unknown/duplicate pair in manifest: {pair_id}")
        pair_index = int(row.get("pair_index", -1))
        if pair_index <= previous_pair_index:
            raise ValueError("Manifest is not strictly ordered by pair_index")
        previous_pair_index = pair_index
        validate_manifest_matches_plan(row, expected[pair_id])
        validate_pair_artifacts(run_dir, row, resolution)
        committed.add(pair_id)
    _inventory(run_dir, [row["pair_id"] for row in manifest])
    preflight_path = run_dir / "anomaly_preflight.json"
    if not preflight_path.is_file():
        raise ValueError("Missing anomaly_preflight.json")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("ok") is not True or int(preflight.get("tested_count", -1)) != 360 or int(preflight.get("failed_count", -1)) != 0:
        raise ValueError("Anomaly 360-stratum preflight did not pass")
    counts = run.get("counts", {})
    if int(counts.get("planned_pairs", -1)) != len(plan) or int(counts.get("completed_pairs", -1)) != len(manifest):
        raise ValueError("run.json count mismatch")
    if require_complete:
        if int(run.get("performance", {}).get("render_call_count", -1)) != 2 * len(plan):
            raise ValueError("Render-call count must equal two per pair")
        source = (REPO_ROOT / run["source"]["path"]).resolve()
        if _sha256(source) != run["source"]["sha256"] or run["source"].get("sha256_after") != run["source"]["sha256"]:
            raise ValueError("Immutable source checksum mismatch")
    contrast_counts = {"normal_contrast": 0, "low_contrast": 0}
    for row in manifest:
        contrast_counts[row["gates"]["mask_and_photometric"]["photometric_status"]] += 1
    return {"ok": True, "planned_pairs": len(plan), "completed_pairs": len(manifest), "rgb_count": 2 * len(manifest), "target_mask_count": len(manifest), "anomaly_mask_count": len(manifest), "photometric_status": contrast_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_anomaly_batch_run(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
