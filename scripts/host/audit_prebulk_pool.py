"""Audit the reusable pre-bulk pool and its canonical raster artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _counter(rows: list[dict], field: str, *, image_weighted: bool) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if field in row["sampling"]:
            counts[str(row["sampling"][field])] += len(row["members"]) if image_weighted else 1
    return dict(sorted(counts.items()))


def _expected_artifact_paths(rows: list[dict]) -> dict[Path, list[tuple[str, str]]]:
    result: dict[Path, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        for member in row["members"]:
            for kind in ("rgb", "target_wheel_mask", "anomaly_mask"):
                if kind in member:
                    artifact = member[kind]
                    result[REPO_ROOT / artifact["path"]].append((kind, artifact["sha256"]))
    return result


def audit(pool_dir: Path, clean_run: Path, anomaly_run: Path) -> dict:
    pool_meta = json.loads((pool_dir / "prebulk.json").read_text(encoding="utf-8"))
    pool_manifest = pool_dir / "manifest.jsonl"
    rows = _read_jsonl(pool_manifest)
    clean_rows = _read_jsonl(clean_run / "manifest.jsonl")
    anomaly_rows = _read_jsonl(anomaly_run / "manifest.jsonl")
    artifacts = _expected_artifact_paths(rows)

    errors: list[str] = []
    warnings: list[str] = []
    expected_manifest_sha = pool_meta["manifest"]["sha256"]
    actual_manifest_sha = _sha256(pool_manifest)
    if actual_manifest_sha != expected_manifest_sha:
        errors.append("Pool manifest SHA-256 does not match prebulk.json")
    if len(rows) != int(pool_meta["units"]):
        errors.append("Pool unit count does not match prebulk.json")

    unit_ids = [row["unit_id"] for row in rows]
    pair_locks = [row["pair_lock_id"] for row in rows if row["unit_type"] == "paired_clean_hole"]
    if len(unit_ids) != len(set(unit_ids)):
        errors.append("Duplicate unit_id in pool manifest")
    if len(pair_locks) != len(set(pair_locks)):
        errors.append("Duplicate pair_lock_id in pool manifest")

    byte_hash_groups: dict[str, list[str]] = defaultdict(list)
    pixel_hash_groups: dict[str, list[str]] = defaultdict(list)
    modes: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    total_bytes = 0
    unique_paths = set()
    for path, references in sorted(artifacts.items(), key=lambda item: item[0].as_posix()):
        unique_paths.add(path.resolve())
        if not path.is_file():
            errors.append(f"Missing artifact: {path.relative_to(REPO_ROOT).as_posix()}")
            continue
        actual_sha = _sha256(path)
        for kind, expected_sha in references:
            if actual_sha != expected_sha:
                errors.append(f"Checksum mismatch: {path.relative_to(REPO_ROOT).as_posix()}")
            kinds[kind] += 1
        rel = path.relative_to(REPO_ROOT).as_posix()
        byte_hash_groups[actual_sha].append(rel)
        total_bytes += path.stat().st_size
        try:
            with Image.open(path) as image:
                image.load()
                modes[f"{image.mode}:{image.width}x{image.height}"] += 1
                if image.size != (1200, 900):
                    errors.append(f"Wrong dimensions: {rel}: {image.size}")
                kind_set = {kind for kind, _sha in references}
                if "rgb" in kind_set:
                    if image.mode != "RGB":
                        errors.append(f"RGB artifact has mode {image.mode}: {rel}")
                else:
                    if image.mode != "L":
                        errors.append(f"Mask artifact has mode {image.mode}: {rel}")
                    extrema = image.getextrema()
                    values = set(image.getdata())
                    if values != {0, 255}:
                        errors.append(f"Mask is not non-empty binary 0/255: {rel}; extrema={extrema}")
                pixel_key = f"{image.mode}:{image.size}:{hashlib.sha256(image.tobytes()).hexdigest()}"
                pixel_hash_groups[pixel_key].append(rel)
        except Exception as error:  # Pillow reports the concrete decode failure.
            errors.append(f"Unreadable PNG {rel}: {error}")

    # Artifact roots are contractual. QA sheets and staging inputs are deliberately excluded.
    scanned = set()
    for root in (
        clean_run / "rgb",
        clean_run / "target_wheel_mask",
        anomaly_run / "rgb",
        anomaly_run / "target_wheel_mask",
        anomaly_run / "anomaly_mask",
    ):
        scanned.update(path.resolve() for path in root.rglob("*.png"))
    extras = sorted(path.relative_to(REPO_ROOT).as_posix() for path in scanned - unique_paths)
    if extras:
        errors.extend(f"Unreferenced canonical artifact: {path}" for path in extras)

    duplicate_bytes = [paths for paths in byte_hash_groups.values() if len(paths) > 1]
    duplicate_pixels = [paths for paths in pixel_hash_groups.values() if len(paths) > 1]
    rgb_duplicate_pixels = [group for group in duplicate_pixels if all("/rgb/" in path for path in group)]
    if rgb_duplicate_pixels:
        warnings.append(f"Found {len(rgb_duplicate_pixels)} groups of pixel-identical RGB files")

    standalone = [row for row in rows if row["unit_type"] == "standalone_clean"]
    paired = [row for row in rows if row["unit_type"] == "paired_clean_hole"]
    image_count = sum(len(row["members"]) for row in rows)
    source_clean_ids = {f"clean_{row['sample_id']}" for row in clean_rows}
    source_pair_ids = {f"pair_{row['pair_id']}" for row in anomaly_rows}
    pool_clean_ids = {row["unit_id"] for row in standalone}
    pool_pair_ids = {row["unit_id"] for row in paired}
    if source_clean_ids != pool_clean_ids:
        errors.append("Pool standalone units do not exactly match clean source manifest")
    if source_pair_ids != pool_pair_ids:
        errors.append("Pool paired units do not exactly match anomaly source manifest")

    anomaly_gate_minima = {
        "changed_fraction_inside_mask": min(row["gates"]["mask_and_photometric"]["changed_fraction_inside_mask"] for row in anomaly_rows),
        "median_delta_8bit": min(row["gates"]["mask_and_photometric"]["median_delta_8bit"] for row in anomaly_rows),
        "difference_energy_local_fraction": min(row["gates"]["mask_and_photometric"]["difference_energy_local_fraction"] for row in anomaly_rows),
        "inside_target_roi_fraction": min(row["gates"]["mask_and_photometric"]["inside_target_roi_fraction"] for row in anomaly_rows),
        "anomaly_area_px": min(row["gates"]["mask_and_photometric"]["area_px"] for row in anomaly_rows),
        "short_side_px": min(row["gates"]["mask_and_photometric"]["short_side_px"] for row in anomaly_rows),
        "terrain_clearance_after_m": min(row["gates"]["terrain_contact_details"]["minimum_clearance_after_m"] for row in anomaly_rows),
    }
    false_gates = []
    for row in anomaly_rows:
        for gate in ("framing", "terrain_contact", "terrain_coverage", "anomaly_visibility", "through_opening"):
            if row["gates"].get(gate) is not True:
                false_gates.append(f"{row['pair_id']}:{gate}")
        if row["gates"]["mask_and_photometric"].get("ok") is not True:
            false_gates.append(f"{row['pair_id']}:mask_and_photometric")
    if false_gates:
        errors.extend(f"False gate: {value}" for value in false_gates)

    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "pool": {
            "path": pool_dir.relative_to(REPO_ROOT).as_posix(),
            "manifest_sha256": actual_manifest_sha,
            "units": len(rows),
            "images": image_count,
            "standalone_clean_units": len(standalone),
            "paired_clean_hole_units": len(paired),
        },
        "artifacts": {
            "unique_files": len(unique_paths),
            "reference_counts": dict(sorted(kinds.items())),
            "total_bytes": total_bytes,
            "total_mib": total_bytes / (1024 * 1024),
            "image_contracts": dict(sorted(modes.items())),
            "unreferenced_files": extras,
        },
        "distributions": {
            "units": {field: _counter(rows, field, image_weighted=False) for field in ("target_wheel", "camera_pose", "lighting_preset", "surface_wear", "healthy_roll_degrees")},
            "images": {field: _counter(rows, field, image_weighted=True) for field in ("target_wheel", "camera_pose", "lighting_preset", "surface_wear", "healthy_roll_degrees")},
            "paired_units": {field: _counter(paired, field, image_weighted=False) for field in ("severity", "surface", "image_sector")},
        },
        "anomaly_gate_minima": anomaly_gate_minima,
        "duplicates": {
            "byte_identical_groups": duplicate_bytes,
            "pixel_identical_groups": duplicate_pixels,
            "rgb_pixel_identical_group_count": len(rgb_duplicate_pixels),
            "note": "Repeated references to a pair target mask are deduplicated before this analysis.",
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--anomaly-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((REPO_ROOT / "outputs").resolve()):
        raise ValueError("Audit report must be written below outputs/")
    report = audit(args.pool_dir.resolve(), args.clean_run.resolve(), args.anomaly_run.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": len(report["errors"]), "warnings": len(report["warnings"]), "output": str(output)}, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
