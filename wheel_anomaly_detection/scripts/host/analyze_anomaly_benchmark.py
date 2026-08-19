"""Compare validated anomaly benchmark runs without trusting run wall clocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


LOCKED_RESOLVED = ("camera_matrix_world", "wheel_matrix_world", "rover_matrix_world", "roll_degrees")


def _rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[np.ndarray], percentile: float) -> float:
    return float(np.percentile(np.concatenate([value.reshape(-1) for value in values]), percentile))


def compare(reference: Path, candidate: Path) -> dict:
    ref_rows, cand_rows = _rows(reference), _rows(candidate)
    if [row["pair_id"] for row in ref_rows] != [row["pair_id"] for row in cand_rows]:
        raise ValueError("Pair order differs")
    semantic_equal = all(
        left["sampling"] == right["sampling"]
        and left["anomaly"] == right["anomaly"]
        and all(left["resolved"][key] == right["resolved"][key] for key in LOCKED_RESOLVED)
        for left, right in zip(ref_rows, cand_rows, strict=True)
    )
    rgb_differences: list[np.ndarray] = []
    small_differences: list[np.ndarray] = []
    mask_total = mask_equal = 0
    mask_stats = {artifact: {"xor": [], "iou": []} for artifact in ("target_wheel_mask", "anomaly_mask")}
    rgb_total = rgb_byte_equal = 0
    reference_bytes = candidate_bytes = 0
    for left, right in zip(ref_rows, cand_rows, strict=True):
        for artifact in ("clean_rgb", "anomaly_rgb"):
            left_path = reference / left["artifacts"][artifact]["path"]
            right_path = candidate / right["artifacts"][artifact]["path"]
            left_pixels = np.asarray(Image.open(left_path).convert("RGB"), dtype=np.int16)
            right_pixels = np.asarray(Image.open(right_path).convert("RGB"), dtype=np.int16)
            delta = np.abs(left_pixels - right_pixels)
            rgb_differences.append(delta)
            if left["anomaly"]["severity"] == "small":
                small_differences.append(delta)
            rgb_total += 1
            rgb_byte_equal += _sha(left_path) == _sha(right_path)
            reference_bytes += left_path.stat().st_size
            candidate_bytes += right_path.stat().st_size
        for artifact in ("target_wheel_mask", "anomaly_mask"):
            left_path = reference / left["artifacts"][artifact]["path"]
            right_path = candidate / right["artifacts"][artifact]["path"]
            left_mask = np.asarray(Image.open(left_path)) > 0
            right_mask = np.asarray(Image.open(right_path)) > 0
            xor = int(np.count_nonzero(left_mask != right_mask))
            union = int(np.count_nonzero(left_mask | right_mask))
            intersection = int(np.count_nonzero(left_mask & right_mask))
            mask_total += 1
            mask_equal += xor == 0
            mask_stats[artifact]["xor"].append(xor)
            mask_stats[artifact]["iou"].append(1.0 if union == 0 else intersection / union)
    all_delta = np.concatenate([value.reshape(-1) for value in rgb_differences])
    mse = float(np.mean(np.square(all_delta.astype(np.float64))))
    pair_seconds = sum(float(row["render"]["total_seconds"]) for row in cand_rows)
    reference_pair_seconds = sum(float(row["render"]["total_seconds"]) for row in ref_rows)
    post_gates = [row["gates"]["mask_and_photometric"] for row in cand_rows]
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "pair_count": len(cand_rows),
        "semantic_lock_equal": semantic_equal,
        "mask_pixel_equal": {"equal": mask_equal, "total": mask_total},
        "mask_difference": {
            artifact: {
                "mean_xor_pixels": float(np.mean(values["xor"])),
                "maximum_xor_pixels": int(max(values["xor"])),
                "minimum_iou": float(min(values["iou"])),
                "mean_iou": float(np.mean(values["iou"])),
            }
            for artifact, values in mask_stats.items()
        },
        "rgb_byte_equal": {"equal": rgb_byte_equal, "total": rgb_total},
        "rgb_difference_8bit": {
            "mean_absolute": float(np.mean(all_delta)),
            "p95_absolute": _percentile(rgb_differences, 95),
            "p99_absolute": _percentile(rgb_differences, 99),
            "maximum_absolute": int(np.max(all_delta)),
            "fraction_channels_ge_6": float(np.mean(all_delta >= 6)),
            "psnr_db": None if mse == 0 else float(20.0 * math.log10(255.0 / math.sqrt(mse))),
            "small_mean_absolute": float(np.mean(np.concatenate([value.reshape(-1) for value in small_differences]))) if small_differences else None,
        },
        "timing": {
            "reference_pair_pipeline_seconds": reference_pair_seconds,
            "candidate_pair_pipeline_seconds": pair_seconds,
            "saved_seconds": reference_pair_seconds - pair_seconds,
            "saved_fraction": (reference_pair_seconds - pair_seconds) / reference_pair_seconds,
            "speedup": reference_pair_seconds / pair_seconds,
        },
        "rgb_storage": {
            "reference_bytes": reference_bytes,
            "candidate_bytes": candidate_bytes,
            "candidate_fraction": candidate_bytes / reference_bytes,
        },
        "gate_margins": {
            "minimum_changed_fraction": min(float(gate["changed_fraction_inside_mask"]) for gate in post_gates),
            "minimum_median_delta_8bit": min(int(gate["median_delta_8bit"]) for gate in post_gates),
            "minimum_local_energy_fraction": min(float(gate["difference_energy_local_fraction"]) for gate in post_gates),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "comparisons": [compare(args.reference.resolve(), path.resolve()) for path in args.candidate]}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
