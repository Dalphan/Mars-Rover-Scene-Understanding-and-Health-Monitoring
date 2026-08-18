"""Reopen validation for the persisted wheel-roll pose-sampling asset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.build_report.resolve(), args.output_report.resolve()


def _matrix(values: list[float]) -> Matrix:
    return Matrix([values[index:index + 4] for index in range(0, 16, 4)])


def validate(config_path: Path, build_report_path: Path, output_report_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.wheel_preparation.pose_sampling import validate_pose_sampling_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(build_report_path.read_text(encoding="utf-8"))
    validate_pose_sampling_config(config)
    errors = []
    scene = bpy.context.scene
    if not bool(scene.get("wheel_pose_sampling_ready", False)):
        errors.append("Scene is not marked pose-sampling ready")
    if scene.get("wheel_pose_sampling_config_sha256") != report["config_sha256"]:
        errors.append("Scene/config signature mismatch")
    if int(scene.get("microterrain_level", 0)) != int(config["source"]["required_microterrain_level"]):
        errors.append("Microterrain Level-3 marker was not preserved")
    for name in config["wheels"]:
        wheel = bpy.data.objects.get(name)
        if wheel is None or wheel.type != "MESH":
            errors.append(f"Missing wheel {name}")
            continue
        metadata = report["wheel_metadata"][name]
        if not bool(wheel.get("pose_sampling_ready", False)):
            errors.append(f"Wheel {name} is not marked pose-sampling ready")
        if wheel.get("pose_sampling_roll_axis_local") != "X":
            errors.append(f"Wheel {name} lost its local-X roll contract")
        expected = _matrix(metadata["base_matrix_local"])
        maximum_delta = max(abs(float(wheel.matrix_local[row][column] - expected[row][column])) for row in range(4) for column in range(4))
        if maximum_delta > 2e-6:
            errors.append(f"Wheel {name} was not persisted at its base transform")
        if not wheel.data.uv_layers or not wheel.data.materials:
            errors.append(f"Wheel {name} lost UVs or materials")
    if not bool(report.get("physical_validation", {}).get("ok")):
        errors.append("Build report physical gate failed")
    if not bool(report.get("anomaly_validation", {}).get("ok")):
        errors.append("Build report anomaly visibility gate failed")
    result = {
        "ok": not errors,
        "blend": bpy.data.filepath,
        "errors": errors,
        "wheel_count": len(config["wheels"]),
        "normal_phase_count": len(report["normal_samples"]),
        "anomaly_gate_sample_count": report["anomaly_validation"]["sample_count"],
        "manual_visual_check_required": False,
    }
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError("Persisted pose-sampling validation failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    validate(*_arguments())
