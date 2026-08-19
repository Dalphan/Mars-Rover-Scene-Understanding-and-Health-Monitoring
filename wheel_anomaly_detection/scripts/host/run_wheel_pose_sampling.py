"""Build and reopen-validate the deterministic wheel-roll pose-sampling asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "blender" / "wheel_pose_sampling.json")
    parser.add_argument("--camera-config", type=Path, default=repo_root / "configs" / "blender" / "wheel_camera_poses.json")
    parser.add_argument("--source-blend", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "microterrain" / "level3" / "microterrain_L3.blend")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "pose_sampling")
    args = parser.parse_args()
    source = args.source_blend.resolve()
    source_before = _sha256(source)
    output = args.output_dir.resolve()
    build_script = repo_root / "scripts" / "blender" / "build_wheel_pose_sampling.py"
    validate_script = repo_root / "scripts" / "blender" / "validate_wheel_pose_sampling.py"
    _run([
        str(args.blender_executable.resolve()), "--background", str(source), "--python", str(build_script), "--",
        "--config", str(args.config.resolve()), "--camera-config", str(args.camera_config.resolve()),
        "--source-blend", str(source), "--source-sha256", source_before, "--output-dir", str(output),
    ], repo_root)
    report = output / "build_pose_sampling.json"
    if not report.is_file():
        raise RuntimeError("Blender did not produce the pose-sampling build report")
    build_result = json.loads(report.read_text(encoding="utf-8"))
    if not bool(build_result.get("validation", {}).get("ok", False)):
        raise RuntimeError("Blender pose-sampling build report failed its hard gates")
    if _sha256(source) != source_before:
        raise RuntimeError("Source Level-3 Blend changed during pose-sampling build")
    blend = output / "wheel_roll_pose_sampling.blend"
    if not blend.is_file() or not report.is_file():
        raise RuntimeError("Blender did not produce the pose-sampling asset and build report")
    _run([
        str(args.blender_executable.resolve()), "--background", str(blend), "--python", str(validate_script), "--",
        "--config", str(args.config.resolve()), "--build-report", str(report),
        "--output-report", str(output / "validation_pose_sampling.json"),
    ], repo_root)
    validation_path = output / "validation_pose_sampling.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not bool(validation.get("ok", False)):
        raise RuntimeError("Persisted pose-sampling asset failed reopen validation")
    if _sha256(source) != source_before:
        raise RuntimeError("Source Level-3 Blend changed during reopen validation")
    print(f"Pose-sampling asset: {blend}")


if __name__ == "__main__":
    main()
