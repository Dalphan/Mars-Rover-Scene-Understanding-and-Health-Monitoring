"""Reproducible launcher for the microterrain Level-3 checkpoint."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "blender" / "gale_terrain.json")
    parser.add_argument("--source-blend", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "microterrain" / "level2" / "microterrain_L2.blend")
    parser.add_argument("--level2-report", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "microterrain" / "level2" / "build_L2.json")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "microterrain" / "level3")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    build_script = repo_root / "scripts" / "blender" / "microterrain" / "build_microterrain_level3.py"
    validate_script = repo_root / "scripts" / "blender" / "microterrain" / "validate_microterrain_level3.py"
    _run([
        str(args.blender_executable.resolve()), "--background", str(args.source_blend.resolve()), "--python", str(build_script), "--",
        "--config", str(args.config.resolve()), "--source-blend", str(args.source_blend.resolve()),
        "--level2-report", str(args.level2_report.resolve()), "--output-dir", str(output),
    ], repo_root)
    blend = output / "microterrain_L3.blend"
    build_report = output / "build_L3.json"
    if not blend.is_file() or not build_report.is_file():
        raise RuntimeError("Blender did not produce the required Level-3 Blend asset and build report")
    _run([
        str(args.blender_executable.resolve()), "--background", str(blend), "--python", str(validate_script), "--",
        "--config", str(args.config.resolve()), "--build-report", str(build_report),
        "--output-report", str(output / "validation_L3.json"),
    ], repo_root)


if __name__ == "__main__":
    main()
