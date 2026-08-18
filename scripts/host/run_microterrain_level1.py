"""Reproducible launcher for the microterrain Level 1 checkpoint."""

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
    parser.add_argument("--geospatial-dir", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "gale_terrain" / "geospatial")
    parser.add_argument("--source-blend", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "gale_terrain" / "scene" / "gale_terrain_scene.blend")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "microterrain" / "level1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    build_script = repo_root / "scripts" / "blender" / "microterrain" / "build_meso_relief.py"
    validate_script = repo_root / "scripts" / "blender" / "microterrain" / "validate_microterrain.py"
    _run([str(args.blender_executable.resolve()), "--background", str(args.source_blend.resolve()), "--python", str(build_script), "--", "--config", str(args.config.resolve()), "--geospatial-dir", str(args.geospatial_dir.resolve()), "--source-blend", str(args.source_blend.resolve()), "--output-dir", str(output)], repo_root)
    blend = output / "microterrain_L1.blend"
    _run([str(args.blender_executable.resolve()), "--background", str(blend), "--python", str(validate_script), "--", "--config", str(args.config.resolve()), "--build-report", str(output / "build_L1.json"), "--output-report", str(output / "validation_L1.json")], repo_root)


if __name__ == "__main__":
    main()
