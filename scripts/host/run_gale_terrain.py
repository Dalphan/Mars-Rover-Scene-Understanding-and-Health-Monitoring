"""Run the deterministic HiRISE-to-Blender Gale terrain milestone."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], cwd: Path) -> None:
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--deps-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "blender" / "gale_terrain.json")
    parser.add_argument("--source-blend", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "curiosity_semantic_clean.blend")
    parser.add_argument("--output-root", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "gale_terrain")
    args = parser.parse_args()
    config = args.config.resolve()
    source_blend = args.source_blend.resolve()
    geospatial = (args.output_root / "geospatial").resolve()
    scene_output = (args.output_root / "scene").resolve()

    _run(
        [sys.executable, str(repo_root / "scripts" / "host" / "download_hirise.py"), "--config", str(config), "--data-root", str(args.data_root.resolve())],
        repo_root,
    )
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "host" / "crop_gale_product.py"),
            "--config",
            str(config),
            "--data-root",
            str(args.data_root.resolve()),
            "--output-dir",
            str(geospatial),
            "--deps-dir",
            str(args.deps_dir.resolve()),
        ],
        repo_root,
    )
    _run(
        [
            str(args.blender_executable.resolve()),
            "--background",
            str(source_blend),
            "--python",
            str(repo_root / "scripts" / "blender" / "build_gale_terrain.py"),
            "--",
            "--config",
            str(config),
            "--geospatial-dir",
            str(geospatial),
            "--source-blend",
            str(source_blend),
            "--output-dir",
            str(scene_output),
        ],
        repo_root,
    )
    blend = scene_output / "gale_terrain_scene.blend"
    _run(
        [
            str(args.blender_executable.resolve()),
            "--background",
            str(blend),
            "--python",
            str(repo_root / "scripts" / "blender" / "validate_gale_scene.py"),
            "--",
            "--config",
            str(config),
            "--report",
            str(scene_output / "validation_report.json"),
        ],
        repo_root,
    )


if __name__ == "__main__":
    main()
