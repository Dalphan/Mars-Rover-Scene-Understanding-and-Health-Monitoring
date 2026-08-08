from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILDER = REPO_ROOT / "scripts" / "blender" / "build_rover_simulator.py"
DEFAULT_VALIDATOR = REPO_ROOT / "scripts" / "blender" / "validate_rover_simulator.py"
PAIR_VALIDATOR = REPO_ROOT / "scripts" / "blender" / "validate_wheel_blend.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "blender" / "rover_simulator.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate milestone 1 of the rover simulator."
    )
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--preparation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_blender(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Blender executable not found: {path}")
    discovered = shutil.which("blender")
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError("Blender not found; provide --blender")


def run(command: list[str], label: str) -> None:
    print(f"{label}: {subprocess.list2cmdline(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} exited with code {completed.returncode}")


def main() -> int:
    args = parse_args()
    source_blend = args.source_blend.expanduser().resolve()
    audit_report = args.audit_report.expanduser().resolve()
    preparation_report = args.preparation_report.expanduser().resolve()
    config = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for label, path in (
        ("source blend", source_blend),
        ("audit report", audit_report),
        ("preparation report", preparation_report),
        ("configuration", config),
        ("Blender builder", DEFAULT_BUILDER),
        ("simulator validator", DEFAULT_VALIDATOR),
        ("pair validator", PAIR_VALIDATOR),
    ):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2
    try:
        blender = find_blender(args.blender)
        source_hash_before = sha256_file(source_blend)
        build_command = [
            str(blender),
            "--background",
            str(source_blend),
            "--python",
            str(DEFAULT_BUILDER),
            "--",
            "--source-blend",
            str(source_blend),
            "--audit-report",
            str(audit_report),
            "--preparation-report",
            str(preparation_report),
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
        ]
        run(build_command, "Building simulator")
        simulator_blend = output_dir / "diagnostics" / "rover_simulator.blend"
        simulator_report = output_dir / "reports" / "simulator_reopen.json"
        pair_report = output_dir / "reports" / "pair_reopen.json"
        run(
            [
                str(blender),
                "--background",
                str(simulator_blend),
                "--python",
                str(DEFAULT_VALIDATOR),
                "--",
                "--output",
                str(simulator_report),
            ],
            "Reopening simulator",
        )
        run(
            [
                str(blender),
                "--background",
                str(simulator_blend),
                "--python",
                str(PAIR_VALIDATOR),
                "--",
                "--kind",
                "perforation",
                "--output",
                str(pair_report),
            ],
            "Revalidating counterfactual pair",
        )
        expected = [
            simulator_blend,
            output_dir / "renders" / "simulator_overview.png",
            output_dir / "renders" / "wheel_closeup.png",
            output_dir / "reports" / "build.json",
            simulator_report,
            pair_report,
        ]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError("Missing simulator artifacts: " + ", ".join(missing))
        for report_path in (output_dir / "reports" / "build.json", simulator_report, pair_report):
            with report_path.open("r", encoding="utf-8") as stream:
                report = json.load(stream)
            if report.get("status") != "completed" or report.get("valid") is False:
                raise RuntimeError(f"Validation report failed: {report_path}")
        source_hash_after = sha256_file(source_blend)
        if source_hash_after != source_hash_before:
            raise RuntimeError("Source perforation blend changed during simulator build")
        print(
            f"Rover simulator milestone 1 validated: {len(expected)} artifacts, "
            "source unchanged"
        )
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
