from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "blender" / "prepare_wheel.py"
DEFAULT_BLEND_VALIDATOR = (
    REPO_ROOT / "scripts" / "blender" / "validate_wheel_blend.py"
)
DEFAULT_CONFIG = REPO_ROOT / "configs" / "blender" / "wheel_preparation.json"
WINDOWS_BLENDER = Path(
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.validation import validate_output


def find_blender(explicit: Path | None) -> Path:
    if explicit:
        candidate = explicit.expanduser().resolve()
    else:
        on_path = shutil.which("blender")
        candidate = Path(on_path) if on_path else WINDOWS_BLENDER
    if not candidate.is_file():
        raise FileNotFoundError(
            "Blender executable not found. Pass --blender with the full path."
        )
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a canonical Curiosity wheel and render a deterministic "
            "normal/perforation probe."
        )
    )
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--terrain", required=True, type=Path)
    parser.add_argument(
        "--candidate-id",
        default="wheel_candidate_05",
        help="Audited wheel candidate to extract; never changed automatically.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run the host-side validator after Blender exits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset = args.asset.expanduser().resolve()
    audit_report = args.audit_report.expanduser().resolve()
    terrain = args.terrain.expanduser().resolve()
    config = args.config.expanduser().resolve()
    for label, path in (
        ("asset", asset),
        ("audit report", audit_report),
        ("terrain", terrain),
        ("configuration", config),
    ):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2
    if asset.suffix.lower() != ".glb":
        print(f"ERROR: expected a .glb asset, got: {asset}", file=sys.stderr)
        return 2
    try:
        blender = find_blender(args.blender)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    command = [
        str(blender),
        "--background",
        "--python",
        str(DEFAULT_SCRIPT),
        "--",
        "--asset",
        str(asset),
        "--audit-report",
        str(audit_report),
        "--terrain",
        str(terrain),
        "--candidate-id",
        args.candidate_id,
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"ERROR: Blender wheel preparation exited with code "
            f"{completed.returncode}",
            file=sys.stderr,
        )
        return completed.returncode

    reopen_jobs = (
        (
            "canonical",
            output_dir / "diagnostics" / "wheel_canonical.blend",
            output_dir / "reports" / "canonical_reopen.json",
        ),
        (
            "perforation",
            output_dir / "diagnostics" / "perforation_probe.blend",
            output_dir / "reports" / "perforation_reopen.json",
        ),
    )
    for kind, blend_path, result_path in reopen_jobs:
        reopen_command = [
            str(blender),
            "--background",
            str(blend_path),
            "--python",
            str(DEFAULT_BLEND_VALIDATOR),
            "--",
            "--kind",
            kind,
            "--output",
            str(result_path),
        ]
        print("Reopening:", subprocess.list2cmdline(reopen_command), flush=True)
        reopened = subprocess.run(reopen_command, cwd=REPO_ROOT, check=False)
        if reopened.returncode != 0:
            print(
                f"ERROR: reopened {kind} blend validation exited with code "
                f"{reopened.returncode}",
                file=sys.stderr,
            )
            return reopened.returncode
    if args.skip_validation:
        base_valid = True
    else:
        result = validate_output(output_dir)
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        for error in result["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        print(
            f"Validated {result['checked_file_count']} files: "
            f"{'OK' if result['valid'] else 'FAILED'}"
        )
        base_valid = bool(result["valid"])

    return 0 if base_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
