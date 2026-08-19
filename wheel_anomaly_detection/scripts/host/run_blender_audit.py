from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "blender" / "audit_asset.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "blender" / "audit.json"
WINDOWS_BLENDER = Path(
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blender_audit.validation import validate_output


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
        description="Host-Python launcher for the headless Blender asset audit."
    )
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run the host-side output validator after Blender exits.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asset = args.asset.expanduser().resolve()
    if not asset.is_file():
        print(f"ERROR: asset not found: {asset}", file=sys.stderr)
        return 2
    if asset.suffix.lower() != ".glb":
        print(f"ERROR: expected a .glb asset, got: {asset}", file=sys.stderr)
        return 2
    if not args.config.expanduser().resolve().is_file():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
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
        "--output-dir",
        str(output_dir),
        "--config",
        str(args.config.expanduser().resolve()),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        print(
            f"ERROR: Blender audit exited with code {completed.returncode}",
            file=sys.stderr,
        )
        return completed.returncode

    if args.skip_validation:
        return 0
    validation = validate_output(output_dir)
    for warning in validation["warnings"]:
        print(f"WARNING: {warning}")
    for error in validation["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    print(
        f"Validated {validation['checked_file_count']} files: "
        f"{'OK' if validation['valid'] else 'FAILED'}"
    )
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
