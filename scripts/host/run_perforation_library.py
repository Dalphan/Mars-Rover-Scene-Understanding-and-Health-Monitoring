from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wheel_preparation.validation import validate_perforation_library


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "scripts" / "blender" / "build_perforation_library.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "blender" / "perforation_library.json"
WINDOWS_BLENDER_CANDIDATES = (
    Path(r"D:\Programmi\Blender Foundation\Blender 5.2\blender.exe"),
    Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
)


def find_blender(explicit: Path | None) -> Path:
    if explicit:
        candidates = (explicit.expanduser().resolve(),)
    else:
        on_path = shutil.which("blender")
        candidates = ((Path(on_path),) if on_path else ()) + WINDOWS_BLENDER_CANDIDATES
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Blender 5.2 executable not found. Pass --blender with the full path.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the M6 irregular-perforation review library.")
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--library-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", help="Render only one variant id; omit for all 12")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--blender", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_blend.expanduser().resolve()
    config = args.library_config.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not source.is_file():
        print(f"ERROR: source blend not found: {source}", file=sys.stderr)
        return 2
    if not config.is_file():
        print(f"ERROR: library configuration not found: {config}", file=sys.stderr)
        return 2
    try:
        blender = find_blender(args.blender)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    command = [
        str(blender), "--background", str(source), "--python", str(DEFAULT_SCRIPT), "--",
        "--output-dir", str(output), "--library-config", str(config),
    ]
    if args.profile:
        command.extend(("--profile", args.profile))
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    print("Running:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    result = validate_perforation_library(output, expected_variants=1 if args.profile else 12)
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    for error in result["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"Validated {result['checked_file_count']} files: {'OK' if result['valid'] else 'FAILED'}")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
