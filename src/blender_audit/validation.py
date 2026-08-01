from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        signature = stream.read(8)
        if signature != PNG_SIGNATURE:
            raise ValueError(f"{path} is not a PNG file")
        length = struct.unpack(">I", stream.read(4))[0]
        chunk_type = stream.read(4)
        if chunk_type != b"IHDR" or length < 8:
            raise ValueError(f"{path} has no valid IHDR chunk")
        width, height = struct.unpack(">II", stream.read(8))
    return width, height


def _load_report(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read JSON report {path}: {exc}")
        return {}
    if not isinstance(report, dict):
        errors.append("JSON report root must be an object")
        return {}
    return report


def validate_output(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked_files: list[str] = []

    required = {
        "json_report": root / "reports" / "audit.json",
        "markdown_report": root / "reports" / "audit.md",
        "diagnostic_blend": root / "diagnostics" / "imported_asset.blend",
        "audit_log": root / "logs" / "audit.log",
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty {label}: {path}")
        else:
            checked_files.append(str(path))

    report = (
        _load_report(required["json_report"], errors)
        if required["json_report"].is_file()
        else {}
    )
    if report:
        if report.get("status") != "completed":
            errors.append(f"Audit status is {report.get('status')!r}, expected 'completed'")
        schema_version = report.get("schema_version")
        if not isinstance(schema_version, str) or not schema_version:
            errors.append("Missing schema_version")

        asset = report.get("asset", {})
        if asset.get("sha256_before") != asset.get("sha256_after"):
            errors.append("Asset checksums differ before and after the audit")
        if asset.get("unchanged") is not True:
            errors.append("Asset is not explicitly marked unchanged")

        environment = report.get("environment", {})
        resolution = environment.get("render_resolution", {})
        expected_size = (
            int(resolution.get("width", 800)),
            int(resolution.get("height", 600)),
        )

        artifacts = report.get("artifacts", {})
        rover_renders = artifacts.get("rover_renders", [])
        if len(rover_renders) < 6:
            errors.append(f"Expected at least 6 rover renders, found {len(rover_renders)}")
        topology_renders = artifacts.get("topology_renders", [])
        if not topology_renders:
            errors.append("No topology/wireframe render recorded")

        candidates = report.get("wheel_detection", {}).get("candidates", [])
        for candidate in candidates:
            candidate_renders = candidate.get("renders", {}).get("closeups", [])
            if len(candidate_renders) < 3:
                errors.append(
                    f"{candidate.get('id', 'candidate')} has fewer than 3 close-up renders"
                )

        image_paths = list(rover_renders) + list(topology_renders)
        for candidate in candidates:
            candidate_artifacts = candidate.get("renders", {})
            image_paths.extend(candidate_artifacts.get("closeups", []))
            wireframe = candidate_artifacts.get("wireframe")
            if wireframe:
                image_paths.append(wireframe)

        seen: set[str] = set()
        for relative in image_paths:
            if relative in seen:
                continue
            seen.add(relative)
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing or empty render: {path}")
                continue
            checked_files.append(str(path))
            try:
                size = read_png_dimensions(path)
            except (OSError, ValueError, struct.error) as exc:
                errors.append(str(exc))
                continue
            if size != expected_size:
                errors.append(
                    f"Unexpected render size for {path}: {size}, expected {expected_size}"
                )

        category = report.get("classification", {}).get("category")
        if category not in {"A", "B", "C", "D"}:
            errors.append(f"Invalid or missing classification category: {category!r}")
        if not candidates and category != "D":
            errors.append("No candidates were found but classification is not D")

        missing_textures = report.get("materials", {}).get("missing_textures", [])
        if missing_textures:
            warnings.append(f"Audit reports {len(missing_textures)} missing texture(s)")

    return {
        "valid": not errors,
        "output_dir": str(root),
        "checked_file_count": len(set(checked_files)),
        "errors": errors,
        "warnings": warnings,
    }
