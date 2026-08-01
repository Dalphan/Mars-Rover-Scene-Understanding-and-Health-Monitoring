from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.blender_audit.validation import read_png_dimensions


def _load_report(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read preparation report {path}: {exc}")
        return {}
    if not isinstance(report, dict):
        errors.append("Preparation report root must be an object")
        return {}
    return report


def validate_output(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    required = {
        "json_report": root / "reports" / "preparation.json",
        "markdown_report": root / "reports" / "preparation.md",
        "canonical_blend": root / "diagnostics" / "wheel_canonical.blend",
        "perforation_blend": root / "diagnostics" / "perforation_probe.blend",
        "log": root / "logs" / "preparation.log",
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty {label}: {path}")
        else:
            checked.append(str(path))

    report = (
        _load_report(required["json_report"], errors)
        if required["json_report"].is_file()
        else {}
    )
    if report:
        if report.get("status") != "completed":
            errors.append(
                f"Preparation status is {report.get('status')!r}, expected 'completed'"
            )
        asset = report.get("asset", {})
        if asset.get("sha256_before") != asset.get("sha256_after"):
            errors.append("Asset checksum changed during wheel preparation")
        if asset.get("unchanged") is not True:
            errors.append("Asset is not explicitly marked unchanged")

        extraction = report.get("extraction", {})
        expected = report.get("candidate", {}).get("expected_counts", {})
        for key in ("vertex_count", "face_count", "component_count"):
            actual_value = extraction.get(key)
            expected_value = expected.get(key)
            if expected_value is not None and actual_value != expected_value:
                errors.append(
                    f"Extraction {key}={actual_value}, expected {expected_value}"
                )
        for key in (
            "coordinates_preserved",
            "materials_preserved",
            "uv_preserved",
            "custom_normals_preserved",
        ):
            if extraction.get(key) is not True:
                errors.append(f"Extraction gate failed: {key}")
        minimum_iou = float(extraction.get("minimum_silhouette_iou", 0.0))
        required_iou = float(
            report.get("configuration", {})
            .get("gates", {})
            .get("minimum_silhouette_iou", 0.995)
        )
        if minimum_iou < required_iou:
            errors.append(
                f"Silhouette IoU {minimum_iou} is below required {required_iou}"
            )

        final_topology = report.get("repair", {}).get("final_topology", {})
        if int(final_topology.get("boundary_edge_count", -1)) != 0:
            errors.append("Prepared skin has boundary edges")
        if int(final_topology.get("multi_face_edge_count", -1)) != 0:
            errors.append("Prepared skin has multi-face edges")
        if final_topology.get("normals_consistent") is not True:
            errors.append("Prepared skin normals are inconsistent")

        boolean = report.get("perforation", {}).get("boolean", {})
        if boolean.get("success") is not True:
            errors.append("Boolean Difference did not succeed")
        if boolean.get("closed_manifold") is not True:
            errors.append("Perforated skin is not closed manifold")

        mask = report.get("perforation", {}).get("mask", {})
        ratio = float(mask.get("area_ratio", -1.0))
        gates = report.get("configuration", {}).get("gates", {})
        minimum_ratio = float(gates.get("minimum_mask_ratio", 0.002))
        maximum_ratio = float(gates.get("maximum_mask_ratio", 0.05))
        if not minimum_ratio <= ratio <= maximum_ratio:
            errors.append(
                f"Mask area ratio {ratio} outside [{minimum_ratio}, {maximum_ratio}]"
            )
        outside_change = float(mask.get("outside_change_ratio", 1.0))
        maximum_outside = float(gates.get("maximum_outside_change_ratio", 0.01))
        if outside_change > maximum_outside:
            errors.append(
                f"Outside-mask change ratio {outside_change} exceeds {maximum_outside}"
            )

        if report.get("validation", {}).get("valid") is not True:
            errors.append("Blender-side validation is not marked valid")

        environment = report.get("environment", {})
        resolution = environment.get("render_resolution", {})
        expected_size = (
            int(resolution.get("width", 800)),
            int(resolution.get("height", 600)),
        )
        artifacts = report.get("artifacts", {})
        image_paths: list[str] = []
        for key in (
            "extraction_renders",
            "topology_renders",
            "perforation_renders",
        ):
            value = artifacts.get(key, [])
            if isinstance(value, list):
                image_paths.extend(str(item) for item in value)
        if len(artifacts.get("extraction_renders", [])) < 3:
            errors.append("Expected at least three extraction renders")
        required_pair_names = {
            "renders/perforation/normal.png",
            "renders/perforation/anomaly.png",
            "renders/perforation/anomaly_mask.png",
            "renders/perforation/difference.png",
        }
        if not required_pair_names.issubset(set(image_paths)):
            errors.append("Perforation artifact list is incomplete")

        for relative in dict.fromkeys(image_paths):
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing or empty render: {path}")
                continue
            checked.append(str(path))
            try:
                actual_size = read_png_dimensions(path)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
                continue
            if actual_size != expected_size:
                errors.append(
                    f"Unexpected render size {actual_size} for {path}; "
                    f"expected {expected_size}"
                )

        if report.get("repair", {}).get("strategy") == "hybrid_parametric_shell":
            warnings.append(
                "The prepared skin uses the documented hybrid parametric-shell fallback"
            )

    return {
        "valid": not errors,
        "output_dir": str(root),
        "checked_file_count": len(set(checked)),
        "errors": errors,
        "warnings": warnings,
    }
