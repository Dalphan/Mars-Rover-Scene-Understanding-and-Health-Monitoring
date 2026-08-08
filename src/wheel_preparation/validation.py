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
        errors.append(f"Cannot read JSON report {path}: {exc}")
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
        "canonical_reopen": root / "reports" / "canonical_reopen.json",
        "perforation_reopen": root / "reports" / "perforation_reopen.json",
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
    for label, expected_kind in (
        ("canonical_reopen", "canonical"),
        ("perforation_reopen", "perforation"),
    ):
        path = required[label]
        if not path.is_file():
            continue
        reopen = _load_report(path, errors)
        if not reopen:
            continue
        if reopen.get("kind") != expected_kind:
            errors.append(
                f"Reopen report kind {reopen.get('kind')!r}, expected {expected_kind!r}"
            )
        if reopen.get("status") != "completed" or reopen.get("valid") is not True:
            errors.append(f"Reopened {expected_kind} blend validation failed")
        if reopen.get("errors") not in ([], None):
            errors.append(f"Reopened {expected_kind} blend reports errors")
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
        inside_change = float(mask.get("inside_change_ratio", 0.0))
        minimum_inside_change = float(
            gates.get("minimum_inside_change_ratio", 0.20)
        )
        if inside_change < minimum_inside_change:
            errors.append(
                f"Inside-mask change ratio {inside_change} is below "
                f"{minimum_inside_change}"
            )
        mean_inside = float(mask.get("mean_inside_difference", 0.0))
        minimum_mean_inside = float(
            gates.get("minimum_inside_mean_difference", 0.01)
        )
        if mean_inside < minimum_mean_inside:
            errors.append(
                f"Mean inside-mask difference {mean_inside} is below "
                f"{minimum_mean_inside}"
            )
        if mask.get("images_identical") is not False:
            errors.append("Normal and anomaly renders are identical or not compared")

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
            "separation_renders",
            "perforation_renders",
        ):
            value = artifacts.get(key, [])
            if isinstance(value, list):
                image_paths.extend(str(item) for item in value)
        if len(artifacts.get("extraction_renders", [])) < 3:
            errors.append("Expected at least three extraction renders")
        if len(artifacts.get("separation_renders", [])) < 3:
            errors.append("Expected isolated skin, details, and composite renders")
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


def validate_perforation_library(output_dir: str | Path, *, expected_variants: int = 12) -> dict[str, Any]:
    """Validate the host-visible artifact contract of the irregular pilot."""

    root = Path(output_dir).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    summary_path = root / "reports" / "library.json"
    blend_candidates = (
        root / "rover_simulator_m6_irregular_perforation_library.blend",
        root / "rover_simulator_m6_irregular_perforation_pilot.blend",
        root / "perforation_library.blend",
    )
    blend_path = next((path for path in blend_candidates if path.is_file()), blend_candidates[0])
    for label, path in (("library summary", summary_path), ("library blend", blend_path)):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty {label}: {path}")
        else:
            checked.append(str(path))
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read library summary {summary_path}: {exc}")
    variants = summary.get("variants", []) if isinstance(summary, dict) else []
    if int(summary.get("variant_count", -1)) != expected_variants:
        errors.append(f"Expected {expected_variants} library variants")
    if len(variants) != expected_variants:
        errors.append(f"Library summary contains {len(variants)} variants")
    if int(summary.get("valid_count", -1)) != expected_variants:
        errors.append("Not every perforation library variant passed Blender gates")
    required_artifacts = tuple(
        summary.get(
            "required_artifacts",
            ("normal", "anomaly", "damage_mask", "effect_mask", "difference", "through_depth"),
        )
    )
    for item in variants:
        variant_meta = item.get("variant", {}) if isinstance(item, dict) else {}
        variant_id = str(item.get("variant_id") or variant_meta.get("variant_id", "unknown"))
        report_path = root / "reports" / f"{variant_id}.json"
        if not report_path.is_file() or report_path.stat().st_size == 0:
            errors.append(f"Missing report for library variant {variant_id}")
            continue
        checked.append(str(report_path))
        if item.get("valid") is not True:
            errors.append(f"Library variant {variant_id} is invalid")
        report = item
        artifacts = report.get("artifacts", {})
        for key in required_artifacts:
            relative = artifacts.get(key)
            if not relative:
                errors.append(f"Variant {variant_id} has no {key} artifact")
                continue
            path = root / str(relative)
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"Missing or empty {key} artifact for {variant_id}: {path}")
                continue
            checked.append(str(path))
            try:
                if read_png_dimensions(path) != (800, 600):
                    errors.append(f"Unexpected render size for {variant_id} {key}")
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        through = report.get("through_visibility", {})
        if through.get("valid") is not True or float(through.get("visible_fraction", 0.0)) < 1.0:
            errors.append(f"Variant {variant_id} failed through-visibility gate")
        boolean = report.get("boolean", {})
        if boolean.get("success") is not True or boolean.get("closed_manifold") is not True:
            errors.append(f"Variant {variant_id} failed closed-manifold Boolean gate")
        if float(report.get("open_area_fraction_after_folds", 0.0)) < 0.60:
            errors.append(f"Variant {variant_id} leaves less than 60% of the aperture open")
        image_metrics = report.get("image_metrics", {})
        maximum_effect = float(report.get("counterfactual_gate", {}).get("maximum_effect_area_ratio", 0.12))
        if float(image_metrics.get("effect_area_ratio", 1.0)) > maximum_effect:
            errors.append(f"Variant {variant_id} has excessive raw counterfactual drift")
    if bool(summary.get("contact_sheet_required", True)):
        contact_sheet = root / "renders" / "contact_sheet.png"
        if not contact_sheet.is_file() or contact_sheet.stat().st_size == 0:
            errors.append(f"Missing or empty contact sheet: {contact_sheet}")
        else:
            checked.append(str(contact_sheet))
    return {
        "valid": not errors,
        "output_dir": str(root),
        "checked_file_count": len(set(checked)),
        "errors": errors,
        "warnings": warnings,
    }
