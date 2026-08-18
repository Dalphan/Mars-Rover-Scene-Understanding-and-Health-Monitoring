"""Host-side report validation for microterrain Level 1."""

from __future__ import annotations


def validate_level1_report(config: dict, report: dict) -> list[str]:
    from .core import validate_level1_config

    errors: list[str] = []
    contract = validate_level1_config(config)
    metrics = report.get("metrics", {})
    patch = report.get("patch", {})
    if report.get("level") != 1:
        errors.append("Report is not Level 1")
    if patch.get("vertex_count") != contract["vertex_count"] or patch.get("face_count") != contract["face_count"]:
        errors.append("Patch topology does not match the configured grid")
    if not metrics.get("signature_sha256") or len(metrics["signature_sha256"]) != 64:
        errors.append("Missing deterministic displacement signature")
    if metrics.get("maximum_absolute_displacement_m", float("inf")) > float(config["microterrain"]["meso_relief"]["maximum_abs_displacement_m"]) + 1e-9:
        errors.append("Displacement exceeds the configured gate")
    if metrics.get("edge_max_abs_displacement_m", float("inf")) > 1e-8:
        errors.append("Patch edge is not continuous with the macroterrain")
    if metrics.get("clipped_fraction", 1.0) > 0.0001:
        errors.append("Too much relief was hard-clipped at the displacement gate")
    if report.get("macroterrain", {}).get("source_modified") is not False:
        errors.append("Macroterrain source must remain unmodified")
    if report.get("scope", {}).get("level2_clasts_enabled") or report.get("scope", {}).get("level3_shading_enabled"):
        errors.append("Later microterrain levels must remain disabled")
    renders = report.get("renders", {})
    for name in ("L0_macro_only", "L1_meso_relief", "L0_terrain_10cm", "L1_terrain_10cm", "L0_terrain_30cm", "L1_terrain_30cm"):
        if not renders.get(name):
            errors.append(f"Missing render record {name}")
    return errors
