"""Host-side validation for the Level-2 clast checkpoint."""

from __future__ import annotations

from .clasts import FAMILIES, validate_level2_config


REQUIRED_RENDERS = (
    "L1_meso_only",
    "L2_meso_clasts",
    "L1_terrain_10cm",
    "L2_terrain_10cm",
    "L1_terrain_30cm",
    "L2_terrain_30cm",
    "L2_patch_distribution",
)


def validate_level2_report(config: dict, report: dict) -> list[str]:
    contract = validate_level2_config(config)
    errors: list[str] = []
    if report.get("level") != 2:
        errors.append("Report is not Level 2")
    metrics = report.get("clasts", {})
    if metrics.get("counts") != contract["counts"] or metrics.get("total_instances") != contract["total_instances"]:
        errors.append("Clast counts do not match configured densities")
    if len(metrics.get("signature_sha256", "")) != 64:
        errors.append("Missing deterministic Level-2 scatter signature")
    signatures = metrics.get("family_signatures_sha256", {})
    if any(len(signatures.get(family, "")) != 64 for family in FAMILIES):
        errors.append("Missing deterministic family signature")
    configured = config["microterrain"]["clasts"]
    for family in FAMILIES:
        expected = [float(configured[family]["size_min_m"]), float(configured[family]["size_max_m"])]
        if metrics.get("size_ranges_m", {}).get(family) != expected:
            errors.append(f"Size range mismatch for {family}")
    if metrics.get("burial_ranges") != contract["burial_ranges"]:
        errors.append("Burial ranges do not match the configured families and size classes")
    if metrics.get("coarse_size_class_counts") != contract["coarse_class_counts"]:
        errors.append("Coarse size-class counts do not match the configured distribution")
    if contract["coarse_class_counts"] and metrics.get("coarse_size_sampling") != "jittered_log_stratified_per_class":
        errors.append("Coarse size classes must use stratified logarithmic sampling")
    if not metrics.get("size_aware_exclusion") or not metrics.get("boundary_radius_guard"):
        errors.append("Size-aware wheel and boundary guards must remain enabled")
    instancing = report.get("instancing", {})
    if not instancing.get("geometry_nodes") or instancing.get("instances_realized") is not False:
        errors.append("Clasts must remain non-realized Geometry Nodes instances")
    if instancing.get("scatter_object_count") != 3:
        errors.append("Expected one scatter object per clast family")
    expected_sources = sum(contract["variants_per_family"].values())
    if report.get("library", {}).get("source_mesh_count") != expected_sources:
        errors.append("Procedural clast source library is incomplete")
    if not metrics.get("exclusion_zones"):
        errors.append("Wheel-contact exclusion zone is missing")
    if report.get("scope", {}).get("level3_shading_enabled"):
        errors.append("Level 3 shading must remain disabled")
    renders = report.get("renders", {})
    for name in REQUIRED_RENDERS:
        if not renders.get(name):
            errors.append(f"Missing render record {name}")
    return errors
