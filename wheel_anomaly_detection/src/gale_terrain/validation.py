"""Host-side validation for geospatial crop reports."""

from __future__ import annotations

import math

from .core import CropBounds, validate_grid_contract


def validate_crop_report(config: dict, report: dict) -> list[str]:
    errors: list[str] = []
    contract = validate_grid_contract(config)
    expected_bounds: CropBounds = contract["bounds"]
    actual_bounds = report.get("crop", {}).get("projected_bounds_m")
    if actual_bounds is None or any(
        not math.isclose(float(actual), expected, abs_tol=1e-6)
        for actual, expected in zip(actual_bounds or [], (expected_bounds.left, expected_bounds.bottom, expected_bounds.right, expected_bounds.top))
    ):
        errors.append("Crop bounds do not match the configured projected bounds")
    if not report.get("coregistration", {}).get("dtm_irb_crs_equivalent"):
        errors.append("DTM and IRB CRS are not equivalent")
    if not report.get("coregistration", {}).get("exact_shared_bounds"):
        errors.append("DTM and color crops do not share exact projected bounds")
    if not report.get("coregistration", {}).get("integer_resolution_ratio"):
        errors.append("DTM and IRB resolutions are not an integer ratio")
    if report.get("crop", {}).get("dtm_valid_fraction", 0.0) < float(config["crop"]["minimum_valid_fraction"]):
        errors.append("DTM valid fraction is below the configured gate")
    if report.get("crop", {}).get("irb_valid_fraction", 0.0) < float(config["crop"]["minimum_valid_fraction"]):
        errors.append("IRB valid fraction is below the configured gate")
    if report.get("crop", {}).get("vertical_exaggeration") != 1.0:
        errors.append("Vertical exaggeration must equal 1.0")
    for role in ("dtm", "irb_ortho", "irb_label", "mrgb", "mrgb_label"):
        item = report.get("sources", {}).get(role)
        if not item or not item.get("sha256"):
            errors.append(f"Missing source hash for {role}")
    selection = report.get("texture_selection", {})
    if selection.get("chosen") not in {"MRGB", "IRB"}:
        errors.append("Final texture selection is missing or invalid")
    mrgb_passed = bool(report.get("coregistration", {}).get("mrgb", {}).get("passed"))
    if mrgb_passed != (selection.get("chosen") == "MRGB"):
        errors.append("Texture selection is inconsistent with the MRGB coregistration result")
    visualization = report.get("visualization", {})
    if selection.get("chosen") == "IRB" and (
        visualization.get("spatial_source") != "IRB only" or visualization.get("mrgb_spatial_data_used") is not False
    ):
        errors.append("IRB fallback must not use MRGB spatial data")
    return errors
