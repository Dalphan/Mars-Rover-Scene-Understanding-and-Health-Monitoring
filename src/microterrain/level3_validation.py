"""Host-side validation for the Level-3 material checkpoint."""

from __future__ import annotations

from .material import validate_level3_config


def required_level3_renders(config: dict) -> tuple[str, ...]:
    distances = validate_level3_config(config)["validation_distances_m"]
    names = ["L2_geometry_only", "L3_full_microterrain"]
    for distance in distances:
        distance_cm = int(round(distance * 100))
        names.extend((f"L2_shading_{distance_cm}cm", f"L3_shading_{distance_cm}cm"))
    return tuple(names)


def validate_level3_report(config: dict, report: dict) -> list[str]:
    contract = validate_level3_config(config)
    errors: list[str] = []
    if report.get("level") != 3:
        errors.append("Report is not Level 3")
    material = report.get("material", {})
    if material.get("signature_sha256") != contract["material_signature_sha256"]:
        errors.append("Level-3 material signature mismatch")
    if material.get("terrain_material") != "GaleTerrainMicroterrain_L3":
        errors.append("Canonical Level-3 terrain material is missing")
    if material.get("terrain_noise_layers") != 6:
        errors.append("Terrain material is not sufficiently multiscale")
    if material.get("clast_material_count") != 4:
        errors.append("Expected four Level-3 clast material variants")
    if report.get("geometry", {}).get("level2_modified") is not False:
        errors.append("Level-2 geometry must remain unchanged")
    if report.get("geometry", {}).get("scatter_signature_sha256") != report.get("level2", {}).get("scatter_signature_sha256"):
        errors.append("Level-2 scatter signature changed")
    if not report.get("ablation", {}).get("level2_materials_preserved"):
        errors.append("Level-2 materials are not preserved for ablation")
    if report.get("scope", {}).get("multi_distance_matrix_generated"):
        errors.append("Final multi-distance matrix must remain deferred")
    renders = report.get("renders", {})
    for name in required_level3_renders(config):
        if not renders.get(name):
            errors.append(f"Missing render record {name}")
    return errors
