"""Pure Level-3 material contract and deterministic signature."""

from __future__ import annotations

import hashlib
import json

from .clasts import validate_level2_config


def validate_level3_config(config: dict) -> dict:
    level2 = validate_level2_config(config)
    micro = config["microterrain"]
    material = micro.get("material", {})
    if int(micro.get("level", 0)) < 3 or not material.get("enabled"):
        raise ValueError("Microterrain Level 3 material must be enabled")
    for key in ("dust_amount", "coarse_albedo_strength", "medium_albedo_strength", "roughness_variation", "micro_bump_strength", "clast_dust_fraction", "clast_bump_strength", "clast_roughness_variation"):
        if not 0 <= float(material[key]) <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    if not 0 <= float(material["dust_roughness"]) <= 1:
        raise ValueError("Dust roughness must lie in [0, 1]")
    wavelengths = (
        "coarse_albedo_wavelength_m",
        "medium_albedo_wavelength_m",
        "fine_roughness_wavelength_m",
        "very_fine_roughness_wavelength_m",
        "micro_bump_wavelength_m",
        "micro_bump_secondary_wavelength_m",
        "clast_bump_wavelength_m",
    )
    if any(float(material[key]) <= 0 for key in wavelengths):
        raise ValueError("All Level-3 material wavelengths must be positive")
    if float(material["micro_bump_wavelength_m"]) > 0.0005 or float(material["micro_bump_secondary_wavelength_m"]) >= float(material["micro_bump_wavelength_m"]):
        raise ValueError("Terrain bump must remain sub-0.5 mm and multiscale")
    if float(material["micro_bump_distance_m"]) <= 0 or float(material["micro_bump_distance_m"]) > 0.0002:
        raise ValueError("Terrain bump distance exceeds the Level-3 gate")
    if float(material["clast_bump_distance_m"]) <= 0 or float(material["clast_bump_distance_m"]) > 0.0002:
        raise ValueError("Clast bump distance exceeds the Level-3 gate")
    distances = list(map(float, material["validation_distances_m"]))
    if not distances or any(distance <= 0 for distance in distances):
        raise ValueError("Level-3 validation distances must be positive")
    payload = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hashlib.sha256(payload).hexdigest()
    return {**level2, "material_signature_sha256": signature, "validation_distances_m": distances}
