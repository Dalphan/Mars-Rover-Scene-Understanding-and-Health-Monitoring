"""Pure deterministic contract for synthetic Curiosity wheel holes.

This module intentionally has no Blender imports.  It materializes every
semantic choice, irregular contour and placement candidate before Blender is
started so that the renderer only resolves source-geometry and camera gates.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from .perforation import FAMILIES, polygon_quality


SCHEMA_VERSION = 1
SURFACES = ("tread", "shoulder")
SEVERITIES = ("small", "medium", "large")
IMAGE_SECTORS = ("leading", "upper", "trailing")


def _require_weights(name: str, values: Mapping[str, float], expected: Sequence[str]) -> None:
    if tuple(values) != tuple(expected):
        raise ValueError(f"{name} keys must be exactly {list(expected)}")
    if any(float(value) <= 0.0 for value in values.values()):
        raise ValueError(f"{name} weights must be positive")
    if abs(sum(map(float, values.values())) - 1.0) > 1e-9:
        raise ValueError(f"{name} weights must sum to one")


def _positive_range(name: str, values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain two values")
    low, high = map(float, values)
    if not 0.0 < low <= high:
        raise ValueError(f"{name} must be a positive ordered range")
    return low, high


def validate_hole_anomaly_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Unsupported wheel-hole anomaly schema")
    if config.get("class_id") != "hole" or int(config.get("holes_per_sample", 0)) != 1:
        raise ValueError("Hole anomaly v1 supports exactly one hole")

    distributions = config["distributions"]
    _require_weights("severity", distributions["severity"], SEVERITIES)
    _require_weights("profile_family", distributions["profile_family"], FAMILIES)
    _require_weights("image_sector", distributions["image_sector"], IMAGE_SECTORS)
    conditional = distributions["surface_by_severity"]
    if tuple(conditional) != SEVERITIES:
        raise ValueError("surface_by_severity must define small, medium and large")
    for severity in ("small", "medium"):
        _require_weights(f"surface_by_severity.{severity}", conditional[severity], SURFACES)
    if conditional["large"] != {"tread": 1.0}:
        raise ValueError("Large holes are restricted to tread")
    shoulder_marginal = sum(
        float(distributions["severity"][severity]) * float(conditional[severity].get("shoulder", 0.0))
        for severity in SEVERITIES
    )
    if abs(shoulder_marginal - 0.20) > 1e-9:
        raise ValueError("Conditional surface weights must yield a 20 percent shoulder marginal")

    severity_settings = config["severity"]
    if tuple(severity_settings) != SEVERITIES:
        raise ValueError("Severity settings must define small, medium and large")
    for severity, settings in severity_settings.items():
        long_range = _positive_range(f"severity.{severity}.long_axis_m", settings["long_axis_m"])
        short_range = _positive_range(f"severity.{severity}.short_axis_m", settings["short_axis_m"])
        if long_range[0] <= short_range[1]:
            raise ValueError(f"{severity} long axis must remain larger than its short axis")
        _positive_range(f"severity.{severity}.cavity_recess_m", settings["cavity_recess_m"])
        if int(settings["minimum_mask_area_px"]) < 1 or int(settings["minimum_mask_short_side_px"]) < 1:
            raise ValueError(f"{severity} mask gates must be positive")

    quality = config["profile_quality"]
    aspect = _positive_range("profile_quality.aspect_ratio", quality["aspect_ratio"])
    convexity = _positive_range("profile_quality.convexity", quality["convexity"])
    if aspect[0] < 1.0 or convexity[1] > 1.0:
        raise ValueError("Invalid profile quality bounds")
    if not 0.0 < float(quality["maximum_circularity"]) <= 1.0:
        raise ValueError("maximum_circularity must be in (0, 1]")
    if not 0.0 < float(quality["minimum_open_area_fraction"]) <= 1.0:
        raise ValueError("minimum_open_area_fraction must be in (0, 1]")

    placement = config["placement"]
    if int(placement["maximum_candidate_attempts"]) != 32:
        raise ValueError("Hole anomaly v1 requires exactly 32 pre-render placement candidates")
    if int(placement["expected_grouser_count"]) != 19:
        raise ValueError("Curiosity placement requires 19 grousers")
    if float(placement["maximum_upper_angle_degrees"]) != 52.0:
        raise ValueError("The audited upper-sector limit is 52 degrees")
    if float(placement["frame_margin_fraction"]) != 0.04:
        raise ValueError("The audited anomaly frame margin is 4 percent")

    mask_gates = config["mask_gates"]
    if int(mask_gates["maximum_raster_speckle_component_px"]) != 4:
        raise ValueError("Anomaly v1 removes only AOV raster speckles of at most 4 pixels")
    if mask_gates.get("require_single_component") is not True:
        raise ValueError("Anomaly v1 requires one connected mask component after de-speckling")

    photometric = config["photometric_gates"]
    if photometric.get("failure_policy", "fail_closed") not in {"fail_closed", "diagnostic"}:
        raise ValueError("photometric_gates.failure_policy must be fail_closed or diagnostic")
    changed_fraction = float(photometric["minimum_changed_fraction_inside_mask"])
    if not 0.0 < changed_fraction <= 1.0:
        raise ValueError("minimum_changed_fraction_inside_mask must be in (0, 1]")
    if int(photometric["changed_pixel_threshold_8bit"]) < 1 or int(photometric["minimum_median_delta_8bit"]) < 1:
        raise ValueError("Photometric 8-bit thresholds must be positive")

    geometry = config["geometry"]
    _positive_range("geometry.rim_width_m", geometry["rim_width_m"])
    if abs(float(geometry["skin_thickness_m"]) - 0.00075) > 1e-12:
        raise ValueError("Curiosity skin thickness must remain 0.75 mm")
    if geometry.get("open_wall_depth_profile") != "T3":
        raise ValueError("Open-wall depth profile must be the approved T3 profile")
    if int(geometry.get("open_wall_fold_arc_count", 0)) != 2:
        raise ValueError("T3 must use exactly two partial fold arcs")
    depth_by_severity = geometry.get("open_wall_depth_max_by_severity_m", {})
    if tuple(depth_by_severity) != SEVERITIES:
        raise ValueError("T3 open-wall depths must be defined for small, medium and large")
    approved_depths = {"small": 0.002, "medium": 0.003, "large": 0.004}
    if any(abs(float(depth_by_severity[name]) - depth) > 1e-12 for name, depth in approved_depths.items()):
        raise ValueError("T3 maximum open-wall depths must be exactly 2, 3 and 4 mm")
    if geometry.get("through_opening") is not True or geometry.get("recessed_cap_enabled") is not False:
        raise ValueError("Hole anomaly must be a true through opening without a recessed cap")
    clear_depth = float(geometry.get("minimum_synthetic_clear_depth_m", 0.0))
    if not 0.10 <= clear_depth <= float(config["wheel_diameter_m"]):
        raise ValueError("Synthetic through-opening clear depth is invalid")
    expected_profiles = ("R0_current", "R1_mild", "R2_balanced", "R3_strong", "R4_minimal", "R4_robust")
    profiles = geometry.get("rim_profiles", {})
    if tuple(profiles) != expected_profiles or geometry.get("active_rim_profile") not in profiles:
        raise ValueError("Rim reduction profiles or active profile are invalid")
    previous = None
    for name in expected_profiles[:-1]:
        profile = profiles[name]
        values = (
            float(profile.get("width_scale", 0.0)),
            float(profile.get("surface_offset_m", -1.0)),
            float(profile.get("exposed_fraction_scale", 0.0)),
        )
        if not 0.0 < values[0] <= 1.0 or not 0.0 <= values[1] <= 0.0002 or not 0.0 < values[2] <= 1.0:
            raise ValueError(f"Rim reduction profile {name} is outside safe bounds")
        if previous is not None and any(value >= prior for value, prior in zip(values, previous, strict=True)):
            raise ValueError("Rim reduction profiles must decrease monotonically")
        previous = values
    robust = profiles["R4_robust"]
    robust_width = tuple(map(float, robust.get("effective_width_m", [])))
    if robust_width != (0.00045, 0.00085):
        raise ValueError("R4_robust effective rim width must be exactly 0.45 to 0.85 mm")
    if float(robust.get("surface_offset_m", -1.0)) != 0.00006:
        raise ValueError("R4_robust surface offset must be exactly 0.06 mm")
    if float(robust.get("exposed_fraction_scale", 0.0)) != 0.25:
        raise ValueError("R4_robust must preserve R4 metal exposure")
    if profiles["R0_current"] != {
        "width_scale": 1.0,
        "surface_offset_m": 0.0002,
        "exposed_fraction_scale": 1.0,
    }:
        raise ValueError("R0_current must preserve the approved reference rim")
    flap = geometry["flap"]
    if abs(float(flap["eligible_probability"]) - 0.20) > 1e-12:
        raise ValueError("Flap probability must remain 20 percent for eligible holes")
    if float(flap["maximum_angle_degrees"]) > 20.0 or float(flap["maximum_lift_m"]) > 0.005:
        raise ValueError("Flap geometry exceeds the approved subtle limits")

    return {
        "severity_ids": list(SEVERITIES),
        "surface_ids": list(SURFACES),
        "profile_ids": list(FAMILIES),
        "image_sector_ids": list(IMAGE_SECTORS),
        "maximum_candidate_attempts": int(placement["maximum_candidate_attempts"]),
    }


def derive_seed(master_seed: int, sample_index: int, namespace: str) -> int:
    payload = f"hole:{int(master_seed)}:{int(sample_index)}:{namespace}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    needle = rng.random()
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += float(weight)
        if needle <= cumulative:
            return str(key)
    return str(next(reversed(weights)))


def _profile_points(config: dict, family: str, severity: str, seed: int) -> tuple[list[list[float]], dict[str, Any]]:
    settings = config["severity"][severity]
    quality_settings = config["profile_quality"]
    count = {"jagged_slit": 16, "branched_tear": 20, "peeled_window": 18}[family]
    for attempt in range(64):
        rng = random.Random(int(seed) + attempt * 7919)
        long_size = rng.uniform(*map(float, settings["long_axis_m"]))
        short_size = rng.uniform(*map(float, settings["short_axis_m"]))
        points: list[list[float]] = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            angle += rng.uniform(-0.10, 0.10) * (2.0 * math.pi / count)
            if family == "jagged_slit":
                factor = rng.uniform(0.72, 1.20)
                if index in {2, 7, 12}:
                    factor *= rng.uniform(1.16, 1.32)
            elif family == "branched_tear":
                factor = rng.uniform(0.68, 1.18)
                factor *= 1.0 + 0.72 * math.exp(-((angle - 0.48 * math.pi) / 0.28) ** 2)
                if index in {4, 5, 6}:
                    factor *= rng.uniform(0.72, 0.88)
            else:
                factor = rng.uniform(0.70, 1.16)
                factor *= 1.0 + 0.85 * math.exp(-((angle - 1.15 * math.pi) / 0.22) ** 2)
            points.append([
                round(0.5 * long_size * factor * math.cos(angle), 8),
                round(0.5 * short_size * factor * math.sin(angle), 8),
            ])
        measured_width = max(point[0] for point in points) - min(point[0] for point in points)
        measured_height = max(point[1] for point in points) - min(point[1] for point in points)
        if measured_width <= 1e-12 or measured_height <= 1e-12:
            continue
        scale_long = long_size / measured_width
        scale_short = short_size / measured_height
        points = [
            [round(point[0] * scale_long, 8), round(point[1] * scale_short, 8)]
            for point in points
        ]
        quality = polygon_quality(points)
        if (
            not bool(quality["self_intersects"])
            and float(quality_settings["aspect_ratio"][0]) <= float(quality["aspect_ratio"]) <= float(quality_settings["aspect_ratio"][1])
            and float(quality_settings["convexity"][0]) <= float(quality["convexity"]) <= float(quality_settings["convexity"][1])
            and float(quality["circularity"]) <= float(quality_settings["maximum_circularity"])
        ):
            return points, {key: bool(value) if isinstance(value, bool) else float(value) for key, value in quality.items()}
    raise ValueError(f"Could not generate valid hole profile for {family}/{severity}")


def _placement_candidates(config: dict, surface: str, seed: int) -> list[dict[str, float | int]]:
    rng = random.Random(int(seed))
    count = int(config["placement"]["maximum_candidate_attempts"])
    grousers = int(config["placement"]["expected_grouser_count"])
    axial_bins = [((2.0 * (index + 0.5) / count) - 1.0) * 0.98 for index in range(count)]
    rng.shuffle(axial_bins)
    result = []
    for attempt in range(count):
        if surface == "tread":
            result.append({
                "attempt": attempt,
                "panel_index": int(rng.randrange(grousers)),
                "panel_jitter_fraction": round(rng.uniform(-0.24, 0.24), 8),
                "axial_fraction": round(axial_bins[attempt], 8),
            })
        else:
            shoulder_band = list(map(float, config["placement"]["shoulder_outboard_center_fraction"]))
            result.append({
                "attempt": attempt,
                "source_angle_degrees": round(rng.uniform(-180.0, 180.0), 8),
                "outboard_center_fraction": round(rng.uniform(*shoulder_band), 8),
            })
    return result


def sample_hole_descriptor(
    config: dict,
    *,
    master_seed: int,
    sample_index: int,
    surface_wear: str,
    severity: str | None = None,
    surface: str | None = None,
    image_sector: str | None = None,
    profile_family: str | None = None,
    flap_enabled: bool | None = None,
) -> dict[str, Any]:
    """Materialize a deterministic one-hole descriptor for a pair."""

    validate_hole_anomaly_config(config)
    selection_seed = derive_seed(master_seed, sample_index, "selection")
    profile_seed = derive_seed(master_seed, sample_index, "profile")
    placement_seed = derive_seed(master_seed, sample_index, "placement")
    material_seed = derive_seed(master_seed, sample_index, "material")
    flap_seed = derive_seed(master_seed, sample_index, "flap")
    rng = random.Random(selection_seed)
    distributions = config["distributions"]
    severity = severity or _weighted_choice(rng, distributions["severity"])
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity override: {severity}")
    surface = surface or _weighted_choice(rng, distributions["surface_by_severity"][severity])
    if surface not in SURFACES or (severity == "large" and surface != "tread"):
        raise ValueError("Invalid surface/severity combination")
    image_sector = image_sector or _weighted_choice(rng, distributions["image_sector"])
    if image_sector not in IMAGE_SECTORS:
        raise ValueError(f"Unknown image sector override: {image_sector}")
    family = profile_family or _weighted_choice(rng, distributions["profile_family"])
    if family not in FAMILIES:
        raise ValueError(f"Unknown profile family override: {family}")

    if surface == "tread":
        orientation = "axial" if severity == "large" or rng.random() < 0.60 else "circumferential"
    else:
        # Curiosity's side face is highly perforated in this asset.  The v1
        # shoulder is therefore the continuous outboard band of the
        # cylindrical skin, where a hole footprint can remain fully on metal.
        orientation = "tangential"
    points, quality = _profile_points(config, family, severity, profile_seed)

    eligible_flap = severity in {"medium", "large"}
    if flap_enabled is None:
        flap_enabled = eligible_flap and random.Random(flap_seed).random() < float(config["geometry"]["flap"]["eligible_probability"])
    if flap_enabled and not eligible_flap:
        raise ValueError("Small holes cannot have a flap")
    flap_rng = random.Random(flap_seed)
    flap = {
        "enabled": bool(flap_enabled),
        "edge_index": int(flap_rng.randrange(len(points))) if flap_enabled else None,
        "angle_degrees": round(flap_rng.uniform(8.0, float(config["geometry"]["flap"]["maximum_angle_degrees"])), 8) if flap_enabled else 0.0,
        "lift_m": round(flap_rng.uniform(0.0015, float(config["geometry"]["flap"]["maximum_lift_m"])), 8) if flap_enabled else 0.0,
    }

    geometry_rng = random.Random(derive_seed(master_seed, sample_index, "geometry"))
    material_rng = random.Random(material_seed)
    exposed_ranges = config["material"]["rim_exposed_fraction_by_wear"]
    if surface_wear not in exposed_ranges:
        raise ValueError(f"Unknown surface-wear state for hole material: {surface_wear}")
    sector_jitter_unit = random.Random(derive_seed(master_seed, sample_index, "sector")).uniform(-1.0, 1.0)
    return {
        "class_id": "hole",
        "count": 1,
        "severity": severity,
        "surface": surface,
        "image_sector": image_sector,
        "orientation": orientation,
        "profile_family": family,
        "profile_seed": int(profile_seed),
        "points_long_short_m": points,
        "quality": quality,
        "minimum_open_area_fraction": float(config["profile_quality"]["minimum_open_area_fraction"]),
        "placement_seed": int(placement_seed),
        "placement_candidates": _placement_candidates(config, surface, placement_seed),
        "sector_jitter_unit": round(float(sector_jitter_unit), 8),
        "rim_width_m": round(geometry_rng.uniform(*map(float, config["geometry"]["rim_width_m"])), 8),
        "skin_thickness_m": float(config["geometry"]["skin_thickness_m"]),
        "cavity_recess_m": round(geometry_rng.uniform(*map(float, config["severity"][severity]["cavity_recess_m"])), 8),
        "flap": flap,
        "material": {
            "cavity_luminance_factor": round(material_rng.uniform(*map(float, config["material"]["cavity_luminance_factor"])), 8),
            "cavity_roughness": round(material_rng.uniform(*map(float, config["material"]["cavity_roughness"])), 8),
            "rim_exposed_fraction": round(material_rng.uniform(*map(float, exposed_ranges[surface_wear])), 8),
            "emission_strength": 0.0,
        },
        "seeds": {
            "selection": int(selection_seed),
            "profile": int(profile_seed),
            "placement": int(placement_seed),
            "material": int(material_seed),
            "flap": int(flap_seed),
        },
    }


def validate_hole_descriptor(config: dict, descriptor: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if descriptor.get("class_id") != "hole" or int(descriptor.get("count", 0)) != 1:
        errors.append("Descriptor must contain exactly one hole")
    severity = descriptor.get("severity")
    surface = descriptor.get("surface")
    if severity not in SEVERITIES:
        errors.append("Unknown severity")
    if surface not in SURFACES or (severity == "large" and surface != "tread"):
        errors.append("Invalid surface/severity combination")
    if descriptor.get("image_sector") not in IMAGE_SECTORS:
        errors.append("Unknown image sector")
    if descriptor.get("profile_family") not in FAMILIES:
        errors.append("Unknown profile family")
    points = descriptor.get("points_long_short_m")
    if not isinstance(points, Sequence) or len(points) < 8:
        errors.append("Hole contour must contain at least eight points")
        return errors
    quality = polygon_quality(points)
    settings = config["profile_quality"]
    if bool(quality["self_intersects"]):
        errors.append("Hole contour self-intersects")
    if not float(settings["aspect_ratio"][0]) <= float(quality["aspect_ratio"]) <= float(settings["aspect_ratio"][1]):
        errors.append("Hole aspect ratio outside configured range")
    if not float(settings["convexity"][0]) <= float(quality["convexity"]) <= float(settings["convexity"][1]):
        errors.append("Hole convexity outside configured range")
    if float(quality["circularity"]) > float(settings["maximum_circularity"]):
        errors.append("Hole contour is too circular")
    candidates = descriptor.get("placement_candidates", [])
    if len(candidates) != int(config["placement"]["maximum_candidate_attempts"]):
        errors.append("Placement candidate count does not match the contract")
    flap = descriptor.get("flap", {})
    if bool(flap.get("enabled")) and severity == "small":
        errors.append("Small hole unexpectedly has a flap")
    return errors
