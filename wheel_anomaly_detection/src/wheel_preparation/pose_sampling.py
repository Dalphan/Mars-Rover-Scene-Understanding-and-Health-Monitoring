"""Pure deterministic contracts for wheel-roll and anomaly-aware pose sampling."""

from __future__ import annotations

import math


EXPECTED_WHEELS = (
    "wheel_front_left",
    "wheel_front_right",
    "wheel_middle_left",
    "wheel_middle_right",
    "wheel_rear_left",
    "wheel_rear_right",
)


def normalize_degrees(angle: float) -> float:
    """Return an angle in the half-open interval [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


def conditioned_roll_degrees(
    anomaly_angle_degrees: float,
    desired_angle_degrees: float,
    positive_roll_effect_sign: float,
) -> float:
    """Roll needed to move a radial anomaly from its current to desired angle."""
    if abs(float(positive_roll_effect_sign)) < 0.5:
        raise ValueError("Positive-roll effect sign must be non-zero")
    return normalize_degrees(
        (float(desired_angle_degrees) - float(anomaly_angle_degrees))
        / (1.0 if positive_roll_effect_sign > 0.0 else -1.0)
    )


def equivalent_travel_m(roll_degrees: float, radius_m: float) -> float:
    if float(radius_m) <= 0.0:
        raise ValueError("Wheel radius must be positive")
    return math.radians(float(roll_degrees)) * float(radius_m)


def validate_pose_sampling_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported wheel-pose sampling schema")
    wheels = tuple(map(str, config.get("wheels", [])))
    if wheels != EXPECTED_WHEELS:
        raise ValueError("Pose sampling requires the six canonical wheels in stable order")
    normal_angles = [float(value) for value in config["normal_sampling"]["roll_degrees"]]
    if len(normal_angles) < 4 or len(normal_angles) != len(set(normalize_degrees(v) for v in normal_angles)):
        raise ValueError("Normal roll phases must contain at least four unique angles")
    anomaly = config["anomaly_sampling"]
    if not bool(anomaly.get("require_anchor")) or not bool(anomaly.get("require_normal")):
        raise ValueError("Anomaly poses must hard-require both a local anchor and local normal")
    jitters = [float(value) for value in anomaly["target_jitter_degrees"]]
    if not jitters or any(abs(value) > 20.0 for value in jitters):
        raise ValueError("Anomaly target jitter must be non-empty and bounded to +/-20 degrees")
    search_offsets = [float(value) for value in anomaly["visibility_search_offsets_degrees"]]
    if not search_offsets or search_offsets[0] != 0.0 or len(search_offsets) != len(set(search_offsets)):
        raise ValueError("Visibility search offsets must be unique, deterministic, and start at zero")
    if any(abs(value) > 60.0 for value in search_offsets):
        raise ValueError("Visibility search offsets must stay within +/-60 degrees")
    upper = float(anomaly["max_upper_angle_degrees"])
    facing = float(anomaly["min_camera_facing_dot"])
    margin = float(anomaly["frame_margin_fraction"])
    visible = float(anomaly["required_visible_probe_fraction"])
    if not 0.0 < upper < 90.0:
        raise ValueError("Anomaly upper-sector angle must lie inside (0, 90) degrees")
    if not 0.0 < facing < 1.0:
        raise ValueError("Camera-facing threshold must lie inside (0, 1)")
    if not 0.0 <= margin < 0.25:
        raise ValueError("Frame margin must lie inside [0, 0.25)")
    if not 0.0 < visible <= 1.0:
        raise ValueError("Required visible-probe fraction must lie inside (0, 1]")
    physical = config["physical_validation"]
    if float(physical["origin_center_tolerance_m"]) <= 0.0:
        raise ValueError("Origin-center tolerance must be positive")
    if int(physical["low_vertex_probe_count"]) < 1:
        raise ValueError("At least one low-vertex clearance probe is required")
    if float(physical["maximum_clearance_loss_from_base_m"]) < 0.0:
        raise ValueError("Maximum clearance loss from the assembled base pose cannot be negative")
    return {
        "wheel_count": len(wheels),
        "normal_phase_count": len(normal_angles),
        "anomaly_jitter_count": len(jitters),
        "hard_visibility_gates": ["upper_sector", "camera_facing", "frame_margin", "non_carrier_occlusion"],
    }
