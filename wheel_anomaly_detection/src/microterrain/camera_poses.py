"""Pure validation helpers for MAHLI-like wheel-camera pose presets."""

from __future__ import annotations

import math


REQUIRED_PILOT_POSES = {
    "A_overhead",
    "C_leading_three_quarter",
    "C_trailing_three_quarter",
    "D_upper_detail",
}


def diagonal_fov_deg(focal_length_mm: float, sensor_width_mm: float, resolution: list[int]) -> float:
    width, height = map(float, resolution)
    sensor_height = sensor_width_mm * height / width
    sensor_diagonal = math.hypot(sensor_width_mm, sensor_height)
    return math.degrees(2.0 * math.atan(sensor_diagonal / (2.0 * focal_length_mm)))


def validate_wheel_camera_pose_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported wheel-camera pose schema")
    camera = config["camera"]
    for key in ("world_up", "rover_forward_world"):
        vector = camera.get(key, [])
        if len(vector) != 3 or math.sqrt(sum(float(value) ** 2 for value in vector)) < 0.99:
            raise ValueError(f"Camera {key} must be a non-zero 3-vector")
    up = [float(value) for value in camera["world_up"]]
    forward = [float(value) for value in camera["rover_forward_world"]]
    cosine = abs(sum(a * b for a, b in zip(up, forward))) / (
        math.sqrt(sum(value * value for value in up)) * math.sqrt(sum(value * value for value in forward))
    )
    if cosine > 0.05:
        raise ValueError("Camera world-up and rover-forward vectors must be nearly orthogonal")
    resolution = list(map(int, camera["resolution"]))
    if len(resolution) != 2 or any(value <= 0 for value in resolution):
        raise ValueError("Camera resolution must contain two positive values")
    focal = float(camera["focal_length_mm"])
    sensor_width = float(camera["sensor_width_mm"])
    if focal <= 0 or sensor_width <= 0:
        raise ValueError("Camera focal length and sensor width must be positive")
    diagonal_fov = diagonal_fov_deg(focal, sensor_width, resolution)
    if not 34.0 <= diagonal_fov <= 39.5:
        raise ValueError("Pilot camera must remain inside the documented MAHLI diagonal-FOV envelope")
    fill = camera.get("pilot_fill_light", {})
    if bool(fill.get("enabled", False)):
        if float(fill.get("energy_w", 0.0)) <= 0.0 or float(fill.get("size_m", 0.0)) <= 0.0:
            raise ValueError("Pilot fill-light energy and size must be positive")
        if float(fill.get("subject_distance_m", 0.0)) <= 0.0:
            raise ValueError("Pilot fill-light subject distance must be positive")
        color = fill.get("color", [])
        if len(color) != 3 or any(not 0.0 <= float(channel) <= 1.0 for channel in color):
            raise ValueError("Pilot fill-light color must contain three normalized channels")
    grade = config.get("pilot", {}).get("terrain_color_grade", {})
    if bool(grade.get("enabled", False)):
        prefixes = grade.get("material_prefixes", [])
        if not prefixes or any(not str(prefix) for prefix in prefixes):
            raise ValueError("Terrain color grade requires material prefixes")
        if not 0.0 <= float(grade.get("hue", -1.0)) <= 1.0:
            raise ValueError("Terrain color-grade hue must lie in [0, 1]")
        if not 0.0 <= float(grade.get("saturation", -1.0)) <= 2.0:
            raise ValueError("Terrain color-grade saturation must lie in [0, 2]")
        if not 0.0 <= float(grade.get("value", -1.0)) <= 2.0:
            raise ValueError("Terrain color-grade value must lie in [0, 2]")
        if not 0.0 <= float(grade.get("factor", -1.0)) <= 1.0:
            raise ValueError("Terrain color-grade factor must lie in [0, 1]")

    poses = config.get("poses", [])
    identifiers = [str(entry["id"]) for entry in poses]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Wheel-camera pose identifiers must be unique")
    missing = sorted(REQUIRED_PILOT_POSES - set(identifiers))
    if missing:
        raise ValueError("Missing pilot poses: " + ", ".join(missing))
    for entry in poses:
        distance = float(entry["distance_m"])
        elevation = float(entry["elevation_deg"])
        tangential = float(entry["tangential_deg"])
        if not 0.021 <= distance <= 2.5:
            raise ValueError(f"Pose {entry['id']} is outside the MAHLI working-distance gate")
        if not -15.0 <= elevation <= 75.0:
            raise ValueError(f"Pose {entry['id']} elevation is invalid")
        if not -60.0 <= tangential <= 60.0:
            raise ValueError(f"Pose {entry['id']} tangential angle is invalid")
        offset = entry.get("target_offset_m", {})
        if set(offset) != {"outward", "forward", "up"}:
            raise ValueError(f"Pose {entry['id']} target offset must use the wheel-local basis")
        if any(abs(float(value)) > 0.25 for value in offset.values()):
            raise ValueError(f"Pose {entry['id']} target offset exceeds the wheel envelope")
        if entry.get("crop_policy") not in {"full_wheel", "intentional_detail_crop"}:
            raise ValueError(f"Pose {entry['id']} crop policy is invalid")
    return {
        "pose_count": len(poses),
        "resolution": resolution,
        "diagonal_fov_deg": diagonal_fov,
        "pose_ids": identifiers,
    }
