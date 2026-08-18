"""Reusable Blender-side wheel-roll and hard anomaly-visibility operations."""

from __future__ import annotations

import math

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def bbox_center_world(obj: bpy.types.Object) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def stable_wheel_basis(wheel: bpy.types.Object, camera_settings: dict) -> tuple[Vector, Vector, Vector, Vector]:
    """Return outward, rover-forward, world-up and axle without using rolling Y/Z axes."""
    axle = (wheel.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
    configured_up = Vector(tuple(map(float, camera_settings["world_up"]))).normalized()
    configured_forward = Vector(tuple(map(float, camera_settings["rover_forward_world"]))).normalized()
    up = configured_up - axle * configured_up.dot(axle)
    if up.length < 0.99:
        raise RuntimeError(f"World up is degenerate for {wheel.name}")
    up.normalize()
    forward = configured_forward - axle * configured_forward.dot(axle) - up * configured_forward.dot(up)
    if forward.length < 0.99:
        raise RuntimeError(f"Rover forward is degenerate for {wheel.name}")
    forward.normalize()
    outward = axle if wheel.name.endswith("_left") else -axle
    return outward, forward, up, axle


def camera_pose(camera: bpy.types.Object, wheel: bpy.types.Object, entry: dict, camera_settings: dict) -> dict:
    from scripts.blender.microterrain.common import look_at

    center = bbox_center_world(wheel)
    outward, forward, up, axle = stable_wheel_basis(wheel, camera_settings)
    offset = entry["target_offset_m"]
    target = center + outward * float(offset["outward"]) + forward * float(offset["forward"]) + up * float(offset["up"])
    elevation = math.radians(float(entry["elevation_deg"]))
    tangential = math.radians(float(entry["tangential_deg"]))
    horizontal = outward * math.cos(tangential) + forward * math.sin(tangential)
    direction = (horizontal * math.cos(elevation) + up * math.sin(elevation)).normalized()
    camera.location = target + direction * float(entry["distance_m"])
    look_at(camera, target)
    bpy.context.view_layer.update()
    return {"center": center, "target": target, "outward": outward, "forward": forward, "up": up, "axle": axle}


def apply_shared_roll(wheels: list[bpy.types.Object], base_local: dict[str, Matrix], roll_degrees: float) -> None:
    rotation = Matrix.Rotation(math.radians(float(roll_degrees)), 4, "X")
    for wheel in wheels:
        wheel.matrix_local = base_local[wheel.name] @ rotation
    bpy.context.view_layer.update()


def restore_base_roll(wheels: list[bpy.types.Object], base_local: dict[str, Matrix]) -> None:
    for wheel in wheels:
        wheel.matrix_local = base_local[wheel.name].copy()
    bpy.context.view_layer.update()


def signed_radial_angle_degrees(radial: Vector, up: Vector, forward: Vector) -> float:
    return math.degrees(math.atan2(radial.dot(forward), radial.dot(up)))


def preferred_radial(camera_location: Vector, center: Vector, axle: Vector, up: Vector, upper_bias: float) -> Vector:
    to_camera = camera_location - center
    projected = to_camera - axle * to_camera.dot(axle)
    if projected.length < 1e-8:
        raise RuntimeError("Camera view is parallel to wheel axle and has no radial preference")
    preferred = projected.normalized() + up * float(upper_bias)
    preferred -= axle * preferred.dot(axle)
    preferred.normalize()
    return preferred


def _ray_visibility(scene: bpy.types.Scene, camera_location: Vector, anchor: Vector, carriers: set[bpy.types.Object], epsilon: float) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ray = anchor - camera_location
    distance = ray.length
    direction = ray.normalized()
    hit, location, _normal, _index, hit_object, _matrix = scene.ray_cast(
        depsgraph, camera_location, direction, distance=distance + epsilon
    )
    if not hit:
        return {"ok": True, "hit_object": None, "hit_distance_m": None, "anchor_delta_m": None}
    hit_distance = (location - camera_location).length
    anchor_delta = abs(distance - hit_distance)
    carrier_hit_at_anchor = hit_object in carriers and anchor_delta <= epsilon
    return {
        "ok": bool(carrier_hit_at_anchor),
        "hit_object": hit_object.name if hit_object else None,
        "hit_distance_m": float(hit_distance),
        "anchor_delta_m": float(anchor_delta),
    }


def evaluate_anomaly_probe(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    wheel: bpy.types.Object,
    anchor_local: Vector,
    normal_local: Vector,
    basis: dict,
    settings: dict,
    carrier_objects: list[bpy.types.Object] | None = None,
) -> dict:
    anchor = wheel.matrix_world @ anchor_local
    normal = (wheel.matrix_world.to_3x3() @ normal_local).normalized()
    center, axle, up = basis["center"], basis["axle"], basis["up"]
    radial = anchor - center
    radial -= axle * radial.dot(axle)
    radial.normalize()
    upper_dot = float(radial.dot(up))
    upper_min = math.cos(math.radians(float(settings["max_upper_angle_degrees"])))
    to_camera = (camera.location - anchor).normalized()
    facing_dot = float(normal.dot(to_camera))
    projected = world_to_camera_view(scene, camera, anchor)
    margin = float(settings["frame_margin_fraction"])
    frame_ok = margin <= projected.x <= 1.0 - margin and margin <= projected.y <= 1.0 - margin and projected.z > 0.0
    carriers = set(carrier_objects or [wheel])
    carriers.add(wheel)
    occlusion = _ray_visibility(scene, camera.location.copy(), anchor, carriers, float(settings["occlusion_epsilon_m"]))
    gates = {
        "upper_sector": upper_dot >= upper_min,
        "camera_facing": facing_dot >= float(settings["min_camera_facing_dot"]),
        "frame_margin": bool(frame_ok),
        "non_carrier_occlusion": bool(occlusion["ok"]),
    }
    return {
        "ok": all(gates.values()),
        "gates": gates,
        "upper_dot": upper_dot,
        "upper_min_dot": upper_min,
        "camera_facing_dot": facing_dot,
        "normalized_image": [float(projected.x), float(projected.y), float(projected.z)],
        "occlusion": occlusion,
        "anchor_world_m": list(map(float, anchor)),
        "normal_world": list(map(float, normal)),
    }


def require_anomaly_metadata(anchor_local, normal_local) -> tuple[Vector, Vector]:
    """Fail closed: anomalous samples cannot fall back to unconstrained roll phases."""
    if anchor_local is None or normal_local is None:
        raise ValueError("Anomaly-aware pose sampling requires local anchor and outward normal")
    anchor = Vector(tuple(map(float, anchor_local)))
    normal = Vector(tuple(map(float, normal_local)))
    if len(anchor) != 3 or len(normal) != 3 or normal.length < 0.9:
        raise ValueError("Anomaly anchor/normal must be valid 3-vectors")
    normal.normalize()
    return anchor, normal


def select_visible_anomaly_roll(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    target_wheel: bpy.types.Object,
    wheels: list[bpy.types.Object],
    base_local: dict[str, Matrix],
    pose_entry: dict,
    camera_settings: dict,
    anomaly_settings: dict,
    probes_local: list[tuple[object, object]],
    positive_roll_effect_sign: float,
    target_jitter_degrees: float = 0.0,
    carrier_objects: list[bpy.types.Object] | None = None,
    camera_basis: dict | None = None,
) -> dict:
    """Select the first deterministic roll for which all configured probe gates pass.

    The anomaly injector must provide one or more ``(anchor_local, outward_normal_local)``
    pairs. Missing metadata fails closed; this function never falls back to an
    unconstrained random roll.
    """
    from src.wheel_preparation.pose_sampling import conditioned_roll_degrees, equivalent_travel_m

    if not probes_local:
        raise ValueError("At least one anomaly visibility probe is required")
    carriers = list(carrier_objects or [target_wheel])
    for carrier in carriers:
        current = carrier
        while current is not None and current != target_wheel:
            current = current.parent
        if current != target_wheel:
            raise ValueError(f"Anomaly carrier {carrier.name} must be the target wheel or its child")
    probes = [require_anomaly_metadata(anchor, normal) for anchor, normal in probes_local]
    # Batch rendering resolves deterministic camera jitter before conditioning
    # the wheel roll.  Re-running camera_pose() here would silently erase that
    # jitter and break the pair lock, so callers may provide the already
    # resolved stable wheel/camera basis.  Legacy build-time audits keep the
    # original behavior by omitting camera_basis.
    basis = camera_basis or camera_pose(camera, target_wheel, pose_entry, camera_settings)
    preferred = preferred_radial(
        camera.location,
        basis["center"],
        basis["axle"],
        basis["up"],
        anomaly_settings["upper_bias"],
    )
    desired_angle = signed_radial_angle_degrees(preferred, basis["up"], basis["forward"])
    primary_radial = probes[0][0].copy()
    primary_radial.x = 0.0
    world_radial = (target_wheel.matrix_world.to_3x3() @ primary_radial).normalized()
    current_angle = signed_radial_angle_degrees(world_radial, basis["up"], basis["forward"])
    attempts = []
    selected = None
    required_fraction = float(anomaly_settings["required_visible_probe_fraction"])
    for search_offset in map(float, anomaly_settings["visibility_search_offsets_degrees"]):
        roll = conditioned_roll_degrees(
            current_angle,
            desired_angle + float(target_jitter_degrees) + search_offset,
            positive_roll_effect_sign,
        )
        apply_shared_roll(wheels, base_local, roll)
        basis = camera_basis or camera_pose(camera, target_wheel, pose_entry, camera_settings)
        results = [
            evaluate_anomaly_probe(scene, camera, target_wheel, anchor, normal, basis, anomaly_settings, carriers)
            for anchor, normal in probes
        ]
        visible_fraction = sum(result["ok"] for result in results) / len(results)
        attempt = {
            "search_offset_degrees": search_offset,
            "roll_degrees": roll,
            "equivalent_travel_m": equivalent_travel_m(
                roll, float(target_wheel.get("pose_sampling_radius_m", max(math.hypot(v.co.y, v.co.z) for v in target_wheel.data.vertices)))
            ),
            "visible_probe_fraction": visible_fraction,
            "required_visible_probe_fraction": required_fraction,
            "probe_results": results,
            "ok": visible_fraction >= required_fraction,
        }
        attempts.append(attempt)
        restore_base_roll(wheels, base_local)
        if attempt["ok"]:
            selected = attempt
            break
    return {
        "ok": selected is not None,
        "selected": selected,
        "attempts": attempts,
        "fail_closed": True,
    }
