"""Prepare a reversible wheel-roll asset and exhaustively validate pose gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.camera_config.resolve(), args.source_blend.resolve(), args.source_sha256, args.output_dir.resolve()


def _matrix_values(matrix: Matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _camera(settings: dict) -> bpy.types.Object:
    old = bpy.data.objects.get("DatasetWheelCamera")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.cameras.new("DatasetWheelCamera_data")
    result = bpy.data.objects.new("DatasetWheelCamera", data)
    bpy.context.scene.collection.objects.link(result)
    data.type = "PERSP"
    data.lens = float(settings["focal_length_mm"])
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = float(settings["sensor_width_mm"])
    data.clip_start = float(settings["clip_start_m"])
    data.dof.use_dof = bool(settings["depth_of_field"])
    return result


def _synthetic_surface_anchors(wheel: bpy.types.Object, count: int = 8) -> list[dict]:
    vertices = [vertex.co.copy() for vertex in wheel.data.vertices]
    axial_limit = max(abs(vertex.x) for vertex in vertices) * 0.25
    candidates = [vertex for vertex in vertices if abs(vertex.x) <= axial_limit]
    if not candidates:
        raise RuntimeError(f"No central-tread vertices found for {wheel.name}")
    anchors = []
    for index in range(count):
        angle = math.radians(index * 360.0 / count)
        direction = Vector((0.0, -math.sin(angle), math.cos(angle)))
        selected = max(candidates, key=lambda vertex: vertex.dot(direction))
        normal = Vector((0.0, selected.y, selected.z)).normalized()
        anchors.append({
            "source_angle_degrees": index * 360.0 / count,
            "anchor_local": selected,
            "normal_local": normal,
        })
    return anchors


def _terrain_clearance(scene: bpy.types.Scene, wheels: list, probe_count: int) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rover = bpy.data.objects.get("Rover")

    def belongs_to_rover(obj):
        current = obj
        while current is not None:
            if current == rover:
                return True
            current = current.parent
        return False

    per_wheel = {}
    for wheel in wheels:
        evaluated = wheel.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            vertices = sorted((evaluated.matrix_world @ vertex.co for vertex in mesh.vertices), key=lambda value: value.z)[:probe_count]
        finally:
            evaluated.to_mesh_clear()
        clearances = []
        for vertex in vertices:
            origin = Vector((vertex.x, vertex.y, vertex.z + 0.5))
            remaining = 3.0
            for _ in range(24):
                hit, location, _normal, _index, obj, _matrix = scene.ray_cast(
                    depsgraph, origin, Vector((0.0, 0.0, -1.0)), distance=remaining
                )
                if not hit:
                    break
                if not belongs_to_rover(obj):
                    clearances.append(float(vertex.z - location.z))
                    break
                step = max(0.0005, float(origin.z - location.z) + 0.0005)
                origin.z -= step
                remaining -= step
            else:
                raise RuntimeError(f"Terrain ray exceeded iteration gate below {wheel.name}")
        if not clearances:
            raise RuntimeError(f"No terrain clearance samples for {wheel.name}")
        per_wheel[wheel.name] = {"minimum_m": min(clearances), "samples_m": clearances}
    return per_wheel


def build(config_path: Path, camera_config_path: Path, source_blend: Path, source_sha256: str, output_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.wheel_pose_sampling import (
        apply_shared_roll,
        restore_base_roll,
        select_visible_anomaly_roll,
        stable_wheel_basis,
    )
    from src.microterrain.camera_poses import validate_wheel_camera_pose_config
    from src.wheel_preparation.pose_sampling import equivalent_travel_m, validate_pose_sampling_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    camera_config = json.loads(camera_config_path.read_text(encoding="utf-8"))
    contract = validate_pose_sampling_config(config)
    validate_wheel_camera_pose_config(camera_config)
    if Path(bpy.data.filepath).resolve() != source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    if int(bpy.context.scene.get("microterrain_level", 0)) != int(config["source"]["required_microterrain_level"]):
        raise RuntimeError("Pose sampling must start from the validated Level-3 scene")
    wheels = [bpy.data.objects.get(name) for name in config["wheels"]]
    if any(wheel is None or wheel.type != "MESH" for wheel in wheels):
        raise RuntimeError("The six canonical wheel meshes are not present")
    base_local = {wheel.name: wheel.matrix_local.copy() for wheel in wheels}
    camera_settings = camera_config["camera"]
    camera = _camera(camera_settings)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = map(int, camera_settings["resolution"])
    tolerance = float(config["physical_validation"]["origin_center_tolerance_m"])
    wheel_metadata = {}
    for wheel in wheels:
        local_center = sum((Vector(corner) for corner in wheel.bound_box), Vector()) / 8.0
        if local_center.length > tolerance:
            raise RuntimeError(f"{wheel.name} origin is {local_center.length:.9f} m from its bounding-box center")
        radius = max(math.hypot(vertex.co.y, vertex.co.z) for vertex in wheel.data.vertices)
        outward, forward, up, axle = stable_wheel_basis(wheel, camera_settings)
        effect = (axle.cross(up)).dot(forward)
        if abs(effect) < 0.95:
            raise RuntimeError(f"Cannot establish deterministic positive roll direction for {wheel.name}")
        wheel_metadata[wheel.name] = {
            "origin_center_error_m": float(local_center.length),
            "radius_m": float(radius),
            "positive_roll_effect_sign": 1.0 if effect > 0.0 else -1.0,
            "base_matrix_local": _matrix_values(base_local[wheel.name]),
            "axle_world": list(map(float, axle)),
        }

    clearance_results = []
    maximum_clearance_loss = float(config["physical_validation"]["maximum_clearance_loss_from_base_m"])
    for roll in map(float, config["normal_sampling"]["roll_degrees"]):
        apply_shared_roll(wheels, base_local, roll)
        values = _terrain_clearance(scene, wheels, int(config["physical_validation"]["low_vertex_probe_count"]))
        minimum = min(entry["minimum_m"] for entry in values.values())
        clearance_results.append({"roll_degrees": roll, "minimum_m": minimum, "per_wheel": values})
    restore_base_roll(wheels, base_local)
    baseline_clearance = next(entry["minimum_m"] for entry in clearance_results if entry["roll_degrees"] == 0.0)
    for entry in clearance_results:
        entry["loss_from_base_m"] = max(0.0, baseline_clearance - entry["minimum_m"])
        entry["ok"] = entry["loss_from_base_m"] <= maximum_clearance_loss

    anomaly_results = []
    anomaly_settings = config["anomaly_sampling"]
    for wheel in wheels:
        anchors = _synthetic_surface_anchors(wheel)
        for pose in camera_config["poses"]:
            for anchor_entry in anchors:
                for jitter in map(float, anomaly_settings["target_jitter_degrees"]):
                    selection = select_visible_anomaly_roll(
                        scene,
                        camera,
                        wheel,
                        wheels,
                        base_local,
                        pose,
                        camera_settings,
                        anomaly_settings,
                        [(anchor_entry["anchor_local"], anchor_entry["normal_local"])],
                        wheel_metadata[wheel.name]["positive_roll_effect_sign"],
                        jitter,
                    )
                    final = selection["selected"] if selection["selected"] is not None else selection["attempts"][-1]
                    anomaly_results.append({
                        "wheel": wheel.name,
                        "camera_pose": pose["id"],
                        "source_anchor_angle_degrees": anchor_entry["source_angle_degrees"],
                        "target_jitter_degrees": jitter,
                        "roll_degrees": final["roll_degrees"],
                        "equivalent_travel_m": final["equivalent_travel_m"],
                        "visibility_search_attempt_count": len(selection["attempts"]),
                        "selected_search_offset_degrees": final["search_offset_degrees"] if selection["selected"] is not None else None,
                        "visible_probe_fraction": final["visible_probe_fraction"],
                        "attempts": selection["attempts"],
                        "ok": selection["ok"],
                    })

    visibility_failures = [entry for entry in anomaly_results if not entry["ok"]]
    clearance_failures = [entry for entry in clearance_results if not entry["ok"]]
    config_signature = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report = {
        "schema_version": 1,
        "source_blend": str(source_blend),
        "source_sha256": source_sha256,
        "source_modified": False,
        "blend": str(output_dir / config["output"]["blend_name"]),
        "config": str(config_path),
        "config_sha256": config_signature,
        "contract": contract,
        "wheel_metadata": wheel_metadata,
        "normal_samples": [
            {"roll_degrees": float(value), "equivalent_travel_m_by_wheel": {
                name: equivalent_travel_m(float(value), metadata["radius_m"]) for name, metadata in wheel_metadata.items()
            }} for value in config["normal_sampling"]["roll_degrees"]
        ],
        "physical_validation": {
            "base_minimum_clearance_m": baseline_clearance,
            "maximum_clearance_loss_from_base_m": maximum_clearance_loss,
            "note": "The imported assembled rover already has local terrain intersections; roll may not worsen the base state beyond tolerance.",
            "phases": clearance_results,
            "ok": not clearance_failures,
        },
        "anomaly_validation": {
            "method": "8 real central-tread surface anchors x 4 cameras x 3 target jitters x 6 wheels; deterministic angular search; no rendering",
            "sample_count": len(anomaly_results),
            "failure_count": len(visibility_failures),
            "ok": not visibility_failures,
            "samples": anomaly_results,
        },
        "validation": {"ok": not visibility_failures and not clearance_failures},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "build_pose_sampling.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if visibility_failures or clearance_failures:
        print(json.dumps({
            "normal_phases": len(report["normal_samples"]),
            "anomaly_samples": len(anomaly_results),
            "visibility_failures": len(visibility_failures),
            "clearance_failures": len(clearance_failures),
        }, indent=2))
        raise RuntimeError(
            f"Pose-sampling hard gate failed: {len(visibility_failures)} visibility, {len(clearance_failures)} clearance"
        )
    for wheel in wheels:
        wheel["pose_sampling_ready"] = True
        wheel["pose_sampling_roll_axis_local"] = "X"
        wheel["pose_sampling_radius_m"] = wheel_metadata[wheel.name]["radius_m"]
        wheel["pose_sampling_base_matrix_local"] = wheel_metadata[wheel.name]["base_matrix_local"]
        wheel["pose_sampling_config_sha256"] = config_signature
    scene["wheel_pose_sampling_ready"] = True
    scene["wheel_pose_sampling_config_sha256"] = config_signature
    scene["wheel_pose_sampling_visibility_policy"] = "fail_closed_anchor_normal_upper_facing_frame_occlusion"
    restore_base_roll(wheels, base_local)
    blend_path = output_dir / config["output"]["blend_name"]
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report["blend"] = str(blend_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "blend": str(blend_path),
        "normal_phases": len(report["normal_samples"]),
        "anomaly_samples": len(anomaly_results),
        "visibility_failures": len(visibility_failures),
        "clearance_failures": len(clearance_failures),
    }, indent=2))
    return report


if __name__ == "__main__":
    build(*_arguments())
