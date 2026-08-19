"""Render non-destructive MAHLI-like wheel-camera pose examples."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.pose_config.resolve(), args.source_blend.resolve(), args.output_dir.resolve()


def _bbox_center_world(obj: bpy.types.Object) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def _wheel_basis(wheel: bpy.types.Object, camera_settings: dict) -> tuple[Vector, Vector, Vector]:
    rotation = wheel.matrix_world.to_3x3()
    axle = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    configured_up = Vector(tuple(map(float, camera_settings["world_up"]))).normalized()
    configured_forward = Vector(tuple(map(float, camera_settings["rover_forward_world"]))).normalized()
    up = (configured_up - axle * configured_up.dot(axle)).normalized()
    forward = configured_forward - axle * configured_forward.dot(axle) - up * configured_forward.dot(up)
    forward.normalize()
    outward = axle if wheel.name.endswith("_left") else -axle
    return outward, forward, up


def _projected_bounds(scene: bpy.types.Scene, camera: bpy.types.Object, wheel: bpy.types.Object) -> dict:
    evaluated = wheel.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        projected = [
            world_to_camera_view(scene, camera, evaluated.matrix_world @ vertex.co)
            for vertex in mesh.vertices
        ]
    finally:
        evaluated.to_mesh_clear()
    if not projected:
        raise RuntimeError(f"Wheel {wheel.name} has no projected mesh vertices")
    x_values = [float(point.x) for point in projected]
    y_values = [float(point.y) for point in projected]
    minimum_x, maximum_x = min(x_values), max(x_values)
    minimum_y, maximum_y = min(y_values), max(y_values)
    return {
        "normalized_bbox": [minimum_x, minimum_y, maximum_x, maximum_y],
        "projection_source": "evaluated_mesh_vertices",
        "bbox_area_fraction": max(0.0, maximum_x - minimum_x) * max(0.0, maximum_y - minimum_y),
        "fully_inside_frame": minimum_x >= 0.0 and minimum_y >= 0.0 and maximum_x <= 1.0 and maximum_y <= 1.0,
    }


def _apply_terrain_color_grade(settings: dict) -> list[str]:
    if not bool(settings.get("enabled", False)):
        return []
    prefixes = tuple(map(str, settings["material_prefixes"]))
    affected = []
    for material in bpy.data.materials:
        if not material.name.startswith(prefixes) or material.node_tree is None:
            continue
        nodes, links = material.node_tree.nodes, material.node_tree.links
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        base_link = next(
            (link for link in links if link.to_node == principled and link.to_socket.name == "Base Color"),
            None,
        )
        if base_link is None:
            continue
        source_socket = base_link.from_socket
        links.remove(base_link)
        grade = nodes.new("ShaderNodeHueSaturation")
        grade.name = "WheelPosePilot_TerrainGrade"
        grade.label = "Temporary Mars terrain grade"
        grade.inputs["Hue"].default_value = float(settings["hue"])
        grade.inputs["Saturation"].default_value = float(settings["saturation"])
        grade.inputs["Value"].default_value = float(settings["value"])
        grade.inputs["Fac"].default_value = float(settings["factor"])
        links.new(source_socket, grade.inputs["Color"])
        links.new(grade.outputs["Color"], principled.inputs["Base Color"])
        affected.append(material.name)
    if not affected:
        raise RuntimeError("Terrain color grade matched no linked terrain materials")
    return sorted(affected)


def render_pilot(pose_config_path: Path, source_blend: Path, output_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.common import look_at, render
    from src.microterrain.camera_poses import validate_wheel_camera_pose_config

    pose_config = json.loads(pose_config_path.read_text(encoding="utf-8"))
    contract = validate_wheel_camera_pose_config(pose_config)
    if Path(bpy.data.filepath).resolve() != source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    if int(bpy.context.scene.get("microterrain_level", 0)) != 3:
        raise RuntimeError("Wheel-camera pilot must start from the validated Level-3 scene")
    wheel = bpy.data.objects.get(pose_config["pilot"]["target_wheel"])
    if wheel is None or wheel.type != "MESH":
        raise RuntimeError("Pilot target wheel is missing")
    grade_settings = pose_config["pilot"].get("terrain_color_grade", {})
    graded_materials = _apply_terrain_color_grade(grade_settings)

    camera_settings = pose_config["camera"]
    old_camera = bpy.data.objects.get(camera_settings["name"])
    if old_camera:
        bpy.data.objects.remove(old_camera, do_unlink=True)
    camera_data = bpy.data.cameras.new(f"{camera_settings['name']}_data")
    camera = bpy.data.objects.new(camera_settings["name"], camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera_data.type = "PERSP"
    camera_data.lens = float(camera_settings["focal_length_mm"])
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = float(camera_settings["sensor_width_mm"])
    camera_data.clip_start = float(camera_settings["clip_start_m"])
    camera_data.dof.use_dof = bool(camera_settings["depth_of_field"])

    fill_settings = camera_settings.get("pilot_fill_light", {})
    fill_light = None
    if bool(fill_settings.get("enabled", False)):
        old_fill = bpy.data.objects.get("WheelPosePilot_Fill")
        if old_fill:
            bpy.data.objects.remove(old_fill, do_unlink=True)
        fill_data = bpy.data.lights.new("WheelPosePilot_Fill_data", type="AREA")
        fill_data.energy = float(fill_settings["energy_w"])
        fill_data.shape = "DISK"
        fill_data.size = float(fill_settings["size_m"])
        fill_data.color = tuple(map(float, fill_settings["color"]))
        fill_light = bpy.data.objects.new("WheelPosePilot_Fill", fill_data)
        bpy.context.scene.collection.objects.link(fill_light)

    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    center = _bbox_center_world(wheel)
    outward, forward, up = _wheel_basis(wheel, camera_settings)
    results = []
    for entry in pose_config["poses"]:
        offset = entry["target_offset_m"]
        target = center + outward * float(offset["outward"]) + forward * float(offset["forward"]) + up * float(offset["up"])
        elevation = math.radians(float(entry["elevation_deg"]))
        tangential = math.radians(float(entry["tangential_deg"]))
        horizontal = outward * math.cos(tangential) + forward * math.sin(tangential)
        direction = horizontal * math.cos(elevation) + up * math.sin(elevation)
        camera.location = target + direction.normalized() * float(entry["distance_m"])
        look_at(camera, target)
        if fill_light is not None:
            fill_light.location = (
                target
                + direction.normalized() * float(fill_settings["subject_distance_m"])
                + up * float(fill_settings["height_offset_m"])
            )
            look_at(fill_light, target)
        bpy.context.view_layer.update()
        image_path = output_dir / f"{entry['id']}.png"
        render(scene, camera, image_path, tuple(contract["resolution"]))
        projection = _projected_bounds(scene, camera, wheel)
        if entry["crop_policy"] == "full_wheel":
            projection["framing_ok"] = projection["fully_inside_frame"] and 0.25 <= projection["bbox_area_fraction"] <= 0.80
        else:
            projection["framing_ok"] = not projection["fully_inside_frame"] and projection["bbox_area_fraction"] > 1.0
        results.append({
            **entry,
            "image": str(image_path),
            "camera_location_m": list(map(float, camera.location)),
            "target_location_m": list(map(float, target)),
            "projection": projection,
        })

    framing_errors = [entry["id"] for entry in results if not entry["projection"]["framing_ok"]]
    report = {
        "schema_version": 1,
        "source_blend": str(source_blend),
        "source_modified": False,
        "target_wheel": wheel.name,
        "wheel_dimensions_m": list(map(float, wheel.dimensions)),
        "terrain_color_grade": {**grade_settings, "affected_materials": graded_materials},
        "camera": {**camera_settings, "diagonal_fov_deg": contract["diagonal_fov_deg"]},
        "wheel_local_basis": {
            "outward": list(map(float, outward)),
            "forward": list(map(float, forward)),
            "up": list(map(float, up)),
        },
        "poses": results,
        "validation": {"ok": not framing_errors, "framing_errors": framing_errors},
    }
    report_path = output_dir / "camera_pose_pilot.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if framing_errors:
        raise RuntimeError("Wheel-camera framing gate failed: " + ", ".join(framing_errors))
    return report


if __name__ == "__main__":
    render_pilot(*_arguments())
