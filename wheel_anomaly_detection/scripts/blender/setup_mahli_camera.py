"""Create or update the approximate, scriptable MAHLI camera preset."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector


def object_bbox_center_world(obj: bpy.types.Object) -> Vector:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(corners, Vector()) / len(corners)


def setup_mahli_camera(config: dict) -> tuple[bpy.types.Object, Vector]:
    camera_config = config["camera"]
    target = bpy.data.objects.get(camera_config["target_wheel"])
    if target is None or target.type != "MESH":
        raise RuntimeError(f"Missing target wheel {camera_config['target_wheel']}")
    target_point = object_bbox_center_world(target) + Vector(camera_config.get("target_offset_local_m", (0, 0, 0)))

    camera = bpy.data.objects.get(camera_config["name"])
    if camera is None:
        data = bpy.data.cameras.new(f"{camera_config['name']}_data")
        camera = bpy.data.objects.new(camera_config["name"], data)
        bpy.context.scene.collection.objects.link(camera)
    elif camera.type != "CAMERA":
        raise RuntimeError(f"{camera.name} exists but is not a camera")

    azimuth = math.radians(float(camera_config["azimuth_deg"]))
    elevation = math.radians(float(camera_config["elevation_deg"]))
    distance = float(camera_config["distance_m"])
    horizontal = distance * math.cos(elevation)
    offset = Vector((horizontal * math.cos(azimuth), horizontal * math.sin(azimuth), distance * math.sin(elevation)))
    camera.location = target_point + offset
    camera.rotation_euler = (target_point - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "PERSP"
    camera.data.lens = float(camera_config["focal_length_mm"])
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = float(camera_config["sensor_width_mm"])
    camera.data.dof.use_dof = False
    camera["camera_model"] = "MAHLI_approximate"
    camera["target_wheel"] = target.name
    camera["distance_m"] = distance
    camera["azimuth_deg"] = float(camera_config["azimuth_deg"])
    camera["elevation_deg"] = float(camera_config["elevation_deg"])

    scene = bpy.context.scene
    scene.camera = camera
    scene.render.resolution_x = int(camera_config["resolution"][0])
    scene.render.resolution_y = int(camera_config["resolution"][1])
    scene.render.resolution_percentage = int(config["render"]["resolution_percentage"])
    return camera, target_point


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not argv:
        raise RuntimeError("Pass the Gale config JSON path after --")
    setup_mahli_camera(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
