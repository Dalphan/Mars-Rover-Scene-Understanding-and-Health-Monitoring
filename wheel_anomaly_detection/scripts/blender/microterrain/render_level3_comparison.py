"""Render deterministic Level-2/Level-3 material comparisons."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector

from scripts.blender.microterrain.common import look_at, render, rover_objects


def _set_material_level(level: int, prototype_names: list[str]) -> None:
    if level not in (2, 3):
        raise ValueError("Material ablation level must be 2 or 3")
    key = f"level{level}_material_name"
    patch = bpy.data.objects["MicroterrainPatch_L1"]
    patch.data.materials[0] = bpy.data.materials[patch[key]]
    for name in prototype_names:
        obj = bpy.data.objects[name]
        obj.data.materials[0] = bpy.data.materials[obj[key]]
    bpy.context.view_layer.update()


def _camera(name: str, camera_config: dict) -> bpy.types.Object:
    old = bpy.data.objects.get(name)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.cameras.new(f"{name}_data")
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    data.lens = float(camera_config["focal_length_mm"])
    data.sensor_width = float(camera_config["sensor_width_mm"])
    data.clip_start = 0.001
    return camera


def _direction(camera_config: dict) -> Vector:
    azimuth = math.radians(float(camera_config["azimuth_deg"]))
    elevation = math.radians(float(camera_config["elevation_deg"]))
    return Vector((math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)))


def render_level3_comparisons(config: dict, output_dir: Path, close_target: Vector, prototype_names: list[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    main_camera = bpy.data.objects[config["camera"]["name"]]
    rover_root = bpy.data.objects[config["rover"]["root_object"]]
    rover_members = rover_objects(rover_root)
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    renders = {}

    for level, key, filename in (
        (3, "L3_full_microterrain", "render_L3_full_microterrain.png"),
        (2, "L2_geometry_only", "render_L2_geometry_only.png"),
    ):
        _set_material_level(level, prototype_names)
        path = output_dir / filename
        render(scene, main_camera, path, tuple(config["camera"]["resolution"]))
        renders[key] = str(path)

    camera_config = config["microterrain"]["validation_camera"]
    camera = _camera("MicroterrainL3ValidationCamera", camera_config)
    direction = _direction(camera_config)
    hidden_states = {obj.name: obj.hide_render for obj in rover_members}
    for obj in rover_members:
        obj.hide_render = True
    for distance in map(float, config["microterrain"]["material"]["validation_distances_m"]):
        camera.location = close_target + direction * distance
        look_at(camera, close_target)
        distance_cm = int(round(distance * 100))
        for level in (3, 2):
            _set_material_level(level, prototype_names)
            key = f"L{level}_shading_{distance_cm}cm"
            path = output_dir / f"render_{key}.png"
            render(scene, camera, path, tuple(camera_config["resolution"]))
            renders[key] = str(path)
    for obj in rover_members:
        obj.hide_render = hidden_states[obj.name]
    _set_material_level(3, prototype_names)
    scene.camera = main_camera
    scene.render.resolution_x, scene.render.resolution_y = original_resolution
    return renders
