"""Render deterministic L0/L1 comparisons and terrain views at 10/30 cm."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector

from scripts.blender.microterrain.common import look_at, render, rover_objects
from scripts.blender.microterrain.build_microterrain_patch import MACRO_PROXY_OBJECT, PATCH_OBJECT


def _set_level(level: int) -> None:
    original = bpy.data.objects["GaleTerrainVisual"]
    proxy = bpy.data.objects[MACRO_PROXY_OBJECT]
    patch = bpy.data.objects[PATCH_OBJECT]
    original.hide_render = level != 0
    proxy.hide_render = level == 0
    patch.hide_render = level == 0


def render_comparisons(config: dict, output_dir: Path, patch_target: Vector) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    main_camera = bpy.data.objects[config["camera"]["name"]]
    rover_root = bpy.data.objects[config["rover"]["root_object"]]
    rover_members = rover_objects(rover_root)
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    renders = {}

    for level, key, filename in ((0, "L0_macro_only", "render_L0_macro_only.png"), (1, "L1_meso_relief", "render_L1_meso_relief.png")):
        _set_level(level)
        path = output_dir / filename
        render(scene, main_camera, path, tuple(config["camera"]["resolution"]))
        renders[key] = str(path)

    camera_config = config["microterrain"]["validation_camera"]
    camera_data = bpy.data.cameras.new(f"{camera_config['name']}_data")
    camera = bpy.data.objects.new(camera_config["name"], camera_data)
    bpy.context.scene.collection.objects.link(camera)
    camera_data.lens = float(camera_config["focal_length_mm"])
    camera_data.sensor_width = float(camera_config["sensor_width_mm"])
    camera_data.clip_start = 0.001
    azimuth = math.radians(float(camera_config["azimuth_deg"]))
    elevation = math.radians(float(camera_config["elevation_deg"]))
    direction = Vector((math.cos(elevation) * math.cos(azimuth), math.cos(elevation) * math.sin(azimuth), math.sin(elevation)))
    hidden_states = {obj.name: obj.hide_render for obj in rover_members}
    for obj in rover_members:
        obj.hide_render = True
    for distance in map(float, camera_config["distances_m"]):
        camera.location = patch_target + direction * distance
        look_at(camera, patch_target)
        distance_cm = int(round(distance * 100))
        for level in (0, 1):
            _set_level(level)
            key = f"L{level}_terrain_{distance_cm}cm"
            path = output_dir / f"render_{key}.png"
            render(scene, camera, path, tuple(camera_config["resolution"]))
            renders[key] = str(path)
    for obj in rover_members:
        obj.hide_render = hidden_states[obj.name]
    _set_level(1)
    scene.camera = main_camera
    scene.render.resolution_x, scene.render.resolution_y = original_resolution
    return renders
