"""Render deterministic Level-1/Level-2 clast comparisons."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Vector

from scripts.blender.microterrain.common import look_at, render, rover_objects
from scripts.blender.microterrain.build_clast_scatter import SCATTER_OBJECTS


def _set_clasts_visible(visible: bool) -> None:
    for name in SCATTER_OBJECTS.values():
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_render = not visible
    bpy.context.view_layer.update()


def _warm_geometry_nodes_render(scene: bpy.types.Scene, camera: bpy.types.Object, resolution: tuple[int, int]) -> None:
    """Populate Eevee's first-frame instance/shadow caches without writing."""
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = map(int, resolution)
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=False)


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


def render_level2_comparisons(config: dict, output_dir: Path, close_target: Vector, patch_center: Vector) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    main_camera = bpy.data.objects[config["camera"]["name"]]
    rover_root = bpy.data.objects[config["rover"]["root_object"]]
    rover_members = rover_objects(rover_root)
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    renders = {}

    # Render the native persisted L2 state first. Eevee 5.2 can otherwise use
    # an invalid first-frame shadow cache after a hide/show transition across
    # thousands of Geometry Nodes instances.
    for visible, key, filename in (
        (True, "L2_meso_clasts", "render_L2_meso_clasts.png"),
        (False, "L1_meso_only", "render_L1_meso_only.png"),
    ):
        _set_clasts_visible(visible)
        if visible:
            _warm_geometry_nodes_render(scene, main_camera, tuple(config["camera"]["resolution"]))
        path = output_dir / filename
        render(scene, main_camera, path, tuple(config["camera"]["resolution"]))
        renders[key] = str(path)

    camera_config = config["microterrain"]["validation_camera"]
    camera = _camera("MicroterrainL2ValidationCamera", camera_config)
    direction = _direction(camera_config)
    hidden_states = {obj.name: obj.hide_render for obj in rover_members}
    for obj in rover_members:
        obj.hide_render = True
    for distance in map(float, camera_config["distances_m"]):
        camera.location = close_target + direction * distance
        look_at(camera, close_target)
        distance_cm = int(round(distance * 100))
        for level, visible in ((1, False), (2, True)):
            _set_clasts_visible(visible)
            key = f"L{level}_terrain_{distance_cm}cm"
            path = output_dir / f"render_{key}.png"
            render(scene, camera, path, tuple(camera_config["resolution"]))
            renders[key] = str(path)

    macro_config = dict(camera_config)
    macro_config["azimuth_deg"] = 35.0
    macro_config["elevation_deg"] = 72.0
    macro_camera = _camera("MicroterrainL2PatchCamera", macro_config)
    macro_camera.location = patch_center + _direction(macro_config) * 3.25
    look_at(macro_camera, patch_center)
    _set_clasts_visible(True)
    macro_path = output_dir / "render_L2_patch_distribution.png"
    render(scene, macro_camera, macro_path, (1200, 1200))
    renders["L2_patch_distribution"] = str(macro_path)

    for obj in rover_members:
        obj.hide_render = hidden_states[obj.name]
    _set_clasts_visible(True)
    scene.camera = main_camera
    scene.render.resolution_x, scene.render.resolution_y = original_resolution
    return renders
