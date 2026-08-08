from __future__ import annotations

import argparse
import bpy
import json
import math
import sys
from pathlib import Path

from mathutils import Vector


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Apply the approved clear Martian day lighting preset.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(values)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.matrix_world.translation).to_track_quat("-Z", "Y").to_euler()


def configure_scene(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    renders_dir = output_dir / "renders"
    reports_dir = output_dir / "reports"
    renders_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    scenes = [bpy.data.scenes.get(name) for name in ("Simulator", "Normal", "Perforation")]
    if any(scene is None for scene in scenes):
        raise RuntimeError("Simulator, Normal, and Perforation scenes are required")
    simulator, normal, perforation = scenes
    wheel_camera = bpy.data.objects.get("WheelCamera")
    chase = bpy.data.objects.get("SimulatorCamera")
    terrain_material = bpy.data.materials.get("MartianTerrain_Visual")
    pair_rig = bpy.data.collections.get("PAIR_RIG")
    if wheel_camera is None or chase is None or terrain_material is None:
        raise RuntimeError("Required cameras or Martian terrain material are missing")

    if pair_rig is not None:
        for scene in scenes:
            if pair_rig.name in scene.collection.children:
                scene.collection.children.unlink(pair_rig)
    for scene in scenes:
        if wheel_camera.name not in scene.objects:
            scene.collection.objects.link(wheel_camera)

    lighting = bpy.data.collections.get("MARS_CLEAR_DAY_LIGHTING")
    if lighting is None:
        lighting = bpy.data.collections.new("MARS_CLEAR_DAY_LIGHTING")
    for scene in scenes:
        if lighting.name not in scene.collection.children:
            scene.collection.children.link(lighting)
    for obj in list(lighting.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.lights.remove(data)

    old_lighting = bpy.data.collections.get("SIMULATOR_SOLAR_LIGHTING")
    if old_lighting is not None:
        for scene in scenes:
            if old_lighting.name in scene.collection.children:
                scene.collection.children.unlink(old_lighting)

    sun_data = bpy.data.lights.get("MarsClearDaySun_Data")
    if sun_data is None:
        sun_data = bpy.data.lights.new("MarsClearDaySun_Data", "SUN")
    sun_data.type = "SUN"
    sun_data.energy = 380.0
    sun_data.angle = math.radians(0.35)
    sun_data.color = (1.0, 0.97, 0.92)
    sun = bpy.data.objects.new("MarsClearDaySun", sun_data)
    lighting.objects.link(sun)
    elevation = math.radians(40.0)
    azimuth = math.radians(135.0)
    radius = 50.0
    sun.location = Vector((
        radius * math.cos(elevation) * math.cos(azimuth),
        radius * math.cos(elevation) * math.sin(azimuth),
        radius * math.sin(elevation),
    ))
    look_at(sun, Vector((0.0, 0.0, 0.0)))
    sun["role"] = "mars_clear_day_sun"
    sun["irradiance_watts_per_square_meter"] = 380.0
    sun["elevation_degrees"] = 40.0
    sun["azimuth_degrees"] = 135.0

    world = bpy.data.worlds.get("Mars_Clear_Day_World")
    if world is None:
        world = bpy.data.worlds.new("Mars_Clear_Day_World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    background = nodes.new("ShaderNodeBackground")
    output = nodes.new("ShaderNodeOutputWorld")
    background.inputs["Color"].default_value = (0.055, 0.045, 0.038, 1.0)
    background.inputs["Strength"].default_value = 0.035
    links.new(background.outputs["Background"], output.inputs["Surface"])
    for scene in scenes:
        scene.world = world
        configure_scene(scene)

    normal_map = next((node for node in terrain_material.node_tree.nodes if node.type == "NORMAL_MAP"), None)
    roughness_image = terrain_material.node_tree.nodes.get("Roughness")
    principled = next((node for node in terrain_material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if normal_map is None or roughness_image is None or principled is None:
        raise RuntimeError("Terrain material nodes are incomplete")
    normal_map.inputs["Strength"].default_value = 0.30
    roughness_range = terrain_material.node_tree.nodes.get("Roughness_085_095")
    if roughness_range is None:
        roughness_range = terrain_material.node_tree.nodes.new("ShaderNodeMapRange")
        roughness_range.name = "Roughness_085_095"
    roughness_range.inputs["From Min"].default_value = 0.0
    roughness_range.inputs["From Max"].default_value = 1.0
    roughness_range.inputs["To Min"].default_value = 0.85
    roughness_range.inputs["To Max"].default_value = 0.95
    roughness_range.clamp = True
    links_material = terrain_material.node_tree.links
    for link in list(principled.inputs["Roughness"].links):
        links_material.remove(link)
    links_material.new(roughness_image.outputs["Color"], roughness_range.inputs["Value"])
    links_material.new(roughness_range.outputs["Result"], principled.inputs["Roughness"])

    simulator["lighting_preset"] = "mars_clear_day_closeup"
    simulator["solar_energy_watts_per_square_meter"] = 380.0
    simulator["solar_elevation_degrees"] = 40.0
    simulator["solar_azimuth_degrees"] = 135.0
    simulator["solar_angle_degrees"] = 0.35
    simulator["world_strength"] = 0.035
    simulator["normal_map_strength"] = 0.30
    simulator["collision_enabled"] = False

    saved_chase = chase.matrix_world.copy()
    rover_close = renders_dir / "mars_clear_day_rover_close.png"
    render(simulator, chase, rover_close)
    normal_close = renders_dir / "mars_clear_day_normal_wheel.png"
    render(normal, wheel_camera, normal_close)
    perforation_close = renders_dir / "mars_clear_day_perforation_wheel.png"
    render(perforation, wheel_camera, perforation_close)
    chase.matrix_world = saved_chase
    simulator.camera = chase

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "preset": {
            "sun_strength_watts_per_square_meter": 380.0,
            "sun_angle_degrees": 0.35,
            "sun_color": [1.0, 0.97, 0.92],
            "elevation_degrees": 40.0,
            "azimuth_degrees": 135.0,
            "world_strength": 0.035,
            "world_color": [0.055, 0.045, 0.038],
            "view_transform": "AgX",
            "look": "AgX - Medium High Contrast",
            "exposure_ev": 0.0,
        },
        "shared_world": {scene.name: scene.world.name for scene in scenes},
        "shared_sun_collection": {
            scene.name: lighting.name in scene.collection.children for scene in scenes
        },
        "pair_rig_removed": {
            scene.name: pair_rig is None or pair_rig.name not in scene.collection.children
            for scene in scenes
        },
        "wheel_camera_present": {scene.name: wheel_camera.name in scene.objects for scene in scenes},
        "terrain": {"normal_strength": 0.30, "roughness_range": [0.85, 0.95]},
        "collision_enabled": False,
        "renders": [str(rover_close), str(normal_close), str(perforation_close)],
    }
    (reports_dir / "mars_clear_day_lighting.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Applied clear Martian day lighting preset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
