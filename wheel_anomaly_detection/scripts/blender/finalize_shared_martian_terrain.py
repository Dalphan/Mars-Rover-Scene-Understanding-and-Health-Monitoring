from __future__ import annotations

import argparse
import bpy
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Apply the approved shared terrain and material tuning.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--render",
        choices=("all", "rover", "normal", "perforation"),
        default="all",
        help="Render one diagnostic view, or all of them (default).",
    )
    return parser.parse_args(values)


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
    scene.view_settings.exposure = -1.5
    scene.view_settings.gamma = 1.0


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)


def srgb_hex(hex_color: str) -> tuple[float, float, float, float]:
    """Convert a user-facing sRGB hex colour to Blender scene-linear RGBA."""
    raw = hex_color.removeprefix("#")
    channels = tuple(int(raw[index : index + 2], 16) / 255.0 for index in range(0, 6, 2))
    linear = tuple(
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    )
    return (*linear, 1.0)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    renders_dir = output_dir / "renders"
    reports_dir = output_dir / "reports"
    renders_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    simulator = bpy.data.scenes.get("Simulator")
    normal = bpy.data.scenes.get("Normal")
    perforation = bpy.data.scenes.get("Perforation")
    terrain = bpy.data.objects.get("MartianTerrain")
    sun = bpy.data.objects.get("MarsClearDaySun")
    world = bpy.data.worlds.get("Mars_Clear_Day_World")
    chase = bpy.data.objects.get("SimulatorCamera")
    wheel_camera = bpy.data.objects.get("WheelCamera")
    material = bpy.data.materials.get("MartianTerrain_Visual")
    lighting = bpy.data.collections.get("MARS_CLEAR_DAY_LIGHTING")
    required = (simulator, normal, perforation, terrain, sun, world, chase, wheel_camera, material, lighting)
    if any(value is None for value in required):
        raise RuntimeError("Shared Martian terrain setup is incomplete")
    scenes = [simulator, normal, perforation]

    for scene in scenes:
        if terrain.name not in scene.objects:
            scene.collection.objects.link(terrain)
        if wheel_camera.name not in scene.objects:
            scene.collection.objects.link(wheel_camera)
        if lighting.name not in scene.collection.children:
            scene.collection.children.link(lighting)
        scene.world = world
        configure_scene(scene)

    sun.data.energy = 60.0
    sun.data.angle = math.radians(0.35)
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("Shared Mars world has no Background node")
    background.inputs["Strength"].default_value = 0.01

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    diffuse = nodes.get("Diffuse")
    normal_map = next((node for node in nodes if node.type == "NORMAL_MAP"), None)
    roughness_range = nodes.get("Roughness_085_095")
    if any(node is None for node in (principled, diffuse, normal_map, roughness_range)):
        raise RuntimeError("Terrain material tuning nodes are incomplete")
    normal_map.inputs["Strength"].default_value = 0.25
    roughness_range.inputs["To Min"].default_value = 0.90
    roughness_range.inputs["To Max"].default_value = 0.95
    principled.inputs["Emission Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    principled.inputs["Emission Strength"].default_value = 0.0

    hue_saturation = nodes.get("Terrain_Hue_Saturation")
    if hue_saturation is None:
        hue_saturation = nodes.new("ShaderNodeHueSaturation")
        hue_saturation.name = "Terrain_Hue_Saturation"
    hue_saturation.inputs["Fac"].default_value = 1.0
    hue_saturation.inputs["Hue"].default_value = 0.52
    hue_saturation.inputs["Saturation"].default_value = 1.55
    hue_saturation.inputs["Value"].default_value = 0.90

    color_ramp = nodes.get("Terrain_Mars_ColorRamp")
    if color_ramp is None:
        color_ramp = nodes.new("ShaderNodeValToRGB")
        color_ramp.name = "Terrain_Mars_ColorRamp"
    color_ramp.color_ramp.interpolation = "EASE"
    while len(color_ramp.color_ramp.elements) > 1:
        color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[-1])
    shadow = color_ramp.color_ramp.elements[0]
    shadow.position = 0.0
    shadow.color = srgb_hex("#3D1408")
    middle = color_ramp.color_ramp.elements.new(0.50)
    middle.color = srgb_hex("#8F3515")
    highlight = color_ramp.color_ramp.elements.new(1.0)
    highlight.color = srgb_hex("#C7682D")

    for link in list(principled.inputs["Base Color"].links):
        links.remove(link)
    for link in list(hue_saturation.inputs["Color"].links):
        links.remove(link)
    for link in list(color_ramp.inputs["Fac"].links):
        links.remove(link)
    links.new(diffuse.outputs["Color"], hue_saturation.inputs["Color"])
    links.new(hue_saturation.outputs["Color"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])

    simulator["solar_energy_watts_per_square_meter"] = 60.0
    simulator["world_strength"] = 0.01
    simulator["exposure_ev"] = -1.5
    simulator["normal_map_strength"] = 0.25
    simulator["terrain_roughness_min"] = 0.90
    simulator["terrain_roughness_max"] = 0.95
    simulator["terrain_hue"] = 0.52
    simulator["terrain_saturation"] = 1.55
    simulator["terrain_value"] = 0.90
    simulator["terrain_color_ramp"] = "#3D1408,#8F3515,#C7682D"
    simulator["terrain_shared_with_pair_scenes"] = True

    rover_close = renders_dir / "shared_terrain_rover_close.png"
    normal_close = renders_dir / "shared_terrain_normal_wheel.png"
    perforation_close = renders_dir / "shared_terrain_perforation_wheel.png"
    render_jobs = {
        "rover": (simulator, chase, rover_close),
        "normal": (normal, wheel_camera, normal_close),
        "perforation": (perforation, wheel_camera, perforation_close),
    }
    for name, job in render_jobs.items():
        if args.render in ("all", name):
            render(*job)
    simulator.camera = chase

    context = {
        scene.name: {
            "terrain_object": terrain.name if terrain.name in scene.objects else None,
            "terrain_data": terrain.data.name,
            "terrain_material": terrain.data.materials[0].name,
            "world": scene.world.name if scene.world else None,
            "exposure": float(scene.view_settings.exposure),
            "camera": wheel_camera.name if wheel_camera.name in scene.objects else None,
            "sun_collection": lighting.name in scene.collection.children,
        }
        for scene in scenes
    }
    pair_context_identical = context["Normal"] == context["Perforation"]
    if not pair_context_identical:
        raise RuntimeError("Normal and Perforation rendering context differs")

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "lighting": {
            "sun_strength_watts_per_square_meter": 60.0,
            "sun_angle_degrees": 0.35,
            "exposure_ev": -1.5,
            "world_strength": 0.01,
        },
        "terrain_material": {
            "normal_strength": 0.25,
            "roughness_range": [0.90, 0.95],
            "hue": 0.52,
            "saturation": 1.55,
            "value": 0.90,
            "color_ramp": ["#3D1408", "#8F3515", "#C7682D"],
            "emission_strength": 0.0,
        },
        "scene_context": context,
        "pair_context_identical": pair_context_identical,
        "terrain_object_shared": all(terrain.name in scene.objects for scene in scenes),
        "collision_enabled": False,
        "renders": [str(rover_close), str(normal_close), str(perforation_close)],
    }
    (reports_dir / "shared_martian_terrain.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Finalized shared Martian terrain and material tuning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
