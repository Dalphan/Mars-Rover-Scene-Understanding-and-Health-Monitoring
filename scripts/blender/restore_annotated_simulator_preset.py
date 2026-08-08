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
    parser = argparse.ArgumentParser(description="Restore the user-approved annotated simulator preset.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--solar-altitude", type=float, default=70.0)
    return parser.parse_args(values)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    # Sun direction is controlled by rotation. Use the just-assigned location
    # directly so a changed elevation cannot use a stale world matrix.
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


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

    simulator = bpy.data.scenes.get("Simulator")
    terrain = bpy.data.objects.get("MartianTerrain")
    root = bpy.data.objects.get("RoverRoot")
    chase = bpy.data.objects.get("SimulatorCamera")
    wheel_camera = bpy.data.objects.get("WheelCamera")
    material = bpy.data.materials.get("MartianTerrain_Visual")
    if any(value is None for value in (simulator, terrain, root, chase, wheel_camera, material)):
        raise RuntimeError("Simulator, terrain, rover, cameras, or material is missing")

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    diffuse = nodes.get("Diffuse")
    normal_map = next((node for node in nodes if node.type == "NORMAL_MAP"), None)
    roughness_image = nodes.get("Roughness")
    if any(node is None for node in (principled, diffuse, normal_map, roughness_image)):
        raise RuntimeError("Original terrain material nodes are incomplete")
    for link in list(principled.inputs["Base Color"].links):
        links.remove(link)
    for link in list(principled.inputs["Roughness"].links):
        links.remove(link)
    links.new(diffuse.outputs["Color"], principled.inputs["Base Color"])
    links.new(roughness_image.outputs["Color"], principled.inputs["Roughness"])
    normal_map.inputs["Strength"].default_value = 0.30

    lighting = bpy.data.collections.get("SIMULATOR_SOLAR_LIGHTING")
    if lighting is None:
        lighting = bpy.data.collections.new("SIMULATOR_SOLAR_LIGHTING")
    if lighting.name not in simulator.collection.children:
        simulator.collection.children.link(lighting)
    old_lighting = bpy.data.collections.get("MARS_CLEAR_DAY_LIGHTING")
    if old_lighting is not None and old_lighting.name in simulator.collection.children:
        simulator.collection.children.unlink(old_lighting)

    sun = bpy.data.objects.get("SimulatorSolarSun")
    if sun is None:
        sun_data = bpy.data.lights.get("SimulatorSolarSun_Data") or bpy.data.lights.new("SimulatorSolarSun_Data", "SUN")
        sun = bpy.data.objects.new("SimulatorSolarSun", sun_data)
    if sun.name not in lighting.objects:
        lighting.objects.link(sun)
    sun.data.type = "SUN"
    sun.data.energy = 3.5
    sun.data.color = (1.0, 0.88, 0.75)
    sun.data.angle = math.radians(4.5)
    azimuth_xy = Vector((-30.0, -40.0))
    horizontal_distance = azimuth_xy.length
    sun.location = Vector((*azimuth_xy, horizontal_distance * math.tan(math.radians(args.solar_altitude))))
    look_at(sun, Vector((0.0, 0.0, 0.0)))
    sun["role"] = "simulator_solar_light"
    sun["solar_altitude_degrees"] = float(args.solar_altitude)
    sun["solar_angle_degrees"] = 4.5

    world = bpy.data.worlds.get("Simulator_Solar_World")
    if world is None:
        world = bpy.data.worlds.new("Simulator_Solar_World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.035, 0.045, 0.06, 1.0)
    background.inputs["Strength"].default_value = 0.08
    simulator.world = world

    root.location = Vector((-18.0, -4.0, 0.342345397069699))
    root.rotation_euler = (0.0, 0.0, math.radians(45.0))
    for obj in root.children_recursive:
        if obj.get("role") in {"ground_probe", "ground_probe_layout"}:
            obj.hide_render = True
    simulator.render.engine = "CYCLES"
    simulator.cycles.device = "CPU"
    simulator.cycles.samples = 40
    simulator.cycles.use_denoising = True
    simulator.render.resolution_x = 800
    simulator.render.resolution_y = 600
    simulator.render.resolution_percentage = 100
    simulator.render.image_settings.file_format = "PNG"
    simulator.render.film_transparent = False
    simulator.view_settings.exposure = 0.0

    output = renders_dir / "annotated_preset_rover_close.png"
    render(simulator, chase, output)
    simulator.camera = chase
    simulator["lighting_status"] = "annotated_preset_restored"
    simulator["solar_energy"] = 3.5
    simulator["solar_altitude_degrees"] = float(args.solar_altitude)
    simulator["solar_angle_degrees"] = 4.5
    simulator["normal_map_strength"] = 0.30
    simulator["terrain_material_preset"] = "original_diffuse_and_roughness"
    simulator["terrain_shared_with_pair_scenes"] = False
    simulator["collision_enabled"] = False
    simulator["terrain_collision_enabled"] = False
    simulator["simulator_milestone"] = "1-terrain-preview"

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "status": "completed",
        "preset": "user_annotation_1",
        "sun": {"energy": 3.5, "altitude_degrees": args.solar_altitude, "angle_degrees": 4.5},
        "world_strength": 0.08,
        "normal_map_strength": 0.30,
        "terrain_base_color": "original Diffuse image",
        "rover_pose": {"location": list(root.location), "yaw_degrees": 45.0},
        "render": str(output),
    }
    (reports_dir / "annotated_preset_restore.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Restored user annotation preset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
