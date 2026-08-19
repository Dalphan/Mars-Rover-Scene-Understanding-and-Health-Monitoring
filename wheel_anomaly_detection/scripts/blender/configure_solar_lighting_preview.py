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
    parser = argparse.ArgumentParser(description="Configure approved solar lighting for the terrain preview.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(values)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.matrix_world.translation).to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
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
    camera = bpy.data.objects.get("SimulatorCamera")
    wheel_camera = bpy.data.objects.get("WheelCamera")
    pair_rig = bpy.data.collections.get("PAIR_RIG")
    if simulator is None or terrain is None or camera is None or wheel_camera is None:
        raise RuntimeError("Simulator terrain preview or required cameras are missing")

    if pair_rig is not None and pair_rig.name in simulator.collection.children:
        simulator.collection.children.unlink(pair_rig)
    if wheel_camera.name not in simulator.objects:
        simulator.collection.objects.link(wheel_camera)

    lighting = bpy.data.collections.get("SIMULATOR_SOLAR_LIGHTING")
    if lighting is None:
        lighting = bpy.data.collections.new("SIMULATOR_SOLAR_LIGHTING")
        simulator.collection.children.link(lighting)
    elif lighting.name not in simulator.collection.children:
        simulator.collection.children.link(lighting)
    for obj in list(lighting.objects):
        lighting.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)

    sun_data = bpy.data.lights.get("SimulatorSolarSun_Data")
    if sun_data is None:
        sun_data = bpy.data.lights.new("SimulatorSolarSun_Data", "SUN")
    sun_data.type = "SUN"
    sun_data.energy = 2.0
    sun_data.color = (1.0, 0.88, 0.75)
    sun_data.angle = math.radians(4.5)
    sun = bpy.data.objects.new("SimulatorSolarSun", sun_data)
    lighting.objects.link(sun)
    sun.location = Vector((-30.0, -40.0, 35.0))
    look_at(sun, Vector((0.0, 0.0, 0.0)))
    sun["role"] = "simulator_solar_light"
    sun["solar_altitude_degrees"] = 35.0
    sun["solar_angle_degrees"] = 4.5

    source_world = simulator.world
    world = bpy.data.worlds.get("Simulator_Solar_World")
    if world is None:
        world = source_world.copy() if source_world is not None else bpy.data.worlds.new("Simulator_Solar_World")
        world.name = "Simulator_Solar_World"
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        background = world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.035, 0.045, 0.06, 1.0)
    background.inputs["Strength"].default_value = 0.08
    simulator.world = world

    saved_camera_matrix = camera.matrix_world.copy()
    saved_camera = simulator.camera
    saved_visibility = {
        obj.name: obj.hide_render
        for obj in simulator.objects
        if obj.type in {"MESH", "CURVE"} and obj != terrain
    }
    for name in saved_visibility:
        bpy.data.objects[name].hide_render = True
    simulator.render.engine = "CYCLES"
    simulator.cycles.device = "CPU"
    simulator.cycles.samples = 32
    simulator.cycles.use_denoising = True
    simulator.render.film_transparent = False
    camera.data.clip_end = max(float(camera.data.clip_end), 500.0)

    camera.matrix_world.translation = Vector((0.0, 0.0, 73.0))
    look_at(camera, Vector((0.0, 0.0, 0.35)))
    overview = renders_dir / "new_terrain_solar_overview.png"
    render(simulator, camera, overview)
    camera.matrix_world.translation = Vector((38.0, -42.0, 27.0))
    look_at(camera, Vector((0.0, 0.0, 0.3)))
    oblique = renders_dir / "new_terrain_solar_oblique.png"
    render(simulator, camera, oblique)

    for name, hidden in saved_visibility.items():
        bpy.data.objects[name].hide_render = hidden
    camera.matrix_world = saved_camera_matrix
    simulator.camera = saved_camera
    simulator["lighting_status"] = "solar_preview_installed_pending_approval"
    simulator["solar_altitude_degrees"] = 35.0
    simulator["solar_energy"] = 2.0
    simulator["solar_angle_degrees"] = 4.5
    simulator["normal_map_adjusted"] = False

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "scene": simulator.name,
        "sun": {
            "object": sun.name,
            "type": sun_data.type,
            "energy": sun_data.energy,
            "color": list(sun_data.color),
            "altitude_degrees": 35.0,
            "angle_degrees": math.degrees(sun_data.angle),
        },
        "world": {
            "name": world.name,
            "background_strength": float(background.inputs["Strength"].default_value),
        },
        "pair_rig_still_in_other_scenes": {
            scene.name: pair_rig is not None and pair_rig.name in scene.collection.children
            for scene in bpy.data.scenes
            if scene.name in {"Normal", "Perforation"}
        },
        "wheel_camera_in_simulator": wheel_camera.name in simulator.objects,
        "normal_map_adjusted": False,
        "collision_enabled": False,
        "renders": [str(overview), str(oblique)],
    }
    (reports_dir / "solar_lighting_preview.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Configured simulator solar lighting preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
