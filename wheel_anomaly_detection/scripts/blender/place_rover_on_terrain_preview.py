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
    parser = argparse.ArgumentParser(description="Place the rover at the approved visual terrain pose.")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(values)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.matrix_world.translation).to_track_quat("-Z", "Y").to_euler()


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
    sun = bpy.data.objects.get("SimulatorSolarSun")
    material = bpy.data.materials.get("MartianTerrain_Visual")
    if any(value is None for value in (simulator, terrain, root, chase, wheel_camera, sun, material)):
        raise RuntimeError("Approved terrain, rover, cameras, sun, or material is missing")
    normal_map = next((node for node in material.node_tree.nodes if node.type == "NORMAL_MAP"), None)
    if normal_map is None:
        raise RuntimeError("Terrain material has no Normal Map node")

    normal_map.inputs["Strength"].default_value = 0.30
    sun.data.energy = 3.5
    root.location = Vector((-18.0, -4.0, 0.342345397069699))
    root.rotation_euler = (0.0, 0.0, math.radians(45.0))
    bpy.context.view_layer.update()
    root["visual_terrain_pose"] = json.dumps({
        "x": -18.0,
        "y": -4.0,
        "z": 0.342345397069699,
        "yaw_degrees": 45.0,
        "height_delta_meters": 0.023901939392089844,
        "mean_slope_degrees": 0.5308178072784638,
        "source": "approved flat-area audit",
    }, sort_keys=True)

    for obj in root.children_recursive:
        if obj.get("role") in {"ground_probe", "ground_probe_layout"}:
            obj.hide_render = True
    bpy.data.objects["Wheel_Skin_Normal"].hide_render = False
    bpy.data.objects["Wheel_Skin_Normal"].hide_viewport = False

    simulator.render.engine = "CYCLES"
    simulator.cycles.device = "CPU"
    simulator.cycles.samples = 40
    simulator.cycles.use_denoising = True
    simulator.render.resolution_x = 800
    simulator.render.resolution_y = 600
    simulator.render.resolution_percentage = 100
    simulator.render.image_settings.file_format = "PNG"
    simulator.render.film_transparent = False
    chase.data.clip_end = max(float(chase.data.clip_end), 500.0)

    saved_chase_matrix = chase.matrix_world.copy()
    overview = renders_dir / "terrain_with_rover_overview.png"
    chase.matrix_world.translation = Vector((0.0, 0.0, 73.0))
    look_at(chase, Vector((0.0, 0.0, 0.35)))
    render(simulator, chase, overview)
    chase.matrix_world = saved_chase_matrix
    bpy.context.view_layer.update()

    rover_close = renders_dir / "terrain_with_rover_close.png"
    render(simulator, chase, rover_close)
    wheel_close = renders_dir / "terrain_with_rover_wheel_close.png"
    render(simulator, wheel_camera, wheel_close)
    simulator.camera = chase

    simulator["terrain_status"] = "visual_preview_with_rover_pending_approval"
    simulator["lighting_status"] = "solar_preview_installed"
    simulator["solar_energy"] = 3.5
    simulator["normal_map_adjusted"] = True
    simulator["normal_map_strength"] = 0.30
    simulator["rover_visual_pose_approved"] = True
    simulator["collision_enabled"] = False
    simulator["terrain_collision_enabled"] = False
    simulator["simulator_milestone"] = "1-terrain-preview"

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "scene": simulator.name,
        "terrain": terrain.name,
        "normal_strength": 0.30,
        "solar_energy": 3.5,
        "rover_pose": json.loads(root["visual_terrain_pose"]),
        "collision_enabled": False,
        "persistent_solver_enabled": False,
        "renders": {
            "overview": str(overview),
            "rover_close": str(rover_close),
            "wheel_close": str(wheel_close),
        },
    }
    (reports_dir / "rover_terrain_visual_pose.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Placed rover at approved visual terrain pose")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
