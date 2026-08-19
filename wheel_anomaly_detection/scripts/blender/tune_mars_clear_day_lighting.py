from __future__ import annotations

import argparse
import bpy
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render an approved Martian lighting tuning step.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sun-strength", required=True, type=float)
    parser.add_argument("--exposure", required=True, type=float)
    parser.add_argument("--label", required=True)
    return parser.parse_args(values)


def configure_scene(scene: bpy.types.Scene, exposure: float) -> None:
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
    scene.view_settings.exposure = exposure
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

    simulator = bpy.data.scenes.get("Simulator")
    normal = bpy.data.scenes.get("Normal")
    perforation = bpy.data.scenes.get("Perforation")
    sun = bpy.data.objects.get("MarsClearDaySun")
    world = bpy.data.worlds.get("Mars_Clear_Day_World")
    chase = bpy.data.objects.get("SimulatorCamera")
    wheel_camera = bpy.data.objects.get("WheelCamera")
    terrain_material = bpy.data.materials.get("MartianTerrain_Visual")
    if any(value is None for value in (
        simulator, normal, perforation, sun, world, chase, wheel_camera, terrain_material
    )):
        raise RuntimeError("Clear-day lighting setup is incomplete")
    scenes = [simulator, normal, perforation]

    sun.data.energy = float(args.sun_strength)
    sun.data.angle = math.radians(0.35)
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("Mars world has no Background node")
    background.inputs["Strength"].default_value = 0.03
    for scene in scenes:
        scene.world = world
        configure_scene(scene, float(args.exposure))

    principled = next(
        (node for node in terrain_material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
        None,
    )
    if principled is None:
        raise RuntimeError("Terrain material has no Principled BSDF")
    principled.inputs["Emission Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    principled.inputs["Emission Strength"].default_value = 0.0

    simulator["solar_energy_watts_per_square_meter"] = float(args.sun_strength)
    simulator["world_strength"] = 0.03
    simulator["exposure_ev"] = float(args.exposure)
    simulator["terrain_emission_strength"] = 0.0

    label = str(args.label)
    rover_close = renders_dir / f"mars_tuning_{label}_rover_close.png"
    normal_close = renders_dir / f"mars_tuning_{label}_normal_wheel.png"
    perforation_close = renders_dir / f"mars_tuning_{label}_perforation_wheel.png"
    render(simulator, chase, rover_close)
    render(normal, wheel_camera, normal_close)
    render(perforation, wheel_camera, perforation_close)
    simulator.camera = chase

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "label": label,
        "sun_strength_watts_per_square_meter": float(args.sun_strength),
        "sun_angle_degrees": 0.35,
        "exposure_ev": float(args.exposure),
        "world_strength": 0.03,
        "terrain_emission_strength": float(principled.inputs["Emission Strength"].default_value),
        "shared_world": {scene.name: scene.world.name for scene in scenes},
        "renders": [str(rover_close), str(normal_close), str(perforation_close)],
    }
    (reports_dir / f"mars_tuning_{label}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Rendered Mars lighting tuning {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
