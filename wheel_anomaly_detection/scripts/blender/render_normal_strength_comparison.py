from __future__ import annotations

import argparse
import bpy
import json
import sys
from pathlib import Path

from mathutils import Vector


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render approved normal-strength terrain comparisons.")
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
    camera = bpy.data.objects.get("SimulatorCamera")
    sun = bpy.data.objects.get("SimulatorSolarSun")
    material = bpy.data.materials.get("MartianTerrain_Visual")
    if simulator is None or terrain is None or camera is None or sun is None or material is None:
        raise RuntimeError("Solar terrain preview is incomplete")
    normal_map = next((node for node in material.node_tree.nodes if node.type == "NORMAL_MAP"), None)
    if normal_map is None:
        raise RuntimeError("Terrain material has no Normal Map node")

    sun.data.energy = 3.5
    simulator["solar_energy"] = 3.5
    saved_camera_matrix = camera.matrix_world.copy()
    saved_camera = simulator.camera
    saved_strength = float(normal_map.inputs["Strength"].default_value)
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
    simulator.render.resolution_x = 800
    simulator.render.resolution_y = 600
    simulator.render.resolution_percentage = 100
    simulator.render.image_settings.file_format = "PNG"
    simulator.render.film_transparent = False
    camera.data.clip_end = max(float(camera.data.clip_end), 500.0)

    artifacts: list[dict[str, object]] = []
    for strength, label in ((0.30, "030"), (0.15, "015")):
        normal_map.inputs["Strength"].default_value = strength
        camera.matrix_world.translation = Vector((0.0, 0.0, 73.0))
        look_at(camera, Vector((0.0, 0.0, 0.35)))
        overview = renders_dir / f"terrain_sun35_normal_{label}_overview.png"
        render(simulator, camera, overview)
        camera.matrix_world.translation = Vector((38.0, -42.0, 27.0))
        look_at(camera, Vector((0.0, 0.0, 0.3)))
        oblique = renders_dir / f"terrain_sun35_normal_{label}_oblique.png"
        render(simulator, camera, oblique)
        artifacts.append({
            "normal_strength": strength,
            "overview": str(overview),
            "oblique": str(oblique),
        })

    normal_map.inputs["Strength"].default_value = saved_strength
    for name, hidden in saved_visibility.items():
        bpy.data.objects[name].hide_render = hidden
    camera.matrix_world = saved_camera_matrix
    simulator.camera = saved_camera
    simulator["normal_map_adjusted"] = False
    simulator["normal_comparison_pending_selection"] = True

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "solar_energy_saved": 3.5,
        "solar_altitude_degrees": float(simulator.get("solar_altitude_degrees", 35.0)),
        "normal_strength_saved": saved_strength,
        "normal_strength_variants": [0.30, 0.15],
        "selection_pending": True,
        "collision_enabled": False,
        "artifacts": artifacts,
    }
    (reports_dir / "normal_strength_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Rendered normal-strength comparison at solar energy 3.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
