from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Tune Milestone 5 lighting and render settings from M3.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-label", default="rover_simulator_m3_rock_library.blend")
    return parser.parse_args(values)


def set_world(scene: bpy.types.Scene, settings: dict) -> dict:
    if scene.world is None:
        scene.world = bpy.data.worlds.new(f"{scene.name}_M5_World")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is None:
        background = scene.world.node_tree.nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = tuple(settings["world_color"])
    background.inputs["Strength"].default_value = float(settings["world_strength"])
    return {
        "world": scene.world.name,
        "color": list(background.inputs["Color"].default_value),
        "strength": float(background.inputs["Strength"].default_value),
    }


def tune_scene(scene: bpy.types.Scene, settings: dict) -> dict:
    scene.view_settings.view_transform = settings["view_transform"]
    scene.view_settings.look = settings["look"]
    scene.view_settings.exposure = float(settings["exposure_ev"])
    scene.render.resolution_x = int(settings["resolution_x"])
    scene.render.resolution_y = int(settings["resolution_y"])
    scene.render.resolution_percentage = int(settings["resolution_percentage"])
    scene.render.film_transparent = False
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(settings["cycles_samples"])
        scene.cycles.use_denoising = bool(settings["cycles_denoising"])
    scene["milestone_5_lighting_render"] = True
    scene["render_resolution"] = f"{scene.render.resolution_x}x{scene.render.resolution_y}@{scene.render.resolution_percentage}%"
    scene["render_engine_preserved"] = scene.render.engine
    scene["lighting_render_preset"] = "m5_solar_closeup"
    return {
        "scene": scene.name,
        "engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "exposure_ev": scene.view_settings.exposure,
        "cycles_samples": scene.cycles.samples if scene.render.engine == "CYCLES" else None,
        "world": set_world(scene, settings),
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    settings = dict(config["lighting_render_m5"])
    light = bpy.data.objects.get(config["sun_object"])
    if light is None or light.type != "LIGHT" or light.data.type != "SUN":
        raise RuntimeError(f"Missing active solar Sun {config['sun_object']}")
    old_light = bpy.data.objects.get("MarsClearDaySun")
    old_light_state = None
    if old_light is not None:
        old_light_state = {"hide_render": old_light.hide_render, "hide_viewport": old_light.hide_viewport}
        # Keep the historical light datablock for rollback, but ensure the M5
        # copy has exactly one render-active Sun in the pair scenes.
        old_light.hide_render = True

    old_light_state_active = {
        "energy": float(light.data.energy),
        "angle_degrees": math.degrees(float(light.data.angle)),
        "color": list(light.data.color),
    }
    light.data.energy = float(settings["sun_energy"])
    light.data.angle = math.radians(float(settings["sun_angle_degrees"]))
    light.data.color = tuple(settings["sun_color"][:3])
    light.data.use_shadow = True
    light["milestone_5_lighting_render"] = True
    light["preset"] = "m5_solar_closeup"
    light["angle_degrees"] = float(settings["sun_angle_degrees"])
    light["energy"] = float(settings["sun_energy"])

    scenes = []
    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is None:
            raise RuntimeError(f"Missing required scene {scene_name}")
        scenes.append(tune_scene(scene, settings))

    root = bpy.data.objects.get(config["rover_root"])
    if root is not None:
        root["milestone_5_lighting_render"] = True
        root["milestone_5_source_milestone"] = "3-rock-library"
        root["active_terrain_distribution"] = "M3_uniform_reference"

    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "milestone": "5-lighting-render",
        "source_scene": args.source_label,
        "output_blend": str(output_blend),
        "preset": settings,
        "active_sun": {
            "object": light.name,
            "before": old_light_state_active,
            "after": {
                "energy": float(light.data.energy),
                "angle_degrees": math.degrees(float(light.data.angle)),
                "color": list(light.data.color),
            },
        },
        "historical_light": {"object": old_light.name, "before": old_light_state} if old_light is not None else None,
        "scenes": scenes,
        "source_milestone_reverted_to": "3-rock-library",
        "distribution_clustered_m4_active": False,
        "pair_context_unchanged": True,
        "camera_changed": False,
        "terrain_material_changed": False,
        "collision_enabled": False,
        "external_resources": [],
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built Milestone 5 lighting/render variant from M3: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
