"""Validate the persisted Gale terrain, semantic rover, MAHLI and lighting."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy


EXPECTED_WHEELS = {
    "wheel_front_left",
    "wheel_front_right",
    "wheel_middle_left",
    "wheel_middle_right",
    "wheel_rear_left",
    "wheel_rear_right",
}


def _arguments() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--report", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    config = globals().get("CONFIG_PATH") or args.config
    report = globals().get("REPORT_PATH") or args.report
    if not config or not report:
        raise RuntimeError("CONFIG_PATH and REPORT_PATH are required")
    return Path(config).resolve(), Path(report).resolve()


def validate(config_path: Path, report_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata_path = Path(bpy.context.scene.get("geospatial_metadata", ""))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    expected_texture = metadata.get("texture_selection", {})
    errors = []
    visual = bpy.data.objects.get(config["terrain"]["visual_object"])
    collision = bpy.data.objects.get(config["terrain"]["collision_object"])
    if visual is None or visual.type != "MESH":
        errors.append("Missing visual terrain mesh")
    if collision is None or collision.type != "MESH":
        errors.append("Missing collision terrain mesh")
    if visual:
        if not math.isclose(visual.dimensions.x, 256.0, abs_tol=1e-4) or not math.isclose(visual.dimensions.y, 256.0, abs_tol=1e-4):
            errors.append(f"Visual terrain metric extent is invalid: {list(visual.dimensions)}")
        if visual.get("vertical_exaggeration") != 1.0:
            errors.append("Visual terrain vertical exaggeration is not 1.0")
        if [layer.name for layer in visual.data.uv_layers] != ["GeoreferencedUV"]:
            errors.append("Visual terrain georeferenced UV layer is missing")
        if not visual.data.materials or visual.data.materials[0].get("microtexture_enabled") is not False:
            errors.append("Visual terrain material contract is invalid")
        elif visual.data.materials[0].get("texture_product") != expected_texture.get("chosen") or visual.data.materials[0].get("color_type") != expected_texture.get("color_type"):
            errors.append("Visual terrain texture does not match the geospatial selection report")
    if collision:
        if not collision.hide_render:
            errors.append("Collision terrain must be hidden from renders")
        if collision.get("mesh_step_m") != float(config["terrain"]["collision_step_m"]):
            errors.append("Collision mesh step does not match config")
        if visual and len(collision.data.vertices) >= len(visual.data.vertices):
            errors.append("Collision mesh is not simplified")

    wheel_names = {obj.name for obj in bpy.data.objects if obj.name.startswith("wheel_")}
    if wheel_names != EXPECTED_WHEELS:
        errors.append(f"Semantic wheel set mismatch: {sorted(wheel_names)}")
    root = bpy.data.objects.get(config["rover"]["root_object"])
    if root is None:
        errors.append("Missing Rover root")

    camera = bpy.data.objects.get(config["camera"]["name"])
    if camera is None or camera.type != "CAMERA":
        errors.append("Missing MAHLI camera")
    else:
        fov = math.degrees(camera.data.angle_x)
        if not 34.0 <= fov <= 39.4:
            errors.append(f"MAHLI horizontal FOV outside expected range: {fov}")
        if not math.isclose(camera.data.lens, float(config["camera"]["focal_length_mm"]), abs_tol=1e-6):
            errors.append("MAHLI focal length mismatch")
        if camera.get("target_wheel") != config["camera"]["target_wheel"]:
            errors.append("MAHLI target wheel mismatch")
    scene = bpy.context.scene
    if [scene.render.resolution_x, scene.render.resolution_y] != config["camera"]["resolution"]:
        errors.append("MAHLI render resolution mismatch")

    sun = bpy.data.objects.get(config["lighting"]["sun_name"])
    if sun is None or sun.type != "LIGHT" or sun.data.type != "SUN":
        errors.append("Missing nominal Gale Sun")
    else:
        for key in ("azimuth_deg", "elevation_deg"):
            if not math.isclose(float(sun.get(key, math.nan)), float(config["lighting"][key]), abs_tol=1e-6):
                errors.append(f"Sun {key} mismatch")
    if scene.world is None or not math.isclose(float(scene.world.get("ambient_strength", math.nan)), float(config["lighting"]["ambient_strength"]), abs_tol=1e-6):
        errors.append("Nominal ambient lighting mismatch")
    if scene.get("microtexture_enabled") is not False or scene.get("bulk_generation_enabled") is not False:
        errors.append("Scope gates for microtexture/bulk generation are invalid")
    if scene.get("terrain_texture_product") != expected_texture.get("chosen") or scene.get("terrain_color_type") != expected_texture.get("color_type"):
        errors.append("Scene color metadata does not match the geospatial selection report")

    report = {
        "ok": not errors,
        "blend": bpy.data.filepath,
        "errors": errors,
        "visual_terrain": None if not visual else {"vertices": len(visual.data.vertices), "faces": len(visual.data.polygons), "dimensions_m": list(visual.dimensions)},
        "collision_terrain": None if not collision else {"vertices": len(collision.data.vertices), "faces": len(collision.data.polygons), "dimensions_m": list(collision.dimensions)},
        "wheels": sorted(wheel_names),
        "camera": None if not camera else {"name": camera.name, "focal_length_mm": camera.data.lens, "horizontal_fov_deg": math.degrees(camera.data.angle_x), "resolution": [scene.render.resolution_x, scene.render.resolution_y]},
        "sun": None if not sun else {"name": sun.name, "azimuth_deg": sun.get("azimuth_deg"), "elevation_deg": sun.get("elevation_deg"), "energy": sun.data.energy},
        "texture_selection": expected_texture,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("Gale scene validation failed: " + "; ".join(errors))
    return report


if __name__ == "__main__":
    validate(*_arguments())
