"""Render a presentation-only overview from the immutable generation scene.

Adversarial review: do not import or edit the original NASA asset, save the
source Blend, alter dataset raster files, or claim this overview is a sampled
inspection view. Preserve native terrain geometry and materials. Fit the full
rover into the frame, warm Eevee's instance/shadow caches, and verify the source
checksum before and after rendering. Resolve relocated external terrain
textures from the existing canonical geospatial outputs, not new imagery.
This wide view intentionally includes
coarse macroterrain outside the localized procedural ground patch.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blender.microterrain.common import look_at, render, rover_objects
from scripts.blender.render_mars_lighting_pilot import _configure_sun_world


SOURCE = PROJECT_ROOT / "outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend"
EXPECTED_SOURCE_SHA256 = "9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030"
OUTPUT = PROJECT_ROOT / "outputs/thesis_figures/rover_terrain_overview_v2.png"
REPORT = OUTPUT.with_suffix(".json")


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.perf_counter()
    source_before = checksum(SOURCE)
    if source_before != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Source checksum differs from the approved generation scene")
    if Path(bpy.data.filepath).resolve() != SOURCE.resolve():
        raise RuntimeError("Launch Blender with the approved source scene already loaded")
    if OUTPUT.exists() or REPORT.exists():
        raise RuntimeError("Refusing to overwrite an existing presentation export")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    for name in ("Rover", "GaleTerrainVisual", "MicroterrainPatch_L1"):
        if name not in bpy.data.objects:
            raise RuntimeError(f"Required scene asset is missing: {name}")
    meshes = [obj for obj in rover_objects(bpy.data.objects["Rover"]) if obj.type == "MESH" and not obj.hide_render]
    if len([obj for obj in meshes if obj.name.startswith("wheel_")]) != 6:
        raise RuntimeError("The overview must preserve all six canonical wheels")

    terrain_directory = PROJECT_ROOT / "outputs/anomaly_detection_2/gale_terrain/geospatial"
    terrain_images = {}
    terrain_nodes = []
    for material in bpy.data.materials:
        if not material.name.startswith("GaleTerrain") or not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                terrain_images[node.image.name] = node.image
                terrain_nodes.append(node)
    texture_resolution = []
    for image in terrain_images.values():
        original_path = image.filepath
        candidate = terrain_directory / Path(bpy.path.abspath(original_path)).name
        if not candidate.is_file():
            raise RuntimeError(f"Cannot resolve the original terrain texture: {original_path}")
        # A serialized missing-image cache can survive reload(). Load the same
        # canonical pixels into a fresh runtime datablock, without saving the
        # source scene or changing the shader graph or color interpretation.
        replacement = bpy.data.images.load(candidate.as_posix(), check_existing=False)
        replacement.colorspace_settings.name = image.colorspace_settings.name
        replacement.alpha_mode = image.alpha_mode
        first_pixel = tuple(replacement.pixels[:4])
        if not replacement.has_data or min(replacement.size) < 1 or len(first_pixel) != 4:
            raise RuntimeError(f"Terrain texture did not load: {candidate}")
        for node in terrain_nodes:
            if node.image == image:
                node.image = replacement
        texture_resolution.append({
            "image": image.name,
            "original_path": original_path,
            "resolved_path": str(candidate),
            "sha256": checksum(candidate),
            "dimensions": list(replacement.size),
        })
    print(f"THESIS_OVERVIEW: resolved terrain images {texture_resolution}", flush=True)

    lighting_config = json.loads((PROJECT_ROOT / "configs/blender/mars_lighting_pilot.json").read_text(encoding="utf-8"))
    preset = next(entry for entry in lighting_config["presets"] if entry["id"] == "mars_clear_refined")
    _configure_sun_world(preset)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = preset["look"]
    scene.view_settings.exposure = float(preset["exposure_ev"])
    scene.use_nodes = False
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100
    scene.eevee.taa_render_samples = 128

    corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
    bounds_min = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    bounds_max = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    target = (bounds_min + bounds_max) * 0.5
    target.z = bounds_min.z + 0.43 * (bounds_max.z - bounds_min.z)
    camera_data = bpy.data.cameras.new("ThesisOverviewCamera_data")
    camera = bpy.data.objects.new("ThesisOverviewCamera", camera_data)
    scene.collection.objects.link(camera)
    camera_data.type = "PERSP"
    camera_data.lens = 38.0
    camera_data.sensor_width = 36.0
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.clip_start = 0.05
    camera_data.clip_end = 500.0
    camera_data.dof.use_dof = False
    direction = Vector((6.8, -8.5, 5.2)).normalized()
    scene.camera = camera
    projected = []
    for step in range(31):
        distance = 5.0 + step * 0.35
        camera.location = target + direction * distance
        look_at(camera, target)
        bpy.context.view_layer.update()
        projected = [world_to_camera_view(scene, camera, point) for point in corners]
        if all(point.z > 0 and 0.12 <= point.x <= 0.88 and 0.10 <= point.y <= 0.90 for point in projected):
            break
    else:
        raise RuntimeError("Could not frame the complete rover with safe image margins")

    print("THESIS_OVERVIEW: source checked, full rover framed, warming renderer", flush=True)
    bpy.ops.render.render(write_still=False)
    render(scene, camera, OUTPUT, (1600, 1200))
    source_after = checksum(SOURCE)
    if source_after != source_before:
        raise RuntimeError("The immutable source scene changed")
    report = {
        "purpose": "presentation_overview_not_dataset_sample",
        "source_blend": str(SOURCE),
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "source_unchanged": True,
        "output": str(OUTPUT),
        "output_sha256": checksum(OUTPUT),
        "blender_version": bpy.app.version_string,
        "engine": scene.render.engine,
        "resolution": [1600, 1200],
        "samples": int(scene.eevee.taa_render_samples),
        "lighting_preset": preset["id"],
        "native_geometry_and_materials_preserved": True,
        "terrain_texture_resolution": texture_resolution,
        "camera_location_m": list(camera.location),
        "camera_target_m": list(target),
        "camera_focal_length_mm": float(camera_data.lens),
        "projected_rover_bounds": {
            "min_x": min(float(point.x) for point in projected),
            "max_x": max(float(point.x) for point in projected),
            "min_y": min(float(point.y) for point in projected),
            "max_y": max(float(point.y) for point in projected),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Overview camera is not one of the inspection sampling families",
            "Local procedural detail occupies only its original ground patch",
            "Lighting is a visual preset, not a radiometric reconstruction",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"THESIS_OVERVIEW: exported {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
