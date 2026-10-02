"""Render controlled, presentation-only examples of existing generation settings.

Adversarial review: use only the canonical derived scene, never the original
NASA GLB. Do not save Blend files, overwrite existing figures, mutate dataset
hardlinks, resample terrain, or change production contracts. Reopen the source
for each example to prevent dust/wear accumulation, resolve relocated textures
from canonical files, and warm Eevee before exporting. Lighting presets include
their documented dust/material response: this is not a lighting-only ablation.
The bounded seven-frame run keeps camera/roll fixed for appearance comparisons,
checks framing/contact/terrain coverage and source hashes, and requires visual
QA of every exported image. These are illustration renders, not dataset items.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import bpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.blender import render_mars_lighting_pilot as lighting
from scripts.blender import render_wheel_surface_wear_pilot as wear
from scripts.blender.microterrain.common import render
from scripts.blender.render_domain_randomization_preview import (
    _align_rover_to_patch,
    _framing_gate,
    _matrix,
    _matrix_values,
    _settle_rover_on_patch,
    _terrain_footprint,
)
from scripts.blender.render_wheel_camera_pose_pilot import _apply_terrain_color_grade
from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose


SOURCE = PROJECT_ROOT / "outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend"
EXPECTED_SHA256 = "9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030"
OUTPUT = PROJECT_ROOT / "outputs/thesis_figures/domain_randomization_v1"
TARGET = "wheel_middle_left"
RESOLUTION = (1200, 900)
CASES = (
    ("dr_baseline", "mars_dusty_refined", "surface_current", "A_overhead"),
    ("dr_lighting_clear", "mars_clear_refined", "surface_current", "A_overhead"),
    ("dr_wear_light", "mars_dusty_refined", "wear_light", "A_overhead"),
    ("dr_wear_evident", "mars_dusty_refined", "wear_evident", "A_overhead"),
    ("dr_pose_leading", "mars_dusty_refined", "surface_current", "C_leading_three_quarter"),
    ("dr_pose_trailing", "mars_dusty_refined", "surface_current", "C_trailing_three_quarter"),
    ("dr_pose_detail", "mars_dusty_refined", "surface_current", "D_upper_detail"),
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(filename: str) -> dict:
    return json.loads((PROJECT_ROOT / "configs/blender" / filename).read_text(encoding="utf-8"))


def resolve_terrain_textures() -> list[dict]:
    directory = PROJECT_ROOT / "outputs/anomaly_detection_2/gale_terrain/geospatial"
    nodes = [
        node
        for material in bpy.data.materials
        if material.name.startswith("GaleTerrain") and material.use_nodes
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image is not None
    ]
    original_images = {node.image.name: node.image for node in nodes}
    resolved = []
    for original in original_images.values():
        path = directory / Path(bpy.path.abspath(original.filepath)).name
        if not path.is_file():
            raise RuntimeError(f"Missing canonical terrain texture: {path}")
        replacement = bpy.data.images.load(path.as_posix(), check_existing=False)
        replacement.colorspace_settings.name = original.colorspace_settings.name
        replacement.alpha_mode = original.alpha_mode
        first_pixel = tuple(replacement.pixels[:4])
        if not replacement.has_data or min(replacement.size) < 1 or len(first_pixel) != 4:
            raise RuntimeError(f"Terrain texture did not load: {path}")
        for node in nodes:
            if node.image == original:
                node.image = replacement
        resolved.append({"source": str(path), "sha256": checksum(path)})
    if not resolved:
        raise RuntimeError("No canonical terrain image nodes were found")
    return resolved


def main() -> None:
    started = time.perf_counter()
    before = checksum(SOURCE)
    if before != EXPECTED_SHA256:
        raise RuntimeError("The approved generation scene checksum changed")
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing figure outputs: {OUTPUT}")
    domain = read_config("domain_randomization_dataset_v1.json")
    light_config = read_config("mars_lighting_pilot.json")
    wear_config = read_config("wheel_surface_wear_pilot.json")
    poses = read_config("wheel_camera_poses.json")
    rolls = read_config("wheel_pose_sampling.json")
    render_config = read_config("clean_batch.json")["render"]
    if tuple(bpy.app.version[:2]) != tuple(render_config["blender_major_minor"]):
        raise RuntimeError(f"Wrong Blender version: {bpy.app.version_string}")
    preset_by_id = {item["id"]: item for item in light_config["presets"]}
    wear_by_id = {item["id"]: item for item in wear_config["variants"]}
    pose_by_id = {item["id"]: item for item in poses["poses"]}
    OUTPUT.mkdir(parents=True)
    results = []
    reference_camera = None
    reference_geometry = None

    for index, (name, preset_id, wear_id, pose_id) in enumerate(CASES, start=1):
        print(f"THESIS_DR: {index}/{len(CASES)} {name}", flush=True)
        bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
        scene = bpy.context.scene
        if not bool(scene.get("wheel_pose_sampling_ready", False)):
            raise RuntimeError("Source is not ready for canonical wheel poses")
        textures = resolve_terrain_textures()
        scene.render.engine = str(render_config["engine"])
        scene.eevee.taa_render_samples = int(render_config["samples"])
        scene.render.film_transparent = False
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.image_settings.color_depth = "8"
        scene.render.resolution_x, scene.render.resolution_y = RESOLUTION
        scene.render.resolution_percentage = 100
        scene.frame_set(1)
        wheels = [bpy.data.objects[item] for item in rolls["wheels"]]
        if len(wheels) != 6:
            raise RuntimeError("All six canonical wheels must remain present")
        base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
        target = bpy.data.objects[TARGET]
        alignment = domain["gates"]["terrain_alignment"]
        root = bpy.data.objects[alignment["rover_root_object"]]
        patch = bpy.data.objects[domain["gates"]["terrain_patch_object"]]
        _align_rover_to_patch(scene, target, alignment, root.matrix_world.copy())
        apply_shared_roll(wheels, base_local, 0.0)
        contact = _settle_rover_on_patch(scene, target, patch, alignment)
        if contact["ok"] is not True:
            raise RuntimeError(f"Terrain-contact gate failed: {contact}")
        meshes = wear._target_meshes(target)
        geometry = wear._geometry_signature(meshes)
        if reference_geometry is None:
            reference_geometry = geometry
        elif geometry != reference_geometry:
            raise RuntimeError("Geometry or wheel transform drift across illustrations")
        dimensions = wear._wheel_dimensions(target, meshes)
        variant = wear_by_id[wear_id]
        materials = []
        if variant["wear_enabled"]:
            materials, _affected = wear._clone_target_materials(meshes, name)
        camera = lighting._configure_camera(poses["camera"])
        pose = pose_by_id[pose_id]
        basis = camera_pose(camera, target, pose, poses["camera"])
        scene.camera = camera
        framing = _framing_gate(scene, camera, target, basis, pose, domain["gates"])
        footprint = _terrain_footprint(scene, camera, patch, domain["gates"]["terrain_minimum_edge_margin_m"])
        if not framing["ok"] or not footprint["ok"]:
            raise RuntimeError(f"Illustration framing/ground-coverage gate failed: {framing} {footprint}")
        camera_matrix = _matrix_values(camera.matrix_world)
        if reference_camera is None:
            reference_camera = camera_matrix
        if pose_id == "A_overhead" and max(abs(a - b) for a, b in zip(camera_matrix, reference_camera)) > 1e-7:
            raise RuntimeError("Appearance comparisons do not share the same camera")
        preset = preset_by_id[preset_id]
        lighting._configure_sun_world(preset)
        _apply_terrain_color_grade(light_config["terrain_grade"])
        if preset["dust_layer"]:
            lighting._apply_dust_layer(preset["dust"])
        mask = None
        if variant["wear_enabled"]:
            mask = PROJECT_ROOT / "outputs/anomaly_detection_2/surface_wear_pilot" / variant["mask_filename"]
            if not mask.is_file():
                raise RuntimeError(f"Retained pilot scratch pattern missing: {mask}")
            wear._apply_wear(materials, target, dimensions, mask, wear_config["shader"])
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = preset["look"]
        scene.view_settings.exposure = float(preset["exposure_ev"])
        lighting._configure_compositor(scene, preset)
        bpy.context.view_layer.update()
        bpy.ops.render.render(write_still=False)
        path = OUTPUT / f"{name}.png"
        render(scene, camera, path, RESOLUTION)
        results.append({
            "image": str(path), "sha256": checksum(path),
            "condition": "normal_no_hole", "target_wheel": TARGET,
            "lighting_preset": preset_id, "surface_wear": wear_id,
            "camera_pose": pose_id, "roll_degrees": 0.0,
            "camera_matrix_world": camera_matrix, "wheel_geometry": geometry,
            "framing": framing, "terrain_footprint": footprint,
            "terrain_contact": contact, "terrain_textures": textures,
            "wear_mask": str(mask) if mask else None,
            "wear_mask_sha256": checksum(mask) if mask else None,
        })
        print(f"THESIS_DR: exported {path.name}", flush=True)

    after = checksum(SOURCE)
    if after != before:
        raise RuntimeError("Canonical source changed during figure rendering")
    report = {
        "purpose": "controlled_method_illustrations_not_dataset_samples",
        "source_blend": str(SOURCE), "source_sha256_before": before,
        "source_sha256_after": after, "source_unchanged": True,
        "blender_version": bpy.app.version_string, "resolution": list(RESOLUTION),
        "engine": render_config["engine"], "samples": render_config["samples"],
        "illustrations": results, "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Clear/dusty presets also change deposited dust and camera response",
            "All examples are normal and contain no injected hole",
            "Scratch patterns are retained deterministic pilot masks",
            "No production metadata or dataset files were changed",
        ],
    }
    (OUTPUT / "figures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("THESIS_DR: all seven figures exported; source unchanged", flush=True)


if __name__ == "__main__":
    main()
