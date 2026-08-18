"""Render deterministic, gated samples from the final domain-randomization contract."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--lighting-config", type=Path, required=True)
    parser.add_argument("--wear-config", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return tuple(
        value.resolve()
        for value in (
            args.source_blend,
            args.config,
            args.plan,
            args.lighting_config,
            args.wear_config,
            args.pose_config,
            args.sampling_config,
            args.output_dir,
        )
    )


def _matrix(values) -> Matrix:
    values = list(map(float, values))
    return Matrix([values[index : index + 4] for index in range(0, 16, 4)])


def _matrix_values(matrix: Matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _jittered_preset(base: dict, sample: dict) -> dict:
    preset = copy.deepcopy(base)
    jitter = sample["lighting_jitter"]
    preset["sun"]["azimuth_deg"] = float(preset["sun"]["azimuth_deg"]) + float(jitter["sun_azimuth_degrees"])
    preset["sun"]["elevation_deg"] = float(preset["sun"]["elevation_deg"]) + float(jitter["sun_elevation_degrees"])
    preset["sun"]["energy"] = float(preset["sun"]["energy"]) * float(jitter["sun_energy_scale"])
    preset["world"]["strength"] = float(preset["world"]["strength"]) * float(jitter["world_strength_scale"])
    preset["exposure_ev"] = float(preset["exposure_ev"]) + float(jitter["exposure_ev"])
    return preset


def _jitter_camera(camera, basis: dict, sample: dict) -> Vector:
    from scripts.blender.microterrain.common import look_at

    jitter = sample["camera_jitter"]
    offset = list(map(float, jitter["position_basis_m"]))
    camera.location += basis["outward"] * offset[0] + basis["forward"] * offset[1] + basis["up"] * offset[2]
    rotation = camera.matrix_world.to_3x3()
    right = (rotation @ Vector((1.0, 0.0, 0.0))).normalized()
    camera_up = (rotation @ Vector((0.0, 1.0, 0.0))).normalized()
    distance = (basis["target"] - camera.location).length
    aim_target = (
        basis["target"]
        + right * math.tan(math.radians(float(jitter["aim_yaw_degrees"]))) * distance
        + camera_up * math.tan(math.radians(float(jitter["aim_pitch_degrees"]))) * distance
    )
    look_at(camera, aim_target)
    bpy.context.view_layer.update()
    return aim_target


def _patch_bounds(patch: bpy.types.Object) -> dict:
    corners = [patch.matrix_world @ Vector(corner) for corner in patch.bound_box]
    return {
        "x_min": min(float(point.x) for point in corners),
        "x_max": max(float(point.x) for point in corners),
        "y_min": min(float(point.y) for point in corners),
        "y_max": max(float(point.y) for point in corners),
        "plane_z": 0.5 * (min(float(point.z) for point in corners) + max(float(point.z) for point in corners)),
    }


def _bbox_center(obj: bpy.types.Object) -> Vector:
    return sum((obj.matrix_world @ Vector(corner) for corner in obj.bound_box), Vector()) / 8.0


def _align_rover_to_patch(scene, target: bpy.types.Object, settings: dict, base_root_matrix: Matrix) -> Vector:
    """Place a target on the fixed detail patch without moving terrain."""
    root = bpy.data.objects.get(settings["rover_root_object"])
    if root is None:
        raise RuntimeError("Rover root required by terrain alignment is missing")
    root.matrix_world = base_root_matrix.copy()
    bpy.context.view_layer.update()
    translation = Vector((0.0, 0.0, 0.0))
    mode = str(settings["mode"])
    if mode == "translate_rover_target_to_anchor_xy":
        anchor = bpy.data.objects.get(str(settings["anchor_wheel_object"]))
        if anchor is None:
            raise RuntimeError("Terrain-alignment anchor wheel is missing")
        delta = _bbox_center(anchor) - _bbox_center(target)
        translation = Vector((float(delta.x), float(delta.y), 0.0))
        root.matrix_world = Matrix.Translation(translation) @ base_root_matrix
        bpy.context.view_layer.update()
    elif mode == "translate_rover_target_to_left_counterpart" and target.name.endswith("_right"):
        counterpart = bpy.data.objects.get(target.name.removesuffix("_right") + "_left")
        if counterpart is None:
            raise RuntimeError(f"Left counterpart missing for {target.name}")
        translation = _bbox_center(counterpart) - _bbox_center(target)
        root.matrix_world = Matrix.Translation(translation) @ base_root_matrix
        bpy.context.view_layer.update()
    elif mode != "translate_rover_target_to_left_counterpart":
        raise RuntimeError(f"Unsupported terrain-alignment mode: {mode}")
    return translation


def _settle_rover_on_patch(
    scene,
    target: bpy.types.Object,
    patch: bpy.types.Object,
    settings: dict,
) -> dict:
    """Settle the rolled target wheel on the fixed base microterrain mesh."""
    contact = settings.get("vertical_contact", {})
    if not bool(contact.get("enabled", False)):
        return {"ok": True, "enabled": False, "vertical_translation_m": 0.0}
    if patch is None or patch.type != "MESH":
        raise RuntimeError("A mesh terrain patch is required for vertical wheel contact")
    root = bpy.data.objects.get(settings["rover_root_object"])
    if root is None:
        raise RuntimeError("Rover root required by vertical contact is missing")

    center_local = sum((Vector(corner) for corner in target.bound_box), Vector()) / 8.0
    radial = [
        math.hypot(float(vertex.co.y - center_local.y), float(vertex.co.z - center_local.z))
        for vertex in target.data.vertices
    ]
    if not radial:
        raise RuntimeError(f"Target wheel {target.name} has no vertices")
    outer_fraction = float(contact["outer_radius_fraction"])
    outer_threshold = max(radial) * outer_fraction
    outer_points = [
        target.matrix_world @ vertex.co
        for vertex, radius in zip(target.data.vertices, radial)
        if radius >= outer_threshold
    ]
    maximum_samples = int(contact["maximum_contact_samples"])
    sample_points = sorted(outer_points, key=lambda point: (float(point.z), float(point.x), float(point.y)))[:maximum_samples]
    if len(sample_points) < int(contact["minimum_contact_samples"]):
        raise RuntimeError(f"Insufficient outer contact samples for {target.name}")

    patch_top = max(float((patch.matrix_world @ Vector(corner)).z) for corner in patch.bound_box)
    inverse = patch.matrix_world.inverted()
    direction_local = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    clearances = []
    for point in sample_points:
        origin_world = Vector((float(point.x), float(point.y), patch_top + 1.0))
        hit, location, _normal, _face = patch.ray_cast(
            inverse @ origin_world,
            direction_local,
            distance=5.0,
        )
        if hit:
            terrain_world = patch.matrix_world @ location
            clearances.append(float(point.z - terrain_world.z))
    if len(clearances) < int(contact["minimum_contact_samples"]):
        raise RuntimeError(f"Terrain contact rays missed the patch for {target.name}")

    before = min(clearances)
    desired = -float(contact["target_penetration_m"])
    vertical_translation = desired - before
    root.matrix_world = Matrix.Translation((0.0, 0.0, vertical_translation)) @ root.matrix_world
    bpy.context.view_layer.update()
    after = before + vertical_translation
    maximum_penetration = float(contact["maximum_penetration_m"])
    maximum_gap = float(contact["maximum_gap_m"])
    return {
        "ok": bool(-maximum_penetration <= after <= maximum_gap),
        "enabled": True,
        "target_wheel": target.name,
        "outer_radius_fraction": outer_fraction,
        "outer_candidate_count": len(outer_points),
        "sampled_contact_count": len(clearances),
        "minimum_clearance_before_m": before,
        "vertical_translation_m": vertical_translation,
        "minimum_clearance_after_m": after,
        "target_penetration_m": float(contact["target_penetration_m"]),
        "maximum_penetration_m": maximum_penetration,
        "maximum_gap_m": maximum_gap,
    }


def _terrain_footprint(scene, camera, patch: bpy.types.Object, minimum_margin: float) -> dict:
    bounds = _patch_bounds(patch)
    origin = camera.matrix_world.translation
    rotation = camera.matrix_world.to_3x3()
    intersections = []
    for corner in camera.data.view_frame(scene=scene):
        direction = (rotation @ corner).normalized()
        if direction.z >= -1e-8:
            return {"ok": False, "reason": "frustum_corner_above_horizon", "bounds": bounds, "corners_world_xy": []}
        distance = (float(bounds["plane_z"]) - float(origin.z)) / float(direction.z)
        if distance <= 0.0:
            return {"ok": False, "reason": "terrain_plane_behind_camera", "bounds": bounds, "corners_world_xy": []}
        point = origin + direction * distance
        intersections.append(point)
    margins = [
        min(
            float(point.x) - bounds["x_min"],
            bounds["x_max"] - float(point.x),
            float(point.y) - bounds["y_min"],
            bounds["y_max"] - float(point.y),
        )
        for point in intersections
    ]
    minimum = min(margins)
    return {
        "ok": minimum >= float(minimum_margin),
        "reason": None if minimum >= float(minimum_margin) else "insufficient_patch_edge_margin",
        "minimum_edge_margin_m": float(minimum),
        "required_edge_margin_m": float(minimum_margin),
        "bounds": bounds,
        "corners_world_xy": [[float(point.x), float(point.y)] for point in intersections],
    }


def _framing_gate(scene, camera, wheel, basis: dict, pose: dict, settings: dict) -> dict:
    from scripts.blender.render_wheel_camera_pose_pilot import _projected_bounds

    projection = _projected_bounds(scene, camera, wheel)
    if pose["crop_policy"] == "full_wheel":
        margin = float(settings["full_wheel_frame_margin_fraction"])
        minimum_area, maximum_area = map(float, settings["full_wheel_bbox_area_fraction"])
        box = projection["normalized_bbox"]
        ok = box[0] >= margin and box[1] >= margin and box[2] <= 1.0 - margin and box[3] <= 1.0 - margin
        ok = ok and minimum_area <= float(projection["bbox_area_fraction"]) <= maximum_area
        return {**projection, "ok": bool(ok), "policy": "full_wheel", "required_margin_fraction": margin}
    box = projection["normalized_bbox"]
    visible_width = max(0.0, min(1.0, box[2]) - max(0.0, box[0]))
    visible_height = max(0.0, min(1.0, box[3]) - max(0.0, box[1]))
    visible_area = visible_width * visible_height
    target = world_to_camera_view(scene, camera, basis["target"])
    margin = float(settings["detail_target_frame_margin_fraction"])
    target_ok = margin <= target.x <= 1.0 - margin and margin <= target.y <= 1.0 - margin and target.z > 0.0
    ok = visible_area >= float(settings["detail_minimum_visible_bbox_area_fraction"]) and target_ok
    return {
        **projection,
        "ok": bool(ok),
        "policy": "intentional_detail_crop",
        "visible_bbox_area_fraction": float(visible_area),
        "target_normalized": [float(target.x), float(target.y), float(target.z)],
    }


def render_preview(source_blend, config_path, plan_path, lighting_config_path, wear_config_path, pose_config_path, sampling_config_path, output_dir):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender import render_wheel_surface_wear_pilot as wear
    from scripts.blender.microterrain.common import render
    from scripts.blender.render_wheel_camera_pose_pilot import _apply_terrain_color_grade
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose
    from src.wheel_preparation.domain_randomization import sample_domain_randomization

    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    lighting_config = json.loads(lighting_config_path.read_text(encoding="utf-8"))
    wear_config = json.loads(wear_config_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_config_path.read_text(encoding="utf-8"))
    sampling_config = json.loads(sampling_config_path.read_text(encoding="utf-8"))
    lighting_by_id = {entry["id"]: entry for entry in lighting_config["presets"]}
    pose_by_id = {entry["id"]: entry for entry in pose_config["poses"]}
    wear_by_id = {entry["id"]: entry for entry in wear_config["variants"]}
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for planned_sample in plan["samples"]:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
        scene = bpy.context.scene
        if not bool(scene.get("wheel_pose_sampling_ready", False)):
            raise RuntimeError("Domain-randomization source is not pose-sampling ready")
        resolution = tuple(map(int, config["preview"]["resolution"]))
        scene.render.resolution_x, scene.render.resolution_y = resolution
        scene.render.resolution_percentage = 100
        wheels = [bpy.data.objects[name] for name in sampling_config["wheels"]]
        base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
        target = bpy.data.objects[planned_sample["target_wheel"]]
        root = bpy.data.objects[config["gates"]["terrain_alignment"]["rover_root_object"]]
        base_root_matrix = root.matrix_world.copy()
        rover_translation = _align_rover_to_patch(
            scene, target, config["gates"]["terrain_alignment"], base_root_matrix
        )
        apply_shared_roll(wheels, base_local, float(planned_sample["healthy_roll_degrees"]))
        pose = pose_by_id[planned_sample["camera_pose"]]
        patch = bpy.data.objects.get(config["gates"]["terrain_patch_object"])
        if patch is None or patch.type != "MESH":
            raise RuntimeError("Microterrain patch required by the jitter gate is missing")
        contact = _settle_rover_on_patch(scene, target, patch, config["gates"]["terrain_alignment"])
        rover_translation = root.matrix_world.translation - base_root_matrix.translation
        if contact["ok"] is not True:
            raise RuntimeError(f"Wheel/terrain contact gate failed: {contact}")
        rejections = []
        maximum_attempts = int(config["gates"]["maximum_camera_resample_attempts"])
        for attempt in range(maximum_attempts):
            candidate = sample_domain_randomization(config, int(planned_sample["sample_index"]), attempt=attempt)
            for field in ("lighting_preset", "surface_wear", "camera_pose", "target_wheel", "healthy_roll_degrees", "wear_seed", "lighting_jitter"):
                if candidate[field] != planned_sample[field]:
                    raise RuntimeError(f"Camera retry changed locked field {field}")
            for field in ("wear_mask", "wear_mask_sha256", "wear_mask_coverage"):
                candidate[field] = planned_sample[field]
            camera = lighting._configure_camera(pose_config["camera"])
            camera.data.lens = float(pose_config["camera"]["focal_length_mm"]) * float(candidate["camera_jitter"]["focal_length_scale"])
            basis = camera_pose(camera, target, pose, pose_config["camera"])
            aim_target = _jitter_camera(camera, basis, candidate)
            scene.camera = camera
            footprint = _terrain_footprint(scene, camera, patch, float(config["gates"]["terrain_minimum_edge_margin_m"]))
            framing = _framing_gate(scene, camera, target, basis, pose, config["gates"])
            if footprint["ok"] and framing["ok"]:
                sample = candidate
                break
            rejections.append(
                {
                    "attempt": attempt,
                    "camera_attempt_seed": candidate["camera_attempt_seed"],
                    "terrain_reason": footprint["reason"],
                    "terrain_minimum_edge_margin_m": footprint.get("minimum_edge_margin_m"),
                    "framing_bbox": framing["normalized_bbox"],
                    "framing_bbox_area_fraction": framing["bbox_area_fraction"],
                }
            )
        else:
            raise RuntimeError(
                f"Jitter gate exhausted {maximum_attempts} attempts for {planned_sample['sample_id']}: "
                f"last_footprint={json.dumps(footprint, sort_keys=True)} "
                f"last_framing={json.dumps(framing, sort_keys=True)}"
            )

        wear_variant = wear_by_id[sample["surface_wear"]]
        target_meshes = wear._target_meshes(target)
        geometry = wear._geometry_signature(target_meshes)
        dimensions = wear._wheel_dimensions(target, target_meshes)
        cloned_materials = []
        affected_objects = []
        if bool(wear_variant["wear_enabled"]):
            cloned_materials, affected_objects = wear._clone_target_materials(target_meshes, sample["sample_id"])
        preset = _jittered_preset(lighting_by_id[sample["lighting_preset"]], sample)
        lighting._configure_sun_world(preset)
        graded = _apply_terrain_color_grade(lighting_config["terrain_grade"])
        dusty_materials = lighting._apply_dust_layer(preset["dust"]) if bool(preset["dust_layer"]) else []
        wear_materials = []
        if bool(wear_variant["wear_enabled"]):
            mask_path = Path(sample["wear_mask"]).resolve()
            wear_materials = wear._apply_wear(cloned_materials, target, dimensions, mask_path, wear_config["shader"])
        scene.view_settings.look = preset["look"]
        scene.view_settings.exposure = float(preset["exposure_ev"])
        lighting._configure_compositor(scene, preset)
        bpy.context.view_layer.update()
        image_path = output_dir / f"{sample['sample_id']}.png"
        render(scene, camera, image_path, resolution)
        sun = bpy.data.objects["GaleNominalSun"]
        background = scene.world.node_tree.nodes["Background"]
        results.append(
            {
                **sample,
                "image": str(image_path),
                "camera_location_m": list(map(float, camera.location)),
                "camera_matrix_world": _matrix_values(camera.matrix_world),
                "aim_target_m": list(map(float, aim_target)),
                "focal_length_mm": float(camera.data.lens),
                "wheel_matrix_world": _matrix_values(target.matrix_world),
                "wheel_geometry": geometry,
                "framing": framing,
                "terrain_footprint": footprint,
                "resolved_lighting": {
                    "sun_energy": float(sun.data.energy),
                    "sun_angle_degrees": math.degrees(float(sun.data.angle)),
                    "sun_color": list(map(float, sun.data.color)),
                    "world_strength": float(background.inputs["Strength"].default_value),
                    "exposure_ev": float(scene.view_settings.exposure),
                },
                "affected_wear_objects": affected_objects,
                "wear_materials": wear_materials,
                "dusty_materials": dusty_materials,
                "terrain_materials_graded": graded,
                "rover_patch_alignment_translation_m": list(map(float, rover_translation)),
                "terrain_contact": contact,
                "camera_gate_rejections": rejections,
                "gates": {"framing": True, "terrain_coverage": True, "terrain_contact": True},
            }
        )

    report = {
        "schema_version": 1,
        "source_blend": str(source_blend),
        "source_modified": False,
        "samples": results,
        "validation": {
            "ok": all(all(entry["gates"].values()) for entry in results),
            "sample_count": len(results),
            "all_camera_jitter_terrain_gates_passed": True,
            "all_wheel_framing_gates_passed": True,
            "pair_lock_contract_present": True,
            "future_anomaly_visibility_and_contrast_gates_still_required": True,
        },
    }
    report_path = output_dir / "domain_randomization_preview.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    render_preview(*_arguments())
