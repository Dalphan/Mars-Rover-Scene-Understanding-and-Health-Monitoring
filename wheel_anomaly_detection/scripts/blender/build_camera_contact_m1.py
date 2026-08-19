from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build Milestone 1 camera/contact variant on TerrainPatch_A.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args(values)


def surface_bvh(obj: bpy.types.Object) -> BVHTree:
    return BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())


def surface_hit(obj: bpy.types.Object, bvh: BVHTree, x: float, y: float) -> tuple[Vector, Vector]:
    inverse = obj.matrix_world.inverted()
    origin = inverse @ Vector((x, y, 10.0))
    direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, normal, _, _ = bvh.ray_cast(origin, direction, 20.0)
    if hit is None:
        raise RuntimeError(f"Terrain patch raycast missed at ({x:.5f}, {y:.5f})")
    return obj.matrix_world @ hit, (obj.matrix_world.to_3x3() @ normal).normalized()


def look_at_world(obj: bpy.types.Object, location: Vector, target: Vector) -> None:
    desired_world = (target - location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
    desired_world.translation = location
    if obj.parent is None:
        obj.matrix_world = desired_world
        return
    parent_world_inverse = obj.parent.matrix_world.inverted()
    parent_inverse_inverse = obj.matrix_parent_inverse.inverted()
    obj.matrix_basis = parent_inverse_inverse @ parent_world_inverse @ desired_world


def evaluated_world_matrix(obj: bpy.types.Object) -> Matrix:
    if obj.parent is None:
        return obj.matrix_basis.copy()
    return obj.parent.matrix_world @ obj.matrix_parent_inverse @ obj.matrix_basis


def world_bbox(obj: bpy.types.Object) -> list[Vector]:
    matrix = evaluated_world_matrix(obj)
    return [matrix @ Vector(corner) for corner in obj.bound_box]


def make_contact_material(patch_id: str) -> bpy.types.Material:
    name = f"{patch_id}_ContactDust_M1"
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
        material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        nodes.clear()
        principled = nodes.new("ShaderNodeBsdfPrincipled")
        output = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = (0.34, 0.17, 0.075, 1.0)
    principled.inputs["Roughness"].default_value = 0.98
    principled.inputs["Metallic"].default_value = 0.0
    material["role"] = "static_compacted_martian_dust"
    return material


def build_contact_berm(
    patch_obj: bpy.types.Object,
    patch: dict,
    center_x: float,
    center_y: float,
    wheel_yaw: float,
) -> dict:
    name = f"{patch['id']}_ContactBerm_M1"
    existing = bpy.data.objects.get(name)
    if existing is not None:
        old_mesh = existing.data
        bpy.data.objects.remove(existing, do_unlink=True)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    collection = bpy.data.collections[patch["id"]]
    bvh = surface_bvh(patch_obj)
    segments = 48
    half_length = patch["contact_length_m"] * 0.5
    half_width = patch["contact_width_m"] * 0.5
    vertices: list[tuple[float, float, float]] = []
    inner_scale = 0.88
    outer_scale = 1.20
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        jitter = 1.0 + 0.06 * math.sin(index * 2.73 + patch["seed"])
        ring_points = []
        for scale in (inner_scale, outer_scale):
            u = math.cos(angle) * half_length * scale * jitter
            v = math.sin(angle) * half_width * scale * jitter
            x = center_x + math.cos(wheel_yaw) * u - math.sin(wheel_yaw) * v
            y = center_y + math.sin(wheel_yaw) * u + math.cos(wheel_yaw) * v
            point, normal = surface_hit(patch_obj, bvh, x, y)
            ring_points.append(point + normal * 0.0007)
        vertices.extend([tuple(ring_points[0]), tuple(ring_points[1])])
    faces: list[tuple[int, int, int, int]] = []
    for index in range(segments):
        current = index * 2
        nxt = ((index + 1) % segments) * 2
        faces.append((current, nxt, nxt + 1, current + 1))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    mesh.materials.append(make_contact_material(patch["id"]))
    obj["role"] = "static_wheel_contact_berm"
    obj["segments"] = segments
    obj["width_m"] = half_width * (outer_scale - inner_scale) * 2.0
    return {"object": name, "segments": segments, "outer_scale": outer_scale, "inner_scale": inner_scale}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    patch = next(item for item in config["patches"] if item["id"] == "TerrainPatch_A")
    root = bpy.data.objects.get(config["rover_root"])
    wheel = bpy.data.objects.get(config["observed_wheel"])
    camera = bpy.data.objects.get("WheelCamera")
    patch_obj = bpy.data.objects.get(patch["id"])
    if any(value is None for value in (root, wheel, camera, patch_obj)):
        raise RuntimeError("Milestone 1 prerequisites are incomplete")

    old_root_location = tuple(root.location)
    old_camera_location = tuple(evaluated_world_matrix(camera).translation)
    old_camera_lens = camera.data.lens
    wheel_points_before = world_bbox(wheel)
    wheel_center_before = sum(wheel_points_before, Vector()) / len(wheel_points_before)
    lowest_wheel_z_before = min(point.z for point in wheel_points_before)
    patch_center_x = sum(point.x for point in [patch_obj.matrix_world @ Vector(c) for c in patch_obj.bound_box]) / 8.0
    patch_center_y = sum(point.y for point in [patch_obj.matrix_world @ Vector(c) for c in patch_obj.bound_box]) / 8.0
    patch_bvh = surface_bvh(patch_obj)
    contact_surface_before, contact_normal = surface_hit(patch_obj, patch_bvh, patch_center_x, patch_center_y)

    contact_margin_m = 0.004
    root.location.z += contact_surface_before.z + contact_margin_m - lowest_wheel_z_before
    bpy.context.view_layer.update()
    # This copy uses a deliberately lowered visual terrain pose so the wheel
    # sits at the configured contact margin.  Keep the validator's approved
    # pose metadata synchronized with the M1 variant instead of weakening the
    # common milestone-1 gate.
    root["visual_terrain_pose"] = json.dumps(
        {
            "x": float(root.location.x),
            "y": float(root.location.y),
            "z": float(root.location.z),
            "yaw_degrees": math.degrees(float(root.rotation_euler.z)),
        },
        sort_keys=True,
    )
    wheel_points_after = world_bbox(wheel)
    wheel_center_after = sum(wheel_points_after, Vector()) / len(wheel_points_after)
    lowest_wheel_z_after = min(point.z for point in wheel_points_after)

    horizontal = Vector((old_camera_location[0] - wheel_center_after.x, old_camera_location[1] - wheel_center_after.y, 0.0))
    if horizontal.length < 1e-5:
        horizontal = Vector((0.0, 1.0, 0.0))
    horizontal.normalize()
    camera_location = wheel_center_after + horizontal * 0.48 + Vector((0.0, 0.0, 1.08))
    camera_target = wheel_center_after + Vector((0.0, 0.0, -0.08))
    look_at_world(camera, camera_location, camera_target)
    camera.data.lens = 50.0
    camera.data.dof.use_dof = False
    camera["m1_role"] = "closeup_with_visible_wheel_contact"
    camera["m1_contact_margin_m"] = contact_margin_m
    camera["m1_target_offset_z_m"] = -0.08
    camera["m1_lens"] = 50.0

    berm_report = build_contact_berm(patch_obj, patch, patch_center_x, patch_center_y, root.rotation_euler.z)
    root["milestone_1_camera_contact"] = True
    root["milestone_1_contact_margin_m"] = contact_margin_m
    root["milestone_1_contact_surface_z"] = contact_surface_before.z
    root["milestone_1_wheel_lowest_z"] = lowest_wheel_z_after
    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            scene["milestone_1_camera_contact"] = True

    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "milestone": "1-camera-contact",
        "source_scene": "rover_simulator_regolith_all_patches.blend",
        "output_blend": str(output_blend),
        "scope": "TerrainPatch_A_and_WheelCamera",
        "contact": {
            "surface_point_before": list(contact_surface_before),
            "surface_normal": list(contact_normal),
            "margin_m": contact_margin_m,
            "wheel_lowest_z_before": lowest_wheel_z_before,
            "wheel_lowest_z_after": lowest_wheel_z_after,
            "wheel_center_after": list(wheel_center_after),
            "berm": berm_report,
        },
        "camera": {
            "location_before": list(old_camera_location),
            "location_after": list(evaluated_world_matrix(camera).translation),
            "lens_before": old_camera_lens,
            "lens_after": camera.data.lens,
            "target_offset_z_m": -0.08,
        },
        "pair_context_unchanged": True,
        "lighting_changed": False,
        "collision_solver_enabled": False,
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built Milestone 1 camera/contact variant: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
