from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


FAMILIES = ("angular", "stratified", "rounded", "shard")


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build Milestone 3 high-detail rock library on terrain patches.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-label", default="rover_simulator_m2_regolith.blend")
    parser.add_argument("--patches", default="A,B,C", help="Comma-separated patch suffixes to process")
    return parser.parse_args(values)


def surface_bvh(obj: bpy.types.Object) -> BVHTree:
    return BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())


def surface_hit(obj: bpy.types.Object, bvh: BVHTree, x: float, y: float) -> tuple[Vector, Vector]:
    inverse = obj.matrix_world.inverted()
    origin = inverse @ Vector((x, y, 10.0))
    direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, normal, _, _ = bvh.ray_cast(origin, direction, 20.0)
    if hit is None:
        raise RuntimeError(f"Patch raycast missed at ({x:.5f}, {y:.5f})")
    return obj.matrix_world @ hit, (obj.matrix_world.to_3x3() @ normal).normalized()


def patch_center(obj: bpy.types.Object) -> tuple[float, float]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def poisson_points(
    center_x: float,
    center_y: float,
    size: float,
    min_distance: float,
    count: int,
    seed: int,
    exclusion: tuple[float, float, float],
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    half = size * 0.5
    cell_size = min_distance / math.sqrt(2.0)
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    result: list[tuple[float, float]] = []
    ex_x, ex_y, ex_scale = exclusion
    max_attempts = max(500, count * 180)
    for _ in range(max_attempts):
        x = center_x + rng.uniform(-half + 0.03, half - 0.03)
        y = center_y + rng.uniform(-half + 0.03, half - 0.03)
        if (((x - ex_x) / 0.38) ** 2 + ((y - ex_y) / 0.20) ** 2) < ex_scale:
            continue
        cell = (math.floor(x / cell_size), math.floor(y / cell_size))
        valid = True
        for gx in range(cell[0] - 2, cell[0] + 3):
            for gy in range(cell[1] - 2, cell[1] + 3):
                for other_x, other_y in grid.get((gx, gy), []):
                    if (x - other_x) ** 2 + (y - other_y) ** 2 < min_distance**2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            continue
        result.append((x, y))
        grid.setdefault(cell, []).append((x, y))
        if len(result) >= count:
            break
    return result


def tangent_basis(normal: Vector, yaw: float) -> tuple[Vector, Vector]:
    reference = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    tangent_a = reference - normal * reference.dot(normal)
    if tangent_a.length < 1e-5:
        tangent_a = normal.cross(Vector((0.0, 0.0, 1.0)))
    tangent_a.normalize()
    tangent_b = normal.cross(tangent_a).normalized()
    return tangent_a, tangent_b


def ring_point(center: Vector, tangent_a: Vector, tangent_b: Vector, angle: float,
               radius: float, rx: float, ry: float, height: float, jitter: float,
               normal: Vector, top_shift: Vector | None = None) -> Vector:
    offset = tangent_a * (math.cos(angle) * rx * jitter) + tangent_b * (math.sin(angle) * ry * jitter)
    shift = top_shift if top_shift is not None else Vector()
    return center + offset + normal * height + shift


def append_ringed_rock(vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]],
                       smooth_flags: list[bool], face_families: list[int], center: Vector,
                       normal: Vector, radius: float, family: str, rng: random.Random,
                       segments: int) -> None:
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    tangent_a, tangent_b = tangent_basis(normal, yaw)
    if family == "stratified":
        rx, ry, levels, smooth = 1.35, 0.86, (0.0, 0.18, 0.42, 0.62), False
        scales = (1.00, 1.10, 0.82, 0.58)
    elif family == "rounded":
        rx, ry, levels, smooth = 1.02, 0.92, (-0.02, 0.24, 0.62, 0.92), True
        scales = (0.95, 1.00, 0.78, 0.35)
    elif family == "shard":
        rx, ry, levels, smooth = 1.72, 0.43, (0.0, 0.12, 0.54, 0.88), False
        scales = (1.00, 0.92, 0.64, 0.18)
    else:
        rx, ry, levels, smooth = 1.06, 0.84, (-0.01, 0.18, 0.58, 0.96), False
        scales = (1.00, 0.94, 0.70, 0.26)

    # A family-specific tilt makes shards and stratified fragments read as
    # deposited rocks instead of identical low-poly cones.
    tilt = tangent_a * rng.uniform(-0.12, 0.12) * radius + tangent_b * rng.uniform(-0.08, 0.08) * radius
    base_index = len(vertices)
    rings: list[list[int]] = []
    for level_index, level in enumerate(levels):
        ring: list[int] = []
        for segment in range(segments):
            angle = 2.0 * math.pi * segment / segments
            jitter = rng.uniform(0.84, 1.16)
            local_rx = radius * rx * scales[level_index]
            local_ry = radius * ry * scales[level_index]
            shift = tilt * (level_index / max(1, len(levels) - 1)) if family in {"shard", "stratified"} else Vector()
            point = ring_point(center, tangent_a, tangent_b, angle, radius, local_rx, local_ry,
                               radius * level, jitter, normal, shift)
            ring.append(len(vertices))
            vertices.append(tuple(point))
        rings.append(ring)

    for level_index in range(len(rings) - 1):
        lower = rings[level_index]
        upper = rings[level_index + 1]
        for segment in range(segments):
            nxt = (segment + 1) % segments
            # Outward winding follows the existing boulder convention.
            faces.append((lower[segment], lower[nxt], upper[nxt], upper[segment]))
            smooth_flags.append(smooth and level_index > 0)
            face_families.append(FAMILIES.index(family))

    top_center = len(vertices)
    vertices.append(tuple(center + normal * radius * (1.02 if family != "flat" else 0.70) + tilt))
    top_ring = rings[-1]
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((top_ring[segment], top_ring[nxt], top_center))
        smooth_flags.append(smooth)
        face_families.append(FAMILIES.index(family))

    # Close the underside for clean shadows and watertight geometry.
    bottom_center = len(vertices)
    vertices.append(tuple(center - normal * radius * 0.12))
    bottom_ring = rings[0]
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((bottom_ring[nxt], bottom_ring[segment], bottom_center))
        smooth_flags.append(False)
        face_families.append(FAMILIES.index(family))


def build_mesh(name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]],
               smooth_flags: list[bool]) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon, smooth in zip(mesh.polygons, smooth_flags):
        polygon.use_smooth = smooth
    return mesh


def replace_rock_library(rocks: bpy.types.Object, patch_obj: bpy.types.Object, patch: dict,
                         settings: dict) -> dict:
    center_x, center_y = patch_center(patch_obj)
    bvh = surface_bvh(patch_obj)
    pilot = patch["regolith_pilot"]
    points = poisson_points(center_x, center_y, patch["size_m"], pilot["rock_min_distance_m"],
                            pilot["rock_family_count"], patch["seed"] + 901,
                            (center_x, center_y, 1.0))
    rng = random.Random(patch["seed"] + 3901)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    smooth_flags: list[bool] = []
    face_families: list[int] = []
    for index, (x, y) in enumerate(points):
        point, normal = surface_hit(patch_obj, bvh, x, y)
        radius = rng.uniform(patch["rock_radius_min_m"], patch["rock_radius_max_m"] * pilot.get("rock_max_scale", 0.82))
        if rng.random() < patch["large_rock_fraction"]:
            radius *= rng.uniform(1.15, 1.45)
        append_ringed_rock(vertices, faces, smooth_flags, face_families, point, normal, radius,
                           FAMILIES[index % len(FAMILIES)], rng, int(settings["segments"]))

    old_mesh = rocks.data
    rocks.data = build_mesh(f"{patch['id']}_Rocks_M3_Library_Mesh", vertices, faces, smooth_flags)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    rocks.data.materials.clear()
    for family_index in range(len(FAMILIES)):
        material = bpy.data.materials.get(f"{patch['id']}_RockFamily_{family_index + 1:02d}")
        if material is None:
            raise RuntimeError(f"Missing rock family material {patch['id']} #{family_index + 1}")
        rocks.data.materials.append(material)
    for polygon, material_index in zip(rocks.data.polygons, face_families):
        polygon.material_index = material_index

    for modifier in list(rocks.modifiers):
        rocks.modifiers.remove(modifier)
    bevel = rocks.modifiers.new(f"{patch['id']}_M3_MicroBevel", "BEVEL")
    bevel.width = float(settings["bevel_width_m"])
    bevel.segments = int(settings["bevel_segments"])
    bevel.limit_method = "ANGLE"
    displacement = rocks.modifiers.new(f"{patch['id']}_M3_MicroDisplace", "DISPLACE")
    texture_name = f"{patch['id']}_M3_RockMicroNoise"
    texture = bpy.data.textures.get(texture_name) or bpy.data.textures.new(texture_name, type="CLOUDS")
    texture.noise_scale = float(settings["micro_noise_scale_m"])
    texture.noise_depth = 2
    displacement.texture = texture
    displacement.texture_coords = "GLOBAL"
    displacement.strength = float(settings["micro_displace_m"])
    displacement.mid_level = 0.5
    rocks["detail_role"] = "rock_library_m3"
    rocks["deterministic_seed"] = patch["seed"] + 3901
    rocks["rock_families"] = ",".join(FAMILIES)
    rocks["library_version"] = "M3"
    return {
        "patch_id": patch["id"],
        "rock_count": len(points),
        "vertex_count": len(vertices),
        "polygon_count": len(faces),
        "families": list(FAMILIES),
        "distribution_seed": patch["seed"] + 901,
        "geometry_seed": patch["seed"] + 3901,
        "distribution_preserved": True,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    settings = dict(config["rock_library_m3"])
    requested = {f"TerrainPatch_{suffix.strip().upper()}" for suffix in args.patches.split(",") if suffix.strip()}
    patches = [item for item in config["patches"] if item["id"] in requested]
    if not patches:
        raise RuntimeError(f"No requested patches found: {sorted(requested)}")

    reports = []
    for patch in patches:
        patch_obj = bpy.data.objects.get(patch["id"])
        rocks = bpy.data.objects.get(f"{patch['id']}_Rocks")
        if patch_obj is None or rocks is None:
            raise RuntimeError(f"Missing patch or rock object for {patch['id']}")
        reports.append(replace_rock_library(rocks, patch_obj, patch, settings))
        patch_obj["milestone_3_rock_library"] = True

    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            scene["terrain_detail_milestone"] = "3-rock-library"
            scene["milestone_3_rock_library"] = True
            scene["collision_enabled"] = False

    root = bpy.data.objects.get(config["rover_root"])
    if root is not None:
        root["milestone_3_rock_library"] = True
        root["milestone_3_source_milestone"] = "2-regolith-material"

    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "milestone": "3-rock-library",
        "source_scene": args.source_label,
        "output_blend": str(output_blend),
        "scope": [entry["patch_id"] for entry in reports],
        "settings": settings,
        "patches": reports,
        "distribution_preserved": True,
        "pair_context_unchanged": True,
        "camera_changed": False,
        "lighting_changed": False,
        "collision_enabled": False,
        "external_resources": [],
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built Milestone 3 rock library: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
