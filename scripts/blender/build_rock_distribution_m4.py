from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_rock_library_m3 import (  # noqa: E402
    FAMILIES,
    append_ringed_rock,
    build_mesh,
    patch_center,
    surface_bvh,
    surface_hit,
)


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build Milestone 4 clustered terrain distribution.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-label", default="rover_simulator_m3_rock_library.blend")
    parser.add_argument("--patches", default="A,B,C", help="Comma-separated patch suffixes to process")
    return parser.parse_args(values)


def inside_patch(x: float, y: float, cx: float, cy: float, size: float, margin: float = 0.04) -> bool:
    half = size * 0.5 - margin
    return cx - half <= x <= cx + half and cy - half <= y <= cy + half


def in_exclusion(x: float, y: float, cx: float, cy: float, scale: float = 1.0) -> bool:
    return (((x - cx) / 0.38) ** 2 + ((y - cy) / 0.20) ** 2) < scale


def far_enough(x: float, y: float, points: list[tuple[float, float]], min_distance: float) -> bool:
    limit_sq = min_distance * min_distance
    return all((x - ox) ** 2 + (y - oy) ** 2 >= limit_sq for ox, oy in points)


def cluster_centers(cx: float, cy: float, size: float, count: int, min_distance: float,
                    seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    centers: list[tuple[float, float]] = []
    for _ in range(max(200, count * 120)):
        x = cx + rng.uniform(-size * 0.45, size * 0.45)
        y = cy + rng.uniform(-size * 0.45, size * 0.45)
        if in_exclusion(x, y, cx, cy, 1.15):
            continue
        if far_enough(x, y, centers, min_distance):
            centers.append((x, y))
            if len(centers) >= count:
                return centers
    # A deterministic fallback still leaves the contact ellipse open.
    fallback = [
        (-0.95, -0.88), (0.86, -0.82), (-0.98, 0.82), (0.92, 0.76),
        (0.05, -1.12), (-0.12, 1.10),
    ]
    for ox, oy in fallback:
        point = (cx + ox, cy + oy)
        if inside_patch(*point, cx, cy, size) and not in_exclusion(*point, cx, cy, 1.15):
            if far_enough(*point, centers, min_distance):
                centers.append(point)
        if len(centers) >= count:
            break
    return centers[:count]


def allocate_counts(total: int, cluster_count: int, rng: random.Random) -> list[int]:
    if cluster_count <= 0:
        return []
    weights = [rng.uniform(0.72, 1.35) for _ in range(cluster_count)]
    counts = [1 for _ in range(cluster_count)]
    remaining = max(0, total - cluster_count)
    for index in range(remaining):
        weighted = [(weights[i] / (counts[i] + 0.8), i) for i in range(cluster_count)]
        counts[max(weighted)[1]] += 1
    return counts


def clustered_points(cx: float, cy: float, size: float, total: int, min_distance: float,
                     cluster_count: int, cluster_radius: float, background_fraction: float,
                     exclusion_scale: float, seed: int, max_attempts_per_point: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[int]]:
    rng = random.Random(seed)
    centers = cluster_centers(cx, cy, size, cluster_count, max(cluster_radius * 1.45, min_distance * 2.2), seed + 17)
    if not centers:
        return [], [], []
    cluster_total = max(0, min(total, int(round(total * (1.0 - background_fraction)))))
    background_total = total - cluster_total
    allocations = allocate_counts(cluster_total, len(centers), rng)
    points: list[tuple[float, float]] = []
    point_cluster_ids: list[int] = []
    for cluster_id, amount in enumerate(allocations):
        center_x, center_y = centers[cluster_id]
        attempts = 0
        while amount > 0 and attempts < amount * max_attempts_per_point:
            attempts += 1
            angle = rng.uniform(0.0, 2.0 * math.pi)
            radial = abs(rng.gauss(0.0, cluster_radius * 0.42))
            x = center_x + math.cos(angle) * radial
            y = center_y + math.sin(angle) * radial * rng.uniform(0.72, 1.18)
            if not inside_patch(x, y, cx, cy, size) or in_exclusion(x, y, cx, cy, exclusion_scale):
                continue
            if not far_enough(x, y, points, min_distance):
                continue
            points.append((x, y))
            point_cluster_ids.append(cluster_id)
            amount -= 1
        if amount:
            # Fill a failed cluster deterministically from the remaining area.
            for _ in range(amount):
                for _attempt in range(max_attempts_per_point * 2):
                    x = center_x + rng.uniform(-cluster_radius, cluster_radius)
                    y = center_y + rng.uniform(-cluster_radius, cluster_radius)
                    if inside_patch(x, y, cx, cy, size) and not in_exclusion(x, y, cx, cy, exclusion_scale) and far_enough(x, y, points, min_distance):
                        points.append((x, y))
                        point_cluster_ids.append(cluster_id)
                        break

    background: list[tuple[float, float]] = []
    attempts = 0
    while len(background) < background_total and attempts < background_total * max_attempts_per_point * 3:
        attempts += 1
        x = cx + rng.uniform(-size * 0.47, size * 0.47)
        y = cy + rng.uniform(-size * 0.47, size * 0.47)
        if not inside_patch(x, y, cx, cy, size) or in_exclusion(x, y, cx, cy, exclusion_scale):
            continue
        if not far_enough(x, y, points + background, min_distance):
            continue
        background.append((x, y))
        points.append((x, y))
        point_cluster_ids.append(-1)
    # The exclusion ellipse and patch borders can leave a cluster short on
    # dense patches. Fill the exact requested count with a relaxed deterministic
    # scatter rather than silently changing the rock count.
    fill_attempts = 0
    relaxed_distance = min_distance * 0.82
    while len(points) < total and fill_attempts < total * max_attempts_per_point * 4:
        fill_attempts += 1
        x = cx + rng.uniform(-size * 0.47, size * 0.47)
        y = cy + rng.uniform(-size * 0.47, size * 0.47)
        if not inside_patch(x, y, cx, cy, size) or in_exclusion(x, y, cx, cy, exclusion_scale):
            continue
        if not far_enough(x, y, points, relaxed_distance):
            continue
        points.append((x, y))
        point_cluster_ids.append(-1)
    return points, centers, point_cluster_ids


def replace_rocks(rocks: bpy.types.Object, patch_obj: bpy.types.Object, patch: dict,
                  library_settings: dict, distribution: dict) -> dict:
    cx, cy = patch_center(patch_obj)
    pilot = patch["regolith_pilot"]
    total = pilot["rock_family_count"]
    points, centers, cluster_ids = clustered_points(
        cx, cy, patch["size_m"], total, pilot["rock_min_distance_m"],
        int(distribution["cluster_counts"][patch["id"]]),
        float(distribution["cluster_radius_m"]), float(distribution["background_fraction"]),
        1.0, patch["seed"] + 4901, int(distribution["max_attempts_per_point"]),
    )
    if len(points) != total:
        raise RuntimeError(f"Cluster sampler produced {len(points)} of {total} rocks for {patch['id']}")
    bvh = surface_bvh(patch_obj)
    rng = random.Random(patch["seed"] + 5901)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    smooth_flags: list[bool] = []
    face_families: list[int] = []
    for index, (x, y) in enumerate(points):
        point, normal = surface_hit(patch_obj, bvh, x, y)
        radius = rng.uniform(patch["rock_radius_min_m"], patch["rock_radius_max_m"] * pilot.get("rock_max_scale", 0.82))
        if rng.random() < patch["large_rock_fraction"]:
            radius *= rng.uniform(1.15, 1.45)
        before = len(face_families)
        append_ringed_rock(vertices, faces, smooth_flags, face_families, point, normal, radius,
                           FAMILIES[index % len(FAMILIES)], rng, int(library_settings["segments"]))
        # append_ringed_rock adds one material id per face; keep the cluster id
        # as a custom report value rather than changing the family material.
        if len(face_families) == before:
            raise RuntimeError("M3 rock library emitted no faces")
    old_mesh = rocks.data
    rocks.data = build_mesh(f"{patch['id']}_Rocks_M4_Clustered_Mesh", vertices, faces, smooth_flags)
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
    bevel.width = float(library_settings["bevel_width_m"])
    bevel.segments = int(library_settings["bevel_segments"])
    bevel.limit_method = "ANGLE"
    displacement = rocks.modifiers.new(f"{patch['id']}_M3_MicroDisplace", "DISPLACE")
    texture_name = f"{patch['id']}_M3_RockMicroNoise"
    displacement.texture = bpy.data.textures.get(texture_name)
    displacement.texture_coords = "GLOBAL"
    displacement.strength = float(library_settings["micro_displace_m"])
    displacement.mid_level = 0.5
    rocks["detail_role"] = "rock_library_m4_clustered"
    rocks["library_version"] = "M3"
    rocks["distribution_version"] = "M4"
    rocks["cluster_count"] = len(centers)
    rocks["cluster_seed"] = patch["seed"] + 4901
    rocks["background_fraction"] = float(distribution["background_fraction"])
    return {
        "patch_id": patch["id"],
        "rock_count": len(points),
        "vertex_count": len(vertices),
        "polygon_count": len(faces),
        "cluster_count": len(centers),
        "cluster_centers": [[round(x, 6), round(y, 6)] for x, y in centers],
        "cluster_members": {str(index): cluster_ids.count(index) for index in range(len(centers))},
        "background_count": cluster_ids.count(-1),
        "cluster_seed": patch["seed"] + 4901,
        "distribution_preserved": False,
    }


def connected_components(mesh: bpy.types.Mesh) -> list[list[int]]:
    adjacency = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = set()
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append(component)
    return components


def reposition_gravel(gravel: bpy.types.Object, patch_obj: bpy.types.Object, patch: dict,
                      distribution: dict) -> dict:
    cx, cy = patch_center(patch_obj)
    mesh = gravel.data
    components = connected_components(mesh)
    components = [component for component in components if component]
    components.sort(key=lambda indices: (sum(mesh.vertices[i].co.x for i in indices) / len(indices),
                                         sum(mesh.vertices[i].co.y for i in indices) / len(indices)))
    points, centers, cluster_ids = clustered_points(
        cx, cy, patch["size_m"], len(components), 0.009,
        int(distribution["cluster_counts"][patch["id"]]),
        float(distribution["gravel_cluster_radius_m"]), float(distribution["gravel_background_fraction"]),
        0.82, patch["seed"] + 6901, int(distribution["max_attempts_per_point"]),
    )
    if len(points) != len(components):
        raise RuntimeError(f"Cluster sampler produced {len(points)} of {len(components)} gravel islands for {patch['id']}")
    bvh = surface_bvh(patch_obj)
    for component, (x, y) in zip(components, points):
        old_center = sum((mesh.vertices[index].co for index in component), Vector()) / len(component)
        target, _ = surface_hit(patch_obj, bvh, x, y)
        delta = target - (gravel.matrix_world @ old_center)
        local_delta = gravel.matrix_world.inverted().to_3x3() @ delta
        for index in component:
            mesh.vertices[index].co += local_delta
    mesh.update()
    gravel["detail_role"] = "poisson_gravel_clustered"
    gravel["distribution_version"] = "M4"
    gravel["cluster_count"] = len(centers)
    gravel["cluster_seed"] = patch["seed"] + 6901
    return {
        "island_count": len(components),
        "cluster_count": len(centers),
        "cluster_centers": [[round(x, 6), round(y, 6)] for x, y in centers],
        "background_count": cluster_ids.count(-1),
        "cluster_seed": patch["seed"] + 6901,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    library_settings = dict(config["rock_library_m3"])
    distribution = dict(config["rock_distribution_m4"])
    requested = {f"TerrainPatch_{suffix.strip().upper()}" for suffix in args.patches.split(",") if suffix.strip()}
    patches = [item for item in config["patches"] if item["id"] in requested]
    if not patches:
        raise RuntimeError(f"No requested patches found: {sorted(requested)}")
    reports = []
    for patch in patches:
        patch_obj = bpy.data.objects.get(patch["id"])
        rocks = bpy.data.objects.get(f"{patch['id']}_Rocks")
        gravel = bpy.data.objects.get(f"{patch['id']}_Gravel")
        if patch_obj is None or rocks is None or gravel is None:
            raise RuntimeError(f"Missing patch, rocks, or gravel for {patch['id']}")
        rock_report = replace_rocks(rocks, patch_obj, patch, library_settings, distribution)
        gravel_report = reposition_gravel(gravel, patch_obj, patch, distribution)
        patch_obj["milestone_4_clustered_distribution"] = True
        reports.append({"patch_id": patch["id"], "rocks": rock_report, "gravel": gravel_report})

    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            scene["terrain_detail_milestone"] = "4-clustered-distribution"
            scene["milestone_4_clustered_distribution"] = True
            scene["collision_enabled"] = False
    root = bpy.data.objects.get(config["rover_root"])
    if root is not None:
        root["milestone_4_clustered_distribution"] = True
        root["milestone_4_source_milestone"] = "3-rock-library"

    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "milestone": "4-clustered-distribution",
        "source_scene": args.source_label,
        "output_blend": str(output_blend),
        "scope": [entry["patch_id"] for entry in reports],
        "settings": distribution,
        "patches": reports,
        "pair_context_unchanged": True,
        "camera_changed": False,
        "lighting_changed": False,
        "collision_enabled": False,
        "external_resources": [],
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built Milestone 4 clustered distribution: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
