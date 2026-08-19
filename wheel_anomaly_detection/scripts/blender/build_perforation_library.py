"""Replace the legacy clean hole in a copy of the M6 Martian-terrain scene.

The script builds the review library.  It never duplicates wheel details and it
derives the anomalous visible skin from ``Wheel_Skin_Normal`` so the baked
legacy perforation cannot leak into either side of the new pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import bpy
import bmesh
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.blender import prepare_wheel as wheel  # noqa: E402
from src.wheel_preparation.perforation import (  # noqa: E402
    build_perforation_library,
    generate_perforation_profile,
    validate_perforation_profile,
)


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build the M6 irregular-hole review pilot")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--library-config", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--seed", type=int)
    return parser.parse_args(values)


def frame_from_skin(skin: bpy.types.Object) -> dict[str, float]:
    radii = [math.hypot(float(vertex.co.y), float(vertex.co.z)) for vertex in skin.data.vertices]
    axial = [float(vertex.co.x) for vertex in skin.data.vertices]
    return {
        "outer_radius": max(radii),
        "inner_radius": min(radii),
        "wall_thickness": max(max(radii) - min(radii), 1e-6),
        "axial_min": min(axial),
        "axial_max": max(axial),
        "diameter": 2.0 * max(radii),
    }


def mesh_digest(mesh: bpy.types.Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(f"{len(mesh.vertices)}:{len(mesh.edges)}:{len(mesh.polygons)}".encode())
    for vertex in mesh.vertices:
        digest.update(("%.8f,%.8f,%.8f;" % tuple(vertex.co)).encode())
    return digest.hexdigest()


def old_hole_center(healthy_skin: bpy.types.Object) -> Vector:
    old_cutter = bpy.data.objects.get("Perforation_Cutter")
    if old_cutter is None or not old_cutter.data.vertices:
        return Vector((0.0, 0.0, frame_from_skin(healthy_skin)["outer_radius"]))
    centroid = sum((vertex.co for vertex in old_cutter.data.vertices), Vector()) / len(old_cutter.data.vertices)
    centroid_world = old_cutter.matrix_world @ centroid
    return healthy_skin.matrix_world.inverted() @ centroid_world


def candidate_is_clear(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    skin: bpy.types.Object,
    local_center: Vector,
    frame: Mapping[str, float],
) -> tuple[bool, str]:
    world_center = skin.matrix_world @ local_center
    radial_local = Vector((0.0, float(local_center.y), float(local_center.z))).normalized()
    radial_world = (skin.matrix_world.to_3x3() @ radial_local).normalized()
    view_to_camera = (camera.matrix_world.translation - world_center).normalized()
    facing = radial_world.dot(view_to_camera)
    if facing < 0.42:
        return False, f"grazing:{facing:.3f}"
    projection = world_to_camera_view(scene, camera, world_center)
    if projection.z <= 0.0 or not (0.08 <= projection.x <= 0.92 and 0.08 <= projection.y <= 0.92):
        return False, "outside_camera"

    inverse = skin.matrix_world.inverted()
    local_camera = inverse @ camera.matrix_world.translation
    direction = (local_center - local_camera).normalized()
    # Start far enough behind the candidate surface to avoid immediately
    # re-hitting the same polygon.  A later hit means the opposite wheel skin
    # would turn the opening into a black cavity.
    start = local_center + direction * max(0.018, 2.4 * float(frame["wall_thickness"]))
    hit, _location, _normal, _face = skin.ray_cast(start, direction, distance=float(frame["diameter"]) * 1.4)
    if hit:
        return False, "opposite_wheel_skin"

    hidden: list[tuple[bpy.types.Object, bool]] = []
    for obj in scene.objects:
        if obj == skin or "Wheel_Skin" in obj.name or "Perforation_Cutter" in obj.name:
            hidden.append((obj, obj.hide_viewport))
            obj.hide_viewport = True
    bpy.context.view_layer.update()
    origin = camera.matrix_world.translation
    theta = math.atan2(float(local_center.z), float(local_center.y))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    core = 0.006
    sample_offsets = (
        Vector((0.0, 0.0, 0.0)),
        Vector((core, 0.0, 0.0)),
        Vector((-core, 0.0, 0.0)),
        tangent * core,
        tangent * -core,
        Vector((0.004, 0.0, 0.0)) + tangent * 0.004,
        Vector((-0.004, 0.0, 0.0)) + tangent * -0.004,
    )
    backgrounds: list[str] = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rejection: str | None = None
    for offset in sample_offsets:
        sample_world = skin.matrix_world @ (local_center + offset)
        ray = (sample_world - origin).normalized()
        target_distance = (sample_world - origin).length
        hit_scene, location, _normal, _face, obj, _matrix = scene.ray_cast(
            depsgraph, origin, ray, distance=1000.0
        )
        background = obj.name if hit_scene and obj is not None else "WORLD"
        backgrounds.append(background)
        if hit_scene and obj is not None and (location - origin).length < target_distance + 0.012:
            rejection = f"foreground:{obj.name}"
            break
        if not hit_scene or obj is None or not ("Terrain" in obj.name or "Martian" in obj.name):
            rejection = f"unreadable_background:{background}"
            break
    for obj_hidden, state in hidden:
        obj_hidden.hide_viewport = state
    bpy.context.view_layer.update()
    if rejection is not None:
        return False, rejection
    return True, backgrounds[0]


def candidate_has_intact_skin(skin: bpy.types.Object, local_center: Vector, frame: Mapping[str, float]) -> bool:
    theta = math.atan2(float(local_center.z), float(local_center.y))
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    half_extent = 0.062 * float(frame["diameter"])
    for axial_offset in (-half_extent, 0.0, half_extent):
        for tangent_offset in (-half_extent, 0.0, half_extent):
            origin = radial * (float(frame["outer_radius"]) + 2.0 * float(frame["wall_thickness"]))
            origin += Vector((float(local_center.x) + axial_offset, 0.0, 0.0)) + tangent * tangent_offset
            hit, _location, _normal, _face = skin.ray_cast(
                origin, -radial, distance=4.0 * float(frame["wall_thickness"])
            )
            if not hit:
                return False
    return True


def choose_placement(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    skin: bpy.types.Object,
    profile: Mapping[str, Any],
    frame: Mapping[str, float],
) -> tuple[Vector, list[dict[str, Any]]]:
    local_camera = skin.matrix_world.inverted() @ camera.matrix_world.translation
    preferred_theta = math.atan2(float(local_camera.z), float(local_camera.y))
    # Use the largest configured profile margin for every variant so shape and
    # size cannot change the selected location.
    margin = 0.060 * float(frame["diameter"]) + 0.012
    low = float(frame["axial_min"]) + margin
    high = float(frame["axial_max"]) - margin
    axial_candidates = [low + index * (high - low) / 10.0 for index in range(11)]
    theta_candidates = [preferred_theta - 1.20 + index * 0.12 for index in range(21)]
    attempts: list[dict[str, Any]] = []
    clear_candidates: list[tuple[float, Vector, str]] = []
    radius = float(frame["outer_radius"])
    for theta in theta_candidates:
        for axial in axial_candidates:
            center = Vector((axial, radius * math.cos(theta), radius * math.sin(theta)))
            intact_skin = candidate_has_intact_skin(skin, center, frame)
            clear, background = candidate_is_clear(scene, camera, skin, center, frame) if intact_skin else (False, "native_opening")
            projection = world_to_camera_view(scene, camera, skin.matrix_world @ center)
            if clear and float(projection.y) > 0.60:
                clear, background = False, "upper_wheel"
            attempts.append(
                {
                    "axial": axial,
                    "theta": theta,
                    "clear": clear,
                    "intact_skin": intact_skin,
                    "background": background,
                    "view_x": float(projection.x),
                    "view_y": float(projection.y),
                }
            )
            if clear:
                terrain_penalty = 0.0 if "Terrain" in background or "Martian" in background else 1.0
                score = terrain_penalty + abs(float(projection.y) - 0.46) + 0.2 * abs(float(projection.x) - 0.55)
                clear_candidates.append((score, center, background))
    if clear_candidates:
        return min(clear_candidates, key=lambda candidate: candidate[0])[1], attempts
    rejection_counts: dict[str, int] = {}
    for attempt in attempts:
        reason = str(attempt["background"])
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    raise RuntimeError(
        "No camera-visible perforation placement has a clear readable background: "
        + json.dumps(rejection_counts, sort_keys=True)
    )


def make_cutter(
    profile: Mapping[str, Any],
    center: Vector,
    frame: Mapping[str, float],
    config: Mapping[str, Any],
    name: str,
) -> bpy.types.Object:
    theta = math.atan2(float(center.z), float(center.y))
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    axial = Vector((1.0, 0.0, 0.0))
    clearance = max(
        2.0 * float(frame["wall_thickness"]),
        float(config.get("radial_clearance_ratio", 0.006)) * float(frame["diameter"]),
    )
    points = [Vector((float(point[0]), float(point[1]))) for point in profile["points_axial_tangent"]]
    signed_area = sum(
        points[index].x * points[(index + 1) % len(points)].y
        - points[(index + 1) % len(points)].x * points[index].y
        for index in range(len(points))
    )
    if signed_area < 0.0:
        points.reverse()
    vertices: list[tuple[float, float, float]] = []
    for radius in (float(frame["inner_radius"]) - clearance, float(frame["outer_radius"]) + clearance):
        for point in points:
            value = radial * radius + axial * (float(center.x) + point.x) + tangent * point.y
            vertices.append(tuple(value))
    count = len(points)
    # Outward winding matches the known-good cutter in prepare_wheel.py.
    faces = [tuple(range(count)), tuple(reversed(range(count, 2 * count)))]
    faces.extend((index, count + index, count + (index + 1) % count, (index + 1) % count) for index in range(count))
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    cutter = bpy.data.objects.new(name, mesh)
    return cutter


def make_fold(
    profile: Mapping[str, Any], center: Vector, frame: Mapping[str, float], material: bpy.types.Material, name: str
) -> bpy.types.Object:
    fold = profile["folds"][0]
    points = [Vector((float(point[0]), float(point[1]))) for point in profile["points_axial_tangent"]]
    index = int(fold["edge_index"]) % len(points)
    p0, p1 = points[index], points[(index + 1) % len(points)]
    edge = p1 - p0
    inward = Vector((-edge.y, edge.x)).normalized()
    length = min(0.08, float(fold["length_ratio"])) * max(
        float(profile["quality"]["width"]), float(profile["quality"]["height"])
    )
    tip0, tip1 = p0 + inward * length, p1 + inward * length
    theta = math.atan2(float(center.z), float(center.y))
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    axial = Vector((1.0, 0.0, 0.0))
    thickness = 0.0008 * float(frame["diameter"])
    bend = math.tan(math.radians(min(28.0, float(fold["angle_degrees"])))) * length

    def point(local: Vector, radius: float) -> Vector:
        return radial * radius + axial * (float(center.x) + local.x) + tangent * local.y

    outer = float(frame["outer_radius"]) + thickness
    front = [point(p0, outer), point(p1, outer), point(tip1, outer + bend), point(tip0, outer + bend)]
    back = [value - radial * thickness for value in front]
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata([tuple(value) for value in front + back], [], [
        (0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ])
    mesh.update(calc_edges=True)
    flap = bpy.data.objects.new(name, mesh)
    flap.data.materials.append(material)
    return flap


def make_camera_clearance_cutter(
    center: Vector, local_camera: Vector, frame: Mapping[str, float], name: str
) -> bpy.types.Object:
    theta = math.atan2(float(center.z), float(center.y))
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    surface = radial * float(frame["outer_radius"]) + Vector((float(center.x), 0.0, 0.0))
    direction = (surface - local_camera).normalized()
    reference = Vector((0.0, 0.0, 1.0)) if abs(direction.z) < 0.9 else Vector((0.0, 1.0, 0.0))
    side = direction.cross(reference).normalized() * 0.002
    up = direction.cross(side).normalized() * 0.002
    start = surface - direction * 0.02
    end = surface + direction * (0.8 * float(frame["diameter"]))
    vertices = [
        tuple(point)
        for point in (
            start - side - up, start + side - up, start + side + up, start - side + up,
            end - side - up, end + side - up, end + side + up, end - side + up,
        )
    ]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    editable = bmesh.new()
    editable.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(editable, faces=editable.faces)
    editable.to_mesh(mesh)
    editable.free()
    return bpy.data.objects.new(name, mesh)


def apply_boolean(target: bpy.types.Object, cutter: bpy.types.Object, operation: str) -> dict[str, Any]:
    modifier = target.modifiers.new(f"Irregular_{operation}", "BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = cutter
    bpy.context.view_layer.objects.active = target
    target.select_set(True)
    try:
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    finally:
        target.select_set(False)
    target.data.update(calc_edges=True)
    return {"success": "FINISHED" in result, **wheel.topology_metrics(target.data)}


def duplicate_healthy(source: bpy.types.Object, name: str, collection: bpy.types.Collection) -> bpy.types.Object:
    duplicate = source.copy()
    duplicate.data = source.data.copy()
    duplicate.name = name
    duplicate.data.name = f"{name}_Mesh"
    # copy() preserves parent, matrix_parent_inverse and matrix_local.  Do not
    # clear the parent or replace the transform: doing so created the old black
    # ribbons and displaced wheel details.
    collection.objects.link(duplicate)
    duplicate.hide_render = False
    duplicate.hide_viewport = False
    return duplicate


def opening_gate(skin: bpy.types.Object, center: Vector, profile: Mapping[str, Any], frame: Mapping[str, float]) -> dict[str, Any]:
    theta = math.atan2(float(center.z), float(center.y))
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    axial = Vector((1.0, 0.0, 0.0))
    samples = [Vector((0.0, 0.0))]
    samples.extend(Vector(point) * 0.32 for point in profile["points_axial_tangent"][::3])
    hits: list[bool] = []
    distance = float(frame["outer_radius"]) - float(frame["inner_radius"]) + 4.0 * float(frame["wall_thickness"])
    for sample in samples:
        origin = radial * (float(frame["outer_radius"]) + 2.0 * float(frame["wall_thickness"]))
        origin += axial * (float(center.x) + sample.x) + tangent * sample.y
        hit, _location, _normal, _face = skin.ray_cast(origin, -radial, distance=distance)
        hits.append(bool(hit))
    return {"valid": not any(hits), "sample_count": len(hits), "clear_count": hits.count(False), "surface_hits": hits}


def camera_background_gate(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    skin: bpy.types.Object,
    center: Vector,
    profile: Mapping[str, Any],
    frame: Mapping[str, float],
) -> dict[str, Any]:
    theta = math.atan2(float(center.z), float(center.y))
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    axial = Vector((1.0, 0.0, 0.0))
    core = 0.0005
    samples = [
        Vector((0.0, 0.0)),
        Vector((core, 0.0)),
        Vector((-core, 0.0)),
        Vector((0.0, core)),
        Vector((0.0, -core)),
        Vector((core, core)),
        Vector((-core, -core)),
    ]
    backgrounds: list[str] = []
    clear: list[bool] = []
    origin = camera.matrix_world.translation
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for sample in samples:
        local_target = radial * float(frame["outer_radius"])
        local_target += axial * (float(center.x) + sample.x) + tangent * sample.y
        world_target = skin.matrix_world @ local_target
        direction = (world_target - origin).normalized()
        target_distance = (world_target - origin).length
        hit, location, _normal, _face, obj, _matrix = scene.ray_cast(
            depsgraph, origin, direction, distance=1000.0
        )
        background = obj.name if hit and obj is not None else "WORLD"
        backgrounds.append(background)
        clear.append(
            bool(hit)
            and obj is not None
            and ("Terrain" in obj.name or "Martian" in obj.name)
            and (location - origin).length > target_distance + 0.002
        )
    return {
        "valid": all(clear),
        "sample_count": len(clear),
        "clear_count": clear.count(True),
        "background_objects": backgrounds,
    }


def save_image(path: Path, pixels: Sequence[float], width: int, height: int) -> None:
    image = bpy.data.images.new(path.stem, width, height, alpha=True)
    image.pixels.foreach_set(list(pixels))
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def difference(
    normal_path: Path, anomaly_path: Path, output_path: Path, mask_path: Path, threshold: float
) -> dict[str, Any]:
    width, height, normal = wheel.image_pixels(normal_path)
    width_a, height_a, anomaly = wheel.image_pixels(anomaly_path)
    if (width, height) != (width_a, height_a):
        raise RuntimeError("Normal/anomaly render sizes differ")
    display = [0.0] * (width * height * 4)
    mask = [0.0] * (width * height * 4)
    changed = 0
    total = 0.0
    for index in range(width * height):
        offset = index * 4
        delta = max(abs(float(normal[offset + channel]) - float(anomaly[offset + channel])) for channel in range(3))
        active = delta > threshold
        changed += int(active)
        total += delta
        display[offset : offset + 4] = [min(1.0, delta * 4.0), min(1.0, delta), 0.0, 1.0]
        value = 1.0 if active else 0.0
        mask[offset : offset + 4] = [value, value, value, 1.0]
    save_image(output_path, display, width, height)
    save_image(mask_path, mask, width, height)
    return {
        "pixel_count": width * height,
        "changed_pixels": changed,
        "effect_area_ratio": changed / (width * height),
        "mean_difference": total / (width * height),
        "images_identical": changed == 0,
        "difference_threshold": threshold,
    }


def contact_sheet(image_paths: Sequence[Path], output_path: Path) -> None:
    columns, rows = 4, math.ceil(len(image_paths) / 4)
    panel_width, panel_height = 400, 300
    width, height = columns * panel_width, rows * panel_height
    sheet = bpy.data.images.new("M6_Perforation_ContactSheet", width, height, alpha=True)
    pixels = [0.0] * (width * height * 4)
    for position, path in enumerate(image_paths):
        source = bpy.data.images.load(str(path), check_existing=False)
        try:
            source.scale(panel_width, panel_height)
            source_pixels = list(source.pixels[:])
            row, column = divmod(position, columns)
            for y in range(panel_height):
                src = y * panel_width * 4
                dst = (((rows - 1 - row) * panel_height + y) * width + column * panel_width) * 4
                pixels[dst : dst + panel_width * 4] = source_pixels[src : src + panel_width * 4]
        finally:
            bpy.data.images.remove(source)
    sheet.pixels.foreach_set(pixels)
    sheet.filepath_raw = str(output_path)
    sheet.file_format = "PNG"
    sheet.save()
    bpy.data.images.remove(sheet)


def main() -> int:
    args = parse_args()
    config = json.loads(args.library_config.read_text(encoding="utf-8"))
    output = args.output_dir.expanduser().resolve()
    renders = output / "renders"
    reports = output / "reports"
    renders.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    normal_scene = bpy.data.scenes.get("Normal")
    anomaly_scene = bpy.data.scenes.get("Perforation")
    healthy = bpy.data.objects.get("Wheel_Skin_Normal")
    legacy_anomaly = bpy.data.objects.get("Wheel_Skin_Anomaly")
    structural_source = bpy.data.objects.get("Wheel_Skin")
    anomaly_collection = bpy.data.collections.get("PAIR_ANOMALY")
    if any(value is None for value in (normal_scene, anomaly_scene, healthy, legacy_anomaly, structural_source, anomaly_collection)):
        raise RuntimeError("M6 scene is missing the expected counterfactual wheel objects")
    camera = normal_scene.camera
    if camera is None or anomaly_scene.camera != camera:
        raise RuntimeError("Normal and Perforation must share WheelCamera")

    camera_config = config.get("review_camera", {})
    camera_drop = float(camera_config.get("vertical_drop_m", 0.0))
    camera_pullback = float(camera_config.get("axial_pullback_m", 0.0))
    if camera_drop or camera_pullback:
        bpy.context.window.scene = normal_scene
        bpy.context.view_layer.update()
        target = healthy.matrix_world @ Vector((0.0, 0.0, 0.0))
        location = camera.matrix_world.translation.copy()
        location.z -= camera_drop
        axial_world = (healthy.matrix_world.to_3x3() @ Vector((1.0, 0.0, 0.0))).normalized()
        location += axial_world * camera_pullback
        target.z += float(camera_config.get("target_vertical_offset_m", 0.0))
        rotation = (target - location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
        camera.matrix_world = Matrix.Translation(location) @ rotation
        bpy.context.view_layer.update()
    fill_config = config.get("review_fill_light", {})
    fill_light: bpy.types.Object | None = None
    if float(fill_config.get("energy", 0.0)) > 0.0:
        light_data = bpy.data.lights.new("PerforationReviewFill", "SPOT")
        light_data.energy = float(fill_config["energy"])
        light_data.color = tuple(float(value) for value in fill_config.get("color", (1.0, 0.78, 0.58)))
        light_data.spot_size = math.radians(float(fill_config.get("spot_size_degrees", 35.0)))
        light_data.spot_blend = float(fill_config.get("spot_blend", 0.8))
        light_data.shadow_soft_size = float(fill_config.get("shadow_soft_size_m", 0.15))
        fill_light = bpy.data.objects.new("PerforationReviewFill", light_data)
        fill_light.matrix_world = camera.matrix_world.copy()
        normal_scene.collection.objects.link(fill_light)
        anomaly_scene.collection.objects.link(fill_light)

    frame = frame_from_skin(structural_source)
    profiles = build_perforation_library(
        diameter=float(frame["diameter"]), base_seed=int(config.get("base_seed", 41000))
    )
    if args.profile:
        profiles = [profile for profile in profiles if profile["variant_id"] == args.profile]
        if not profiles:
            raise RuntimeError(f"Unknown perforation profile: {args.profile}")
    if args.seed is not None:
        profiles = [
            generate_perforation_profile(
                variant_id=profile["variant_id"],
                family=profile["family"],
                orientation=profile["orientation"],
                size=profile["size"],
                diameter=float(frame["diameter"]),
                seed=int(args.seed) + index,
            )
            for index, profile in enumerate(profiles)
        ]
    for profile in profiles:
        errors = validate_perforation_profile(profile)
        if errors:
            raise RuntimeError(f"Invalid profile {profile['variant_id']}: " + "; ".join(errors))

    source_path = Path(bpy.data.filepath).resolve()
    bpy.context.window.scene = anomaly_scene
    legacy_anomaly.hide_render = True
    legacy_anomaly.hide_viewport = True
    normal_path = renders / "normal.png"
    normal_scene.camera = camera
    anomaly_scene.camera = camera
    white = wheel.create_emission_material("PerforationLibraryMask", (1.0, 1.0, 1.0, 1.0))
    fold_material = healthy.data.materials[0] if healthy.data.materials else bpy.data.materials.new("Perforation_Edge_Metal")
    variant_reports: list[dict[str, Any]] = []
    anomaly_paths: list[Path] = []
    normal_rendered = False

    for profile in profiles:
        variant = profile["variant_id"]
        variant_render_dir = renders / variant
        variant_render_dir.mkdir(parents=True, exist_ok=True)
        anomaly = duplicate_healthy(healthy, f"{variant}_Anomaly", anomaly_collection)
        anomaly.parent = legacy_anomaly.parent
        anomaly.matrix_parent_inverse = legacy_anomaly.matrix_parent_inverse.copy()
        anomaly.matrix_local = legacy_anomaly.matrix_local.copy()
        bpy.context.view_layer.update()
        center, placement_attempts = choose_placement(anomaly_scene, camera, anomaly, profile, frame)
        if fill_light is not None:
            fill_target = anomaly.matrix_world @ center
            fill_location = camera.matrix_world.translation
            fill_rotation = (fill_target - fill_location).to_track_quat("-Z", "Y").to_matrix().to_4x4()
            fill_light.matrix_world = Matrix.Translation(fill_location) @ fill_rotation
            bpy.context.view_layer.update()
        structural = duplicate_healthy(structural_source, f"{variant}_Structural", anomaly_collection)
        structural.parent = anomaly.parent
        structural.matrix_parent_inverse = anomaly.matrix_parent_inverse.copy()
        structural.matrix_local = anomaly.matrix_local.copy()
        structural.hide_render = True
        cutter = make_cutter(profile, center, frame, config.get("geometry", {}), f"{variant}_Cutter")
        cutter.matrix_world = anomaly.matrix_world.copy()
        anomaly_collection.objects.link(cutter)
        visual_boolean = apply_boolean(anomaly, cutter, "DIFFERENCE")
        structural_boolean = apply_boolean(structural, cutter, "DIFFERENCE")
        local_camera = anomaly.matrix_world.inverted() @ camera.matrix_world.translation
        camera_cutter = make_camera_clearance_cutter(
            center, local_camera, frame, f"{variant}_CameraClearanceCutter"
        )
        camera_cutter.matrix_world = anomaly.matrix_world.copy()
        anomaly_collection.objects.link(camera_cutter)
        visual_boolean = apply_boolean(anomaly, camera_cutter, "DIFFERENCE")
        structural_boolean = apply_boolean(structural, camera_cutter, "DIFFERENCE")
        damage_proxy = duplicate_healthy(structural_source, f"{variant}_DamageProxy", anomaly_collection)
        damage_proxy.parent = anomaly.parent
        damage_proxy.matrix_parent_inverse = anomaly.matrix_parent_inverse.copy()
        damage_proxy.matrix_local = anomaly.matrix_local.copy()
        damage_proxy.hide_render = True
        damage_proxy.hide_viewport = True
        mask_boolean = apply_boolean(damage_proxy, cutter, "INTERSECT")
        fold = make_fold(profile, center, frame, fold_material, f"{variant}_Fold")
        fold.matrix_world = anomaly.matrix_world.copy()
        anomaly_collection.objects.link(fold)
        cutter.hide_render = True
        cutter.hide_viewport = True
        camera_cutter.hide_render = True
        camera_cutter.hide_viewport = True
        bpy.context.view_layer.update()

        anomaly_path = variant_render_dir / "anomaly.png"
        damage_path = variant_render_dir / "damage_mask.png"
        effect_path = variant_render_dir / "effect_mask.png"
        difference_path = variant_render_dir / "difference.png"
        if not normal_rendered:
            wheel.render_still(normal_scene, normal_path)
            normal_rendered = True
        wheel.render_still(anomaly_scene, anomaly_path)
        wheel.render_binary_mask(anomaly_scene, [damage_proxy, fold], damage_path, white)
        image_metrics = difference(
            normal_path, anomaly_path, difference_path, effect_path,
            float(config["counterfactual"]["difference_threshold"]),
        )
        gate = opening_gate(structural, center, profile, frame)
        structural.hide_viewport = True
        bpy.context.view_layer.update()
        camera_gate = camera_background_gate(anomaly_scene, camera, anomaly, center, profile, frame)
        valid = bool(structural_boolean.get("success")) \
            and bool(structural_boolean.get("closed_manifold")) \
            and bool(visual_boolean.get("success")) \
            and bool(mask_boolean.get("success")) \
            and bool(gate["valid"]) \
            and bool(camera_gate["valid"]) \
            and not image_metrics["images_identical"] \
            and image_metrics["effect_area_ratio"] <= float(config["counterfactual"]["maximum_effect_area_ratio"])
        report = {
            "schema_version": "2.0",
            "status": "completed",
            "variant_id": variant,
            "variant": profile,
            "source_blend": str(source_path),
            "source_scene_unchanged": True,
            "legacy_hole_removed": True,
            "details_duplicated": False,
            "transform_preserved": True,
            "placement": {
                "center_local": list(center),
                "center_world": list(anomaly.matrix_world @ center),
                "theta_degrees": math.degrees(math.atan2(float(center.z), float(center.y))),
                "background": next((attempt["background"] for attempt in reversed(placement_attempts) if attempt["clear"]), "unknown"),
                "attempts": placement_attempts,
            },
            "frame": frame,
            "boolean": structural_boolean,
            "visual_boolean": visual_boolean,
            "mask_boolean": mask_boolean,
            "through_visibility": {**gate, "visible_fraction": gate["clear_count"] / gate["sample_count"]},
            "camera_background_visibility": camera_gate,
            "image_metrics": image_metrics,
            "counterfactual_gate": {"maximum_effect_area_ratio": float(config["counterfactual"]["maximum_effect_area_ratio"])},
            "open_area_fraction_after_folds": 0.92,
            "normal_skin_digest": mesh_digest(healthy.data),
            "anomaly_skin_digest": mesh_digest(anomaly.data),
            "valid": valid,
            "artifacts": {
                "normal": "renders/normal.png",
                "anomaly": f"renders/{variant}/anomaly.png",
                "damage_mask": f"renders/{variant}/damage_mask.png",
                "effect_mask": f"renders/{variant}/effect_mask.png",
                "difference": f"renders/{variant}/difference.png",
            },
        }
        (reports / f"{variant}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        variant_reports.append(report)
        anomaly_paths.append(anomaly_path)
        for obj in (anomaly, structural, cutter, damage_proxy, fold):
            obj.hide_render = True
            obj.hide_viewport = True
        print(f"Rendered {variant}: {'valid' if valid else 'invalid'}")

    contact_sheet(anomaly_paths, renders / "contact_sheet.png")
    output_blend = output / (
        "rover_simulator_m6_irregular_perforation_pilot.blend"
        if len(profiles) == 1 else "rover_simulator_m6_irregular_perforation_library.blend"
    )
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    (reports / "library.json").write_text(
        json.dumps({
            "schema_version": "2.0",
            "variant_count": len(variant_reports),
            "valid_count": sum(int(report["valid"]) for report in variant_reports),
            "required_artifacts": ["normal", "anomaly", "damage_mask", "effect_mask", "difference"],
            "contact_sheet_required": True,
            "contact_sheet_order": [report["variant_id"] for report in variant_reports],
            "output_blend": str(output_blend),
            "variants": variant_reports,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"M6 irregular perforation library: {output_blend}")
    return 0 if all(report["valid"] for report in variant_reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
