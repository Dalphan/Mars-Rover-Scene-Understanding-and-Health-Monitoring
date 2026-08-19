from __future__ import annotations

import argparse
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Validate a reopened wheel-preparation blend artifact."
    )
    parser.add_argument("--kind", choices=("canonical", "perforation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def missing_external_resources() -> list[str]:
    missing: list[str] = []
    for image in bpy.data.images:
        if image.source != "FILE" or image.packed_file is not None:
            continue
        raw_path = str(image.filepath or image.filepath_raw)
        if not raw_path:
            continue
        resolved = Path(bpy.path.abspath(raw_path))
        if not resolved.is_file():
            missing.append(f"image:{image.name}:{resolved}")
    return missing


def object_names(scene: bpy.types.Scene, object_type: str) -> list[str]:
    return sorted(obj.name for obj in scene.objects if obj.type == object_type)


def mesh_geometry_digest(mesh: bpy.types.Mesh) -> str:
    digest = hashlib.sha256()
    digest.update(f"{len(mesh.vertices)}:{len(mesh.edges)}:{len(mesh.polygons)}".encode())
    for vertex in mesh.vertices:
        digest.update(
            ("%.8f,%.8f,%.8f;" % tuple(float(value) for value in vertex.co)).encode()
        )
    for polygon in mesh.polygons:
        digest.update((",".join(str(int(value)) for value in polygon.vertices) + ";").encode())
    return digest.hexdigest()


def mesh_component_descriptors(mesh: bpy.types.Mesh) -> list[dict[str, Any]]:
    adjacency = [set() for _ in mesh.vertices]
    for edge in mesh.edges:
        left, right = (int(value) for value in edge.vertices)
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[list[int]] = []
    component_by_vertex: dict[int, int] = {}
    unseen = set(range(len(mesh.vertices)))
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        vertices: list[int] = []
        while stack:
            current = stack.pop()
            component_by_vertex[current] = len(components)
            vertices.append(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(vertices)
    face_counts = [0] * len(components)
    for polygon in mesh.polygons:
        if polygon.vertices:
            face_counts[component_by_vertex[int(polygon.vertices[0])]] += 1
    descriptors = []
    for index, vertices in enumerate(components):
        coordinates = [mesh.vertices[value].co for value in vertices]
        radii = [math.hypot(float(point.y), float(point.z)) for point in coordinates]
        descriptors.append(
            {
                "index": index,
                "vertex_count": len(vertices),
                "face_count": face_counts[index],
                "axial_min": min(float(point.x) for point in coordinates),
                "axial_max": max(float(point.x) for point in coordinates),
                "axial_mean": sum(float(point.x) for point in coordinates)
                / len(coordinates),
                "radial_min": min(radii),
                "radial_max": max(radii),
                "radial_mean": sum(radii) / len(radii),
            }
        )
    return descriptors


def validate_canonical(errors: list[str], checks: dict[str, Any]) -> None:
    scene = bpy.data.scenes.get("PreparationWorkspace")
    checks["scene_present"] = scene is not None
    if scene is None:
        errors.append("Missing PreparationWorkspace scene")
        return
    required_objects = {"Wheel_Attachment", "Wheel_Details", "Wheel_Skin"}
    present = {obj.name for obj in scene.objects}
    checks["required_objects_present"] = required_objects.issubset(present)
    if not checks["required_objects_present"]:
        errors.append(
            "Canonical scene is missing objects: "
            + ", ".join(sorted(required_objects - present))
        )
    checks["camera_present"] = scene.camera is not None
    if scene.camera is None:
        errors.append("Canonical scene has no active camera")
    collection = bpy.data.collections.get("WHEEL_CANONICAL")
    checks["canonical_collection_present"] = collection is not None
    if collection is None:
        errors.append("Missing WHEEL_CANONICAL collection")
    else:
        collection_objects = {obj.name for obj in collection.objects}
        checks["canonical_collection_objects"] = sorted(collection_objects)
        if not required_objects.issubset(collection_objects):
            errors.append("WHEEL_CANONICAL does not contain all canonical objects")
    if required_objects.issubset(present):
        skin = scene.objects["Wheel_Skin"]
        details = scene.objects["Wheel_Details"]
        skin_coordinates = [vertex.co for vertex in skin.data.vertices]
        outer_radius = max(
            math.hypot(float(point.y), float(point.z)) for point in skin_coordinates
        )
        axial_min = min(float(point.x) for point in skin_coordinates)
        axial_max = max(float(point.x) for point in skin_coordinates)
        descriptors = mesh_component_descriptors(details.data)
        outer_axial_outliers = [
            descriptor
            for descriptor in descriptors
            if descriptor["radial_min"] >= outer_radius * 0.72
            and (
                descriptor["axial_max"] < axial_min
                or descriptor["axial_min"] > axial_max
            )
        ]
        checks["detail_component_count"] = len(descriptors)
        checks["detail_components"] = descriptors
        checks["skin_envelope"] = {
            "outer_radius": outer_radius,
            "axial_min": axial_min,
            "axial_max": axial_max,
        }
        checks["outer_axial_detail_outliers"] = outer_axial_outliers
        checks["outer_detail_components"] = [
            descriptor
            for descriptor in descriptors
            if descriptor["radial_min"] >= outer_radius * 0.72
        ]
    checks["canonical_axis"] = scene.get("canonical_axis")
    checks["outboard_direction"] = scene.get("outboard_direction")
    if checks["canonical_axis"] != "X" or checks["outboard_direction"] != "+X":
        errors.append("Canonical axis metadata is missing or invalid")


def validate_perforation(errors: list[str], checks: dict[str, Any]) -> None:
    normal = bpy.data.scenes.get("Normal")
    anomaly = bpy.data.scenes.get("Perforation")
    checks["normal_scene_present"] = normal is not None
    checks["perforation_scene_present"] = anomaly is not None
    if normal is None or anomaly is None:
        errors.append("Missing Normal or Perforation scene")
        return
    normal_required = {"Wheel_Attachment", "Wheel_Details_Normal", "Wheel_Skin_Normal"}
    anomaly_required = {"Wheel_Attachment", "Wheel_Details_Perforation", "Wheel_Skin_Anomaly"}
    normal_objects = {obj.name for obj in normal.objects}
    anomaly_objects = {obj.name for obj in anomaly.objects}
    checks["normal_objects_present"] = normal_required.issubset(normal_objects)
    checks["anomaly_objects_present"] = anomaly_required.issubset(anomaly_objects)
    if not checks["normal_objects_present"]:
        errors.append("Normal scene does not contain the expected wheel objects")
    if not checks["anomaly_objects_present"]:
        errors.append("Perforation scene does not contain the expected wheel objects")
    for label, scene in (("normal", normal), ("anomaly", anomaly)):
        rover_objects = [
            obj.name
            for obj in scene.objects
            if obj.get("role") == "full_rover_context_without_selected_wheel"
        ]
        terrain_objects = [
            obj.name
            for obj in scene.objects
            if obj.get("role") == "mars_terrain_context" and obj.type == "MESH"
        ]
        checks[f"{label}_rover_context_objects"] = rover_objects
        checks[f"{label}_terrain_mesh_count"] = len(terrain_objects)
        if len(rover_objects) != 1:
            errors.append(f"{label} scene is missing the full rover context")
        if not terrain_objects:
            errors.append(f"{label} scene is missing Mars terrain meshes")
    if checks["normal_objects_present"] and checks["anomaly_objects_present"]:
        normal_skin = normal.objects["Wheel_Skin_Normal"]
        anomaly_skin = anomaly.objects["Wheel_Skin_Anomaly"]
        normal_digest = mesh_geometry_digest(normal_skin.data)
        anomaly_digest = mesh_geometry_digest(anomaly_skin.data)
        checks["normal_skin_counts"] = {
            "vertices": len(normal_skin.data.vertices),
            "edges": len(normal_skin.data.edges),
            "faces": len(normal_skin.data.polygons),
        }
        checks["anomaly_skin_counts"] = {
            "vertices": len(anomaly_skin.data.vertices),
            "edges": len(anomaly_skin.data.edges),
            "faces": len(anomaly_skin.data.polygons),
        }
        checks["normal_skin_digest"] = normal_digest
        checks["anomaly_skin_digest"] = anomaly_digest
        checks["skin_geometry_differs"] = normal_digest != anomaly_digest
        if not checks["skin_geometry_differs"]:
            errors.append("Normal and perforated skin geometry are identical")
        normal_coordinates = {
            tuple(round(float(value), 7) for value in vertex.co)
            for vertex in normal_skin.data.vertices
        }
        added_coordinates = [
            tuple(float(value) for value in vertex.co)
            for vertex in anomaly_skin.data.vertices
            if tuple(round(float(value), 7) for value in vertex.co)
            not in normal_coordinates
        ]
        checks["boolean_added_vertex_count"] = len(added_coordinates)
        if added_coordinates:
            checks["boolean_added_vertex_bbox"] = {
                "min": [min(point[axis] for point in added_coordinates) for axis in range(3)],
                "max": [max(point[axis] for point in added_coordinates) for axis in range(3)],
            }
            angles = [
                math.degrees(math.atan2(point[2], point[1])) % 360.0
                for point in added_coordinates
            ]
            checks["boolean_added_vertex_angles_degrees"] = {
                "minimum": min(angles),
                "maximum": max(angles),
                "circular_mean": (
                    math.degrees(
                        math.atan2(
                            sum(math.sin(math.radians(value)) for value in angles),
                            sum(math.cos(math.radians(value)) for value in angles),
                        )
                    )
                    % 360.0
                ),
            }
            theta = math.radians(
                float(checks["boolean_added_vertex_angles_degrees"]["circular_mean"])
            )
            radial = Vector((0.0, math.cos(theta), math.sin(theta)))
            outer_radius = max(
                math.hypot(float(vertex.co.y), float(vertex.co.z))
                for vertex in normal_skin.data.vertices
            )
            ray_results: dict[str, Any] = {}
            for label, obj in (("normal", normal_skin), ("anomaly", anomaly_skin)):
                tree = BVHTree.FromPolygons(
                    [vertex.co.copy() for vertex in obj.data.vertices],
                    [list(polygon.vertices) for polygon in obj.data.polygons],
                    all_triangles=False,
                )
                samples = []
                for axial in (-0.01, 0.0, 0.01):
                    origin = Vector((axial, 0.0, 0.0)) + radial * outer_radius * 1.25
                    location, normal_vector, polygon_index, _distance = tree.ray_cast(
                        origin, -radial, outer_radius * 3.0
                    )
                    hit = location is not None
                    samples.append(
                        {
                            "axial": axial,
                            "hit": bool(hit),
                            "location": (
                                [float(value) for value in location] if hit else None
                            ),
                            "normal": (
                                [float(value) for value in normal_vector]
                                if hit
                                else None
                            ),
                            "polygon_index": int(polygon_index) if hit else None,
                            "hit_radius": (
                                math.hypot(float(location.y), float(location.z))
                                if hit
                                else None
                            ),
                        }
                    )
                ray_results[label] = samples
            checks["radial_ray_samples_at_boolean"] = ray_results

    checks["same_camera"] = normal.camera is not None and normal.camera == anomaly.camera
    if not checks["same_camera"]:
        errors.append("Counterfactual scenes do not share the same active camera")
    normal_lights = object_names(normal, "LIGHT")
    anomaly_lights = object_names(anomaly, "LIGHT")
    checks["normal_lights"] = normal_lights
    checks["anomaly_lights"] = anomaly_lights
    checks["same_lights"] = normal_lights == anomaly_lights and bool(normal_lights)
    if not checks["same_lights"]:
        errors.append("Counterfactual scenes do not share the same lights")
    normal_render = (
        normal.render.engine,
        normal.render.resolution_x,
        normal.render.resolution_y,
        normal.render.resolution_percentage,
    )
    anomaly_render = (
        anomaly.render.engine,
        anomaly.render.resolution_x,
        anomaly.render.resolution_y,
        anomaly.render.resolution_percentage,
    )
    checks["normal_render"] = list(normal_render)
    checks["anomaly_render"] = list(anomaly_render)
    checks["same_render_settings"] = normal_render == anomaly_render
    if not checks["same_render_settings"]:
        errors.append("Counterfactual scene render settings differ")
    checks["scene_locks"] = {
        "normal_camera": normal.get("camera_locked"),
        "normal_lighting": normal.get("lighting_locked"),
        "anomaly_camera": anomaly.get("camera_locked"),
        "anomaly_lighting": anomaly.get("lighting_locked"),
    }
    if not all(value is True for value in checks["scene_locks"].values()):
        errors.append("Counterfactual camera/lighting lock metadata is incomplete")


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    checks: dict[str, Any] = {}
    checks["blender_version"] = bpy.app.version_string
    checks["blender_version_tuple"] = list(bpy.app.version)
    checks["source_blend"] = str(Path(bpy.data.filepath).resolve())
    if tuple(bpy.app.version[:2]) != (5, 2):
        errors.append(f"Expected Blender 5.2, got {bpy.app.version_string}")
    missing = missing_external_resources()
    checks["missing_external_resources"] = missing
    if missing:
        errors.extend(f"Missing external resource: {item}" for item in missing)
    if args.kind == "canonical":
        validate_canonical(errors, checks)
    else:
        validate_perforation(errors, checks)

    report = {
        "schema_version": "1.0",
        "status": "completed" if not errors else "failed",
        "kind": args.kind,
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated reopened {args.kind} blend: {bpy.data.filepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
