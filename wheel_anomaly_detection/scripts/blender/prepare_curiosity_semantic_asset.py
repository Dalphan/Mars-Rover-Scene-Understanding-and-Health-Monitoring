"""Prepare a minimally modified, semantically addressable Curiosity asset.

Run this inside Blender after importing the verified NASA GLB.  The script
separates only geometry whose material is named ``wheels``.  Every connected
component using that material is assigned to one of six spatial clusters; no
mesh is rebuilt, so face material indices, UV loops and custom normals survive
Blender's native Separate operation.

Blender command-line arguments may be supplied after ``--``::

    --source-glb PATH --output-dir PATH

When invoked through Blender MCP, ``SOURCE_GLB`` and ``OUTPUT_DIR`` may instead
be injected as globals before executing this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy


EXPECTED_WHEELS = (
    "wheel_front_left",
    "wheel_front_right",
    "wheel_middle_left",
    "wheel_middle_right",
    "wheel_rear_left",
    "wheel_rear_right",
)


def _arguments() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-glb", type=Path)
    parser.add_argument("--output-dir", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    source = globals().get("SOURCE_GLB") or args.source_glb
    output = globals().get("OUTPUT_DIR") or args.output_dir
    if not source or not output:
        raise RuntimeError("SOURCE_GLB and OUTPUT_DIR are required")
    return Path(source).resolve(), Path(output).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _world_bbox(obj: bpy.types.Object) -> list[list[float]]:
    corners = [obj.matrix_world @ obj.data.vertices[index].co for index in range(len(obj.data.vertices))]
    return [
        [min(point[axis] for point in corners) for axis in range(3)],
        [max(point[axis] for point in corners) for axis in range(3)],
    ]


def _connected_components(obj: bpy.types.Object) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    mesh = obj.data
    parent = list(range(len(mesh.vertices)))
    rank = [0] * len(parent)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    for edge in mesh.edges:
        union(edge.vertices[0], edge.vertices[1])

    vertices_by_root: dict[int, list[int]] = defaultdict(list)
    polygons_by_root: dict[int, list[int]] = defaultdict(list)
    for vertex in mesh.vertices:
        vertices_by_root[find(vertex.index)].append(vertex.index)
    for polygon in mesh.polygons:
        polygons_by_root[find(polygon.vertices[0])].append(polygon.index)
    return dict(vertices_by_root), dict(polygons_by_root)


def audit_source(obj: bpy.types.Object, source_glb: Path) -> dict:
    mesh = obj.data
    vertices_by_root, polygons_by_root = _connected_components(obj)
    wheel_slot = next(
        (index for index, slot in enumerate(obj.material_slots) if slot.material and slot.material.name == "wheels"),
        None,
    )
    if wheel_slot is None:
        raise RuntimeError("The imported mesh has no material named 'wheels'")

    wheel_roots = []
    mixed_roots = []
    records = []
    for root, polygon_ids in polygons_by_root.items():
        slots = {mesh.polygons[index].material_index for index in polygon_ids}
        if wheel_slot not in slots:
            continue
        wheel_roots.append(root)
        if slots != {wheel_slot}:
            mixed_roots.append({"root": root, "material_slots": sorted(slots)})
        vertex_ids = vertices_by_root[root]
        points = [obj.matrix_world @ mesh.vertices[index].co for index in vertex_ids]
        records.append(
            {
                "root": root,
                "vertices": len(vertex_ids),
                "faces": len(polygon_ids),
                "center": [sum(point[axis] for point in points) / len(points) for axis in range(3)],
            }
        )

    row_centers = [-1.1, 0.09, 1.16]
    for _ in range(30):
        buckets = [[] for _ in row_centers]
        for record in records:
            row = min(range(3), key=lambda index: abs(record["center"][1] - row_centers[index]))
            buckets[row].append((record["center"][1], record["vertices"]))
        updated = [sum(y * weight for y, weight in bucket) / sum(weight for _, weight in bucket) for bucket in buckets]
        if max(abs(updated[index] - row_centers[index]) for index in range(3)) < 1e-12:
            break
        row_centers = updated
    row_centers.sort()

    rows = ("front", "middle", "rear")
    clusters: dict[str, list[int]] = defaultdict(list)
    for record in records:
        row = min(range(3), key=lambda index: abs(record["center"][1] - row_centers[index]))
        side = "left" if record["center"][0] > 0 else "right"
        clusters[f"wheel_{rows[row]}_{side}"].append(record["root"])

    cluster_report = {}
    for name in EXPECTED_WHEELS:
        roots = set(clusters.get(name, []))
        vertex_ids = sorted({index for root in roots for index in vertices_by_root[root]})
        polygon_ids = sorted({index for root in roots for index in polygons_by_root[root]})
        points = [obj.matrix_world @ mesh.vertices[index].co for index in vertex_ids]
        cluster_report[name] = {
            "connected_components": len(roots),
            "vertices": len(vertex_ids),
            "faces": len(polygon_ids),
            "uv_loops": sum(mesh.polygons[index].loop_total for index in polygon_ids),
            "bbox_world": [
                [min(point[axis] for point in points) for axis in range(3)],
                [max(point[axis] for point in points) for axis in range(3)],
            ],
        }

    return {
        "source_glb": str(source_glb),
        "source_sha256": _sha256(source_glb),
        "blender_version": bpy.app.version_string,
        "source_object": obj.name,
        "source_transform": {
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "scale": list(obj.scale),
        },
        "mesh": {
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "faces": len(mesh.polygons),
            "connected_components": len(vertices_by_root),
            "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "material_face_counts": dict(
                Counter(obj.material_slots[polygon.material_index].material.name for polygon in mesh.polygons)
            ),
            "uv_layers": [layer.name for layer in mesh.uv_layers],
            "has_custom_normals": bool(mesh.has_custom_normals),
        },
        "wheel_material_slot": wheel_slot,
        "wheel_components_total": len(wheel_roots),
        "mixed_material_wheel_components": mixed_roots,
        "axis_convention": {
            "front": "negative world Y",
            "rear": "positive world Y",
            "left": "positive world X when facing front",
            "right": "negative world X",
            "up": "positive world Z",
        },
        "row_centers_y": row_centers,
        "wheel_clusters": cluster_report,
    }


def _new_collection(name: str, parent: bpy.types.Collection) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if collection.name not in parent.children:
        parent.children.link(collection)
    return collection


def _move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for current in list(obj.users_collection):
        if current != collection:
            current.objects.unlink(obj)


def _new_empty(name: str, collection: bpy.types.Collection, parent: bpy.types.Object | None = None) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.35
    collection.objects.link(empty)
    empty.parent = parent
    return empty


def _separate_wheel(body: bpy.types.Object, name: str) -> bpy.types.Object:
    row_name, side = name.removeprefix("wheel_").rsplit("_", 1)
    row_index = {"front": 0, "middle": 1, "rear": 2}[row_name]
    row_centers = (-1.098105736, 0.086624896, 1.161624066)
    boundaries = ((row_centers[0] + row_centers[1]) / 2, (row_centers[1] + row_centers[2]) / 2)
    wheel_slot = next(
        (index for index, slot in enumerate(body.material_slots) if slot.material and slot.material.name == "wheels"),
        None,
    )
    if wheel_slot is None:
        raise RuntimeError(f"The remaining body mesh lost the wheels material before separating {name}")

    def belongs(polygon: bpy.types.MeshPolygon) -> bool:
        if polygon.material_index != wheel_slot:
            return False
        center = body.matrix_world @ polygon.center
        actual_row = 0 if center.y < boundaries[0] else (1 if center.y < boundaries[1] else 2)
        actual_side = "left" if center.x > 0 else "right"
        return actual_row == row_index and actual_side == side

    bpy.ops.object.mode_set(mode="OBJECT") if body.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.object.mode_set(mode="OBJECT")
    for polygon in body.data.polygons:
        polygon.select = belongs(polygon)
    if not any(polygon.select for polygon in body.data.polygons):
        raise RuntimeError(f"No polygons selected for {name}")

    previous_objects = set(bpy.context.scene.objects)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    created = [obj for obj in set(bpy.context.scene.objects) - previous_objects if obj.type == "MESH"]
    if len(created) != 1:
        raise RuntimeError(f"Expected one separated object for {name}, found {len(created)}")
    wheel = created[0]
    wheel.name = name
    wheel.data.name = f"{name}_mesh"
    bpy.ops.object.select_all(action="DESELECT")
    wheel.select_set(True)
    bpy.context.view_layer.objects.active = wheel
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    return wheel


def build(source_glb: Path, output_dir: Path) -> dict:
    source_sha_before = _sha256(source_glb)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name == "MSL"]
    if len(mesh_objects) != 1:
        raise RuntimeError(f"Expected exactly one imported mesh named MSL, found {len(mesh_objects)}")
    source = mesh_objects[0]
    audit = audit_source(source, source_glb)
    expected_cluster_shape = {(324, 2742, 1912, 5736)}
    actual_cluster_shape = {
        (item["connected_components"], item["vertices"], item["faces"], item["uv_loops"])
        for item in audit["wheel_clusters"].values()
    }
    if set(audit["wheel_clusters"]) != set(EXPECTED_WHEELS) or actual_cluster_shape != expected_cluster_shape:
        raise RuntimeError(f"Unexpected wheel clustering: {actual_cluster_shape}")
    if audit["mixed_material_wheel_components"]:
        raise RuntimeError("Wheel components unexpectedly mix material slots")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "audit_baseline.blend"))

    for name in ("Cube", "Camera", "Light"):
        obj = bpy.data.objects.get(name)
        if obj is not None and obj != source:
            bpy.data.objects.remove(obj, do_unlink=True)

    scene_root = bpy.context.scene.collection
    rover_collection = _new_collection("Rover", scene_root)
    body_collection = _new_collection("Body", rover_collection)
    suspension_collection = _new_collection("Suspension", rover_collection)
    wheels_collection = _new_collection("Wheels", rover_collection)

    rover_root = _new_empty("Rover", rover_collection)
    body_root = _new_empty("Body", body_collection, rover_root)
    suspension_root = _new_empty("Suspension", suspension_collection, rover_root)
    wheels_root = _new_empty("Wheels", wheels_collection, rover_root)

    source.name = "body_suspension_assembly"
    source.data.name = "body_suspension_assembly_mesh"
    _move_to_collection(source, body_collection)
    source.parent = body_root
    source["semantic_roles"] = "body;chassis;suspension;rocker_bogie;wheel_adjacent_components"
    source["separation_status"] = "identified_together_not_separated"
    suspension_root["geometry_object"] = source.name
    suspension_root["separation_status"] = "identified_within_body_suspension_assembly"

    wheels = []
    for name in EXPECTED_WHEELS:
        wheel = _separate_wheel(source, name)
        _move_to_collection(wheel, wheels_collection)
        world_matrix = wheel.matrix_world.copy()
        wheel.parent = wheels_root
        wheel.matrix_world = world_matrix
        wheel["semantic_role"] = "wheel"
        wheel["source_material"] = "wheels"
        wheels.append(wheel)

    for collection in list(bpy.data.collections):
        if collection.name == "Collection" and not collection.objects and not collection.children:
            bpy.data.collections.remove(collection)

    bpy.context.scene["asset_source_sha256"] = source_sha_before
    bpy.context.scene["axis_convention"] = "front=-Y; rear=+Y; left=+X; right=-X; up=+Z"
    bpy.context.scene["semantic_schema_version"] = "1.0"

    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "curiosity_semantic_clean.blend"))
    source_sha_after = _sha256(source_glb)
    if source_sha_after != source_sha_before:
        raise RuntimeError("Source GLB checksum changed during processing")

    result = {
        "blend": str(output_dir / "curiosity_semantic_clean.blend"),
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "wheels": [wheel.name for wheel in wheels],
        "body_object": source.name,
        "body_and_suspension_intentionally_combined": True,
    }
    (output_dir / "build_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    build(*_arguments())
