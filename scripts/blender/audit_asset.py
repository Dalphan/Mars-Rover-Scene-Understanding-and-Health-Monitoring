"""Headless GLB audit for Curiosity wheel editability.

Run with:
    blender --background --python scripts/blender/audit_asset.py -- \
        --asset path/to/asset.glb \
        --output-dir outputs/blender_audit/run_name

This file is executed by Blender's Python and intentionally imports no external
packages. It never writes to the source asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blender_audit.core import (  # noqa: E402
    bbox_from_points,
    center_from_bbox,
    classify_asset,
    connected_components,
    deep_merge,
    dimensions_from_bbox,
    make_markdown_report,
    repeat_counts,
    repeat_group_labels,
    score_wheel_candidate,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "blender" / "audit.json"
KNOWN_RENDER_ENGINES = (
    "BLENDER_EEVEE_NEXT",
    "BLENDER_EEVEE",
    "BLENDER_WORKBENCH",
    "CYCLES",
)
SOFTWARE_RENDERER_PATTERN = re.compile(
    r"(llvmpipe|softpipe|software|microsoft basic render|swiftshader)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args(script_args)


def create_output_tree(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "reports": output_dir / "reports",
        "diagnostics": output_dir / "diagnostics",
        "rover_renders": output_dir / "renders" / "rover",
        "candidate_renders": output_dir / "renders" / "candidates",
        "topology_renders": output_dir / "renders" / "topology",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("blender_asset_audit")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float_list(values: Iterable[Any], digits: int = 8) -> list[float]:
    return [round(float(value), digits) for value in values]


def matrix_as_rows(matrix: Matrix) -> list[list[float]]:
    return [as_float_list(row) for row in matrix]


def safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def load_config(path: Path) -> dict[str, Any]:
    default = {
        "schema_version": "1.0",
        "render": {
            "engine_preference": [
                "BLENDER_EEVEE_NEXT",
                "BLENDER_EEVEE",
                "BLENDER_WORKBENCH",
            ],
            "resolution_x": 800,
            "resolution_y": 600,
            "resolution_percentage": 100,
            "samples": 16,
            "image_format": "PNG",
            "transparent": False,
            "background_rgba": [0.035, 0.045, 0.065, 1.0],
            "camera_margin": 1.25,
        },
        "wheel_detection": {
            "candidate_threshold": 0.50,
            "min_component_vertices": 24,
            "min_component_faces": 16,
            "max_candidates": 18,
        },
        "topology": {
            "component_details_limit": 64,
            "thickness_samples": 96,
            "minimum_visible_thickness_ratio": 0.003,
        },
        "diagnostics": {
            "pack_resources_in_blend": True,
            "save_candidate_wireframes": True,
        },
        "logging": {"level": "INFO"},
    }
    with path.open("r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a JSON object")
    return deep_merge(default, loaded)


def available_render_engines(scene: bpy.types.Scene) -> list[str]:
    engines: set[str] = set()
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        engines.update(item.identifier for item in prop.enum_items)
    except Exception:
        pass
    original = scene.render.engine
    for identifier in KNOWN_RENDER_ENGINES:
        try:
            scene.render.engine = identifier
            engines.add(identifier)
        except Exception:
            continue
    try:
        scene.render.engine = original
    except Exception:
        pass
    return sorted(engines)


def select_render_engine(
    scene: bpy.types.Scene, preferences: Sequence[str], engines: Sequence[str]
) -> str:
    for identifier in preferences:
        if identifier not in engines:
            continue
        try:
            scene.render.engine = identifier
            return identifier
        except Exception:
            continue
    raise RuntimeError(f"No requested render engine is available; found {engines}")


def inspect_gpu() -> dict[str, Any]:
    info: dict[str, Any] = {
        "available": False,
        "backend": "unknown",
        "vendor": "unknown",
        "renderer": "unknown",
        "version": "unknown",
        "device_type": "unknown",
        "devices": [],
        "cycles_devices": [],
    }
    try:
        import gpu

        initializer = getattr(gpu, "init", None)
        if initializer:
            initializer()
        platform_api = gpu.platform
        getters = {
            "backend": "backend_type_get",
            "device_type": "device_type_get",
            "vendor": "vendor_get",
            "renderer": "renderer_get",
            "version": "version_get",
        }
        for key, getter_name in getters.items():
            getter = getattr(platform_api, getter_name, None)
            if getter:
                info[key] = str(getter())
        devices_get = getattr(platform_api, "devices_get", None)
        if devices_get:
            info["devices"] = [str(device) for device in devices_get()]
        renderer_text = " ".join(
            str(info.get(key, "")) for key in ("backend", "vendor", "renderer")
        )
        info["available"] = bool(
            info["renderer"] != "unknown"
            and not SOFTWARE_RENDERER_PATTERN.search(renderer_text)
        )
    except Exception as exc:
        info["probe_error"] = repr(exc)

    try:
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon:
            cycles_preferences = cycles_addon.preferences
            try:
                cycles_preferences.get_devices()
            except Exception:
                pass
            for device in getattr(cycles_preferences, "devices", []):
                info["cycles_devices"].append(
                    {
                        "name": str(getattr(device, "name", "")),
                        "type": str(getattr(device, "type", "")),
                        "use": bool(getattr(device, "use", False)),
                    }
                )
    except Exception as exc:
        info["cycles_probe_error"] = repr(exc)
    return info


def world_bbox_for_object(obj: bpy.types.Object) -> dict[str, Any]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum, maximum = bbox_from_points(corners)
    return {
        "min": as_float_list(minimum),
        "max": as_float_list(maximum),
        "center": as_float_list(center_from_bbox(minimum, maximum)),
        "dimensions": as_float_list(dimensions_from_bbox(minimum, maximum)),
        "corners": [as_float_list(corner) for corner in corners],
    }


def merge_bboxes(bboxes: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    if not bboxes:
        return {
            "min": [0.0, 0.0, 0.0],
            "max": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
            "dimensions": [0.0, 0.0, 0.0],
        }
    minimum = [
        min(float(bbox["min"][axis]) for bbox in bboxes) for axis in range(3)
    ]
    maximum = [
        max(float(bbox["max"][axis]) for bbox in bboxes) for axis in range(3)
    ]
    return {
        "min": as_float_list(minimum),
        "max": as_float_list(maximum),
        "center": as_float_list(center_from_bbox(minimum, maximum)),
        "dimensions": as_float_list(dimensions_from_bbox(minimum, maximum)),
    }


def analyze_edge_topology(
    mesh: bpy.types.Mesh,
) -> tuple[dict[str, Any], dict[tuple[int, int], list[tuple[int, int, int]]]]:
    face_uses: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for polygon in mesh.polygons:
        vertices = list(polygon.vertices)
        for index, left in enumerate(vertices):
            right = vertices[(index + 1) % len(vertices)]
            key = (left, right) if left < right else (right, left)
            direction = 1 if (left, right) == key else -1
            face_uses[key].append((polygon.index, direction, left))

    mesh_edge_keys = {
        tuple(sorted((int(edge.vertices[0]), int(edge.vertices[1]))))
        for edge in mesh.edges
    }
    loose_edges = len(mesh_edge_keys - set(face_uses))
    boundary_edges = sum(len(uses) == 1 for uses in face_uses.values())
    multi_face_edges = sum(len(uses) > 2 for uses in face_uses.values())
    inconsistent_edges = sum(
        len(uses) == 2 and uses[0][1] == uses[1][1]
        for uses in face_uses.values()
    )
    non_manifold = loose_edges + boundary_edges + multi_face_edges
    return (
        {
            "edge_count": len(mesh.edges),
            "loose_edge_count": loose_edges,
            "boundary_edge_count": boundary_edges,
            "multi_face_edge_count": multi_face_edges,
            "non_manifold_edge_count": non_manifold,
            "has_non_manifold_edges": non_manifold > 0,
            "inconsistent_orientation_edge_count": inconsistent_edges,
        },
        face_uses,
    )


def analyze_normals(mesh: bpy.types.Mesh, topology: Mapping[str, Any], obj: bpy.types.Object) -> dict[str, Any]:
    zero_length = 0
    non_finite = 0
    for polygon in mesh.polygons:
        normal = polygon.normal
        if not all(math.isfinite(float(value)) for value in normal):
            non_finite += 1
        elif normal.length_squared <= 1e-20:
            zero_length += 1
    custom_normals: Any = "api_unavailable"
    try:
        custom_normals = bool(mesh.has_custom_normals)
    except Exception:
        try:
            custom_normals = "corner_normals_present" if len(mesh.corner_normals) else False
        except Exception:
            pass
    determinant = obj.matrix_world.to_3x3().determinant()
    orientation_issues = int(topology["inconsistent_orientation_edge_count"])
    state = (
        "ok"
        if zero_length == 0 and non_finite == 0 and orientation_issues == 0
        else "issues_detected"
    )
    return {
        "state": state,
        "custom_split_normals": custom_normals,
        "zero_length_face_normals": zero_length,
        "non_finite_face_normals": non_finite,
        "inconsistent_orientation_edge_count": orientation_issues,
        "negative_world_transform_determinant": determinant < 0.0,
    }


def component_topology(
    mesh: bpy.types.Mesh,
    components: Sequence[Sequence[int]],
    face_uses: Mapping[tuple[int, int], Sequence[tuple[int, int, int]]],
    obj: bpy.types.Object,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vertex_component: dict[int, int] = {}
    for component_index, vertices in enumerate(components):
        for vertex_index in vertices:
            vertex_component[vertex_index] = component_index

    face_indices: list[list[int]] = [[] for _ in components]
    edge_counts = [0] * len(components)
    boundary_counts = [0] * len(components)
    multi_face_counts = [0] * len(components)
    loose_counts = [0] * len(components)
    inconsistent_counts = [0] * len(components)

    for polygon in mesh.polygons:
        if polygon.vertices:
            face_indices[vertex_component[int(polygon.vertices[0])]].append(polygon.index)
    for edge in mesh.edges:
        component_index = vertex_component[int(edge.vertices[0])]
        edge_counts[component_index] += 1
        key = tuple(sorted((int(edge.vertices[0]), int(edge.vertices[1]))))
        uses = face_uses.get(key, ())
        if len(uses) == 0:
            loose_counts[component_index] += 1
        elif len(uses) == 1:
            boundary_counts[component_index] += 1
        elif len(uses) > 2:
            multi_face_counts[component_index] += 1
        elif uses[0][1] == uses[1][1]:
            inconsistent_counts[component_index] += 1

    serializable: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    for component_index, vertex_indices in enumerate(components):
        world_points = [
            obj.matrix_world @ mesh.vertices[vertex_index].co
            for vertex_index in vertex_indices
        ]
        minimum, maximum = bbox_from_points(world_points)
        dimensions = dimensions_from_bbox(minimum, maximum)
        component_faces = face_indices[component_index]
        triangle_count = sum(
            max(0, len(mesh.polygons[face_index].vertices) - 2)
            for face_index in component_faces
        )
        topology = {
            "edge_count": edge_counts[component_index],
            "face_count": len(component_faces),
            "triangle_count": triangle_count,
            "loose_edge_count": loose_counts[component_index],
            "boundary_edge_count": boundary_counts[component_index],
            "multi_face_edge_count": multi_face_counts[component_index],
            "non_manifold_edge_count": (
                loose_counts[component_index]
                + boundary_counts[component_index]
                + multi_face_counts[component_index]
            ),
            "inconsistent_orientation_edge_count": inconsistent_counts[component_index],
        }
        descriptor = {
            "component_index": component_index,
            "vertex_count": len(vertex_indices),
            **topology,
            "bbox": {
                "min": as_float_list(minimum),
                "max": as_float_list(maximum),
            },
            "center": as_float_list(center_from_bbox(minimum, maximum)),
            "dimensions": as_float_list(dimensions),
        }
        serializable.append(descriptor)
        internal.append(
            {
                **descriptor,
                "vertex_indices": list(vertex_indices),
                "face_indices": list(component_faces),
            }
        )
    return serializable, internal


def inspect_object_dependencies(obj: bpy.types.Object) -> dict[str, Any]:
    references: list[dict[str, str]] = []
    target_attributes = (
        "object",
        "target",
        "origin",
        "mirror_object",
        "offset_object",
        "start_position_object",
    )
    for modifier in obj.modifiers:
        for attribute in target_attributes:
            try:
                target = getattr(modifier, attribute)
            except Exception:
                continue
            if isinstance(target, bpy.types.Object):
                references.append(
                    {
                        "source": f"modifier:{modifier.name}.{attribute}",
                        "target": target.name,
                    }
                )
    for constraint in obj.constraints:
        try:
            target = constraint.target
        except Exception:
            target = None
        if isinstance(target, bpy.types.Object):
            references.append(
                {
                    "source": f"constraint:{constraint.name}.target",
                    "target": target.name,
                }
            )
    has_drivers = bool(obj.animation_data and obj.animation_data.drivers)
    has_shape_keys = bool(getattr(obj.data, "shape_keys", None))
    return {
        "parent": obj.parent.name if obj.parent else None,
        "modifiers": [
            {"name": modifier.name, "type": modifier.type} for modifier in obj.modifiers
        ],
        "constraints": [
            {"name": constraint.name, "type": constraint.type}
            for constraint in obj.constraints
        ],
        "external_object_references": references,
        "has_drivers": has_drivers,
        "has_shape_keys": has_shape_keys,
    }


def analyze_mesh_object(
    obj: bpy.types.Object, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    mesh = obj.data
    mesh.calc_loop_triangles()
    edges = [(int(edge.vertices[0]), int(edge.vertices[1])) for edge in mesh.edges]
    components = connected_components(len(mesh.vertices), edges)
    topology, face_uses = analyze_edge_topology(mesh)
    component_report, component_internal = component_topology(
        mesh, components, face_uses, obj
    )
    detail_limit = int(config["topology"]["component_details_limit"])
    world_bbox = world_bbox_for_object(obj)
    local_min, local_max = bbox_from_points(vertex.co for vertex in mesh.vertices)
    report = {
        "object_name": obj.name,
        "mesh_data_name": mesh.name,
        "transforms": {
            "location": as_float_list(obj.location),
            "rotation_mode": obj.rotation_mode,
            "rotation_euler": as_float_list(obj.rotation_euler),
            "rotation_quaternion": as_float_list(obj.rotation_quaternion),
            "scale": as_float_list(obj.scale),
            "matrix_world": matrix_as_rows(obj.matrix_world),
        },
        "bbox_local": {
            "min": as_float_list(local_min),
            "max": as_float_list(local_max),
        },
        "bbox_world": world_bbox,
        "dimensions": as_float_list(obj.dimensions),
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "face_count": len(mesh.polygons),
        "triangle_count": len(mesh.loop_triangles),
        "materials": [
            material.name if material else None for material in mesh.materials
        ],
        "uv_maps": [layer.name for layer in mesh.uv_layers],
        "has_uv_map": bool(mesh.uv_layers),
        "disconnected_component_count": len(components),
        "component_details_truncated": len(component_report) > detail_limit,
        "components": component_report[:detail_limit],
        "topology": topology,
        "normals": analyze_normals(mesh, topology, obj),
        "dependencies": inspect_object_dependencies(obj),
    }
    internal = {
        "object": obj,
        "components": component_internal,
        "all_components_serializable": component_report,
    }
    return report, internal


def inspect_materials_and_textures() -> dict[str, Any]:
    materials: list[dict[str, Any]] = []
    missing_textures: list[dict[str, Any]] = []
    image_users: dict[str, list[str]] = defaultdict(list)
    for material in bpy.data.materials:
        image_nodes: list[dict[str, Any]] = []
        node_tree = getattr(material, "node_tree", None)
        if node_tree:
            for node in node_tree.nodes:
                if node.type != "TEX_IMAGE":
                    continue
                image = getattr(node, "image", None)
                if image is None:
                    entry = {
                        "material": material.name,
                        "node": node.name,
                        "reason": "image node has no image",
                    }
                    missing_textures.append(entry)
                    image_nodes.append({"node": node.name, "image": None})
                    continue
                image_nodes.append({"node": node.name, "image": image.name})
                image_users[image.name].append(material.name)
        materials.append(
            {
                "name": material.name,
                "has_node_tree": bool(node_tree),
                "image_nodes": image_nodes,
            }
        )

    images: list[dict[str, Any]] = []
    for image in bpy.data.images:
        packed = bool(getattr(image, "packed_file", None))
        try:
            packed = packed or bool(image.packed_files)
        except Exception:
            pass
        raw_path = str(image.filepath or "")
        resolved = str(Path(bpy.path.abspath(raw_path)).resolve()) if raw_path else ""
        source = str(getattr(image, "source", ""))
        exists = bool(resolved and Path(resolved).is_file())
        missing = source == "FILE" and not packed and not exists
        descriptor = {
            "name": image.name,
            "source": source,
            "filepath": raw_path,
            "resolved_filepath": resolved,
            "packed": packed,
            "exists": exists,
            "size": list(image.size),
            "materials": sorted(set(image_users.get(image.name, []))),
            "missing": missing,
        }
        images.append(descriptor)
        if missing:
            missing_textures.append(
                {
                    "image": image.name,
                    "filepath": raw_path,
                    "resolved_filepath": resolved,
                    "reason": "external image file is absent and not packed",
                }
            )
    return {
        "materials": materials,
        "images": images,
        "missing_textures": missing_textures,
    }


def collection_descriptor(collection: bpy.types.Collection) -> dict[str, Any]:
    return {
        "name": collection.name,
        "parent_collections": sorted(
            parent.name
            for parent in bpy.data.collections
            if collection.name in parent.children
        ),
        "child_collections": sorted(child.name for child in collection.children),
        "objects": sorted(obj.name for obj in collection.objects),
    }


def collection_tree(collection: bpy.types.Collection) -> dict[str, Any]:
    return {
        "name": collection.name,
        "objects": sorted(obj.name for obj in collection.objects),
        "children": [collection_tree(child) for child in collection.children],
    }


def object_tree(obj: bpy.types.Object) -> dict[str, Any]:
    return {
        "name": obj.name,
        "type": obj.type,
        "children": [object_tree(child) for child in obj.children],
    }


def inspect_scene_objects(imported_objects: Sequence[bpy.types.Object]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for obj in imported_objects:
        objects.append(
            {
                "name": obj.name,
                "type": obj.type,
                "data_name": obj.data.name if obj.data else None,
                "parent": obj.parent.name if obj.parent else None,
                "parent_type": obj.parent_type,
                "children": sorted(child.name for child in obj.children),
                "collections": sorted(collection.name for collection in obj.users_collection),
                "transforms": {
                    "location": as_float_list(obj.location),
                    "rotation_euler": as_float_list(obj.rotation_euler),
                    "scale": as_float_list(obj.scale),
                    "matrix_world": matrix_as_rows(obj.matrix_world),
                },
            }
        )
    roots = [obj for obj in imported_objects if obj.parent not in imported_objects]
    return {
        "collections": [
            collection_descriptor(collection) for collection in bpy.data.collections
        ],
        "collection_hierarchy": collection_tree(bpy.context.scene.collection),
        "objects": objects,
        "object_hierarchy": [object_tree(root) for root in roots],
    }


def primitive_from_mesh(
    mesh_report: Mapping[str, Any], mesh_index: int
) -> dict[str, Any]:
    bbox = mesh_report["bbox_world"]
    topology = mesh_report["topology"]
    return {
        "name": mesh_report["object_name"],
        "object_name": mesh_report["object_name"],
        "mesh_index": mesh_index,
        "source_kind": "mesh_object",
        "component_index": None,
        "center": bbox["center"],
        "dimensions": bbox["dimensions"],
        "bbox": {"min": bbox["min"], "max": bbox["max"]},
        "vertex_count": mesh_report["vertex_count"],
        "edge_count": mesh_report["edge_count"],
        "face_count": mesh_report["face_count"],
        "triangle_count": mesh_report["triangle_count"],
        **{
            key: topology[key]
            for key in (
                "loose_edge_count",
                "boundary_edge_count",
                "multi_face_edge_count",
                "non_manifold_edge_count",
                "inconsistent_orientation_edge_count",
            )
        },
    }


def build_candidate_primitives(
    mesh_reports: Sequence[Mapping[str, Any]],
    mesh_internal: Sequence[Mapping[str, Any]],
    scene_bbox: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    component_min_vertices = int(config["wheel_detection"]["min_component_vertices"])
    component_min_faces = int(config["wheel_detection"]["min_component_faces"])
    for mesh_index, mesh_report in enumerate(mesh_reports):
        object_primitive = primitive_from_mesh(mesh_report, mesh_index)
        primitives.append(object_primitive)
        if mesh_report["disconnected_component_count"] <= 1:
            continue
        for component in mesh_internal[mesh_index]["components"]:
            if (
                component["vertex_count"] < component_min_vertices
                or component["face_count"] < component_min_faces
            ):
                continue
            vertex_fraction = component["vertex_count"] / max(
                mesh_report["vertex_count"], 1
            )
            if vertex_fraction > 0.97:
                continue
            primitives.append(
                {
                    "name": (
                        f"{mesh_report['object_name']} component "
                        f"{component['component_index']}"
                    ),
                    "object_name": mesh_report["object_name"],
                    "mesh_index": mesh_index,
                    "source_kind": "disconnected_component",
                    "component_index": component["component_index"],
                    "center": component["center"],
                    "dimensions": component["dimensions"],
                    "bbox": component["bbox"],
                    "vertex_count": component["vertex_count"],
                    "edge_count": component["edge_count"],
                    "face_count": component["face_count"],
                    "triangle_count": component["triangle_count"],
                    "loose_edge_count": component["loose_edge_count"],
                    "boundary_edge_count": component["boundary_edge_count"],
                    "multi_face_edge_count": component["multi_face_edge_count"],
                    "non_manifold_edge_count": component["non_manifold_edge_count"],
                    "inconsistent_orientation_edge_count": component[
                        "inconsistent_orientation_edge_count"
                    ],
                }
            )

    repeats = repeat_counts(primitives)
    group_labels = repeat_group_labels(primitives)
    for primitive, repeated, group_label in zip(primitives, repeats, group_labels):
        primitive["repeat_group"] = group_label
        primitive["repeat_count"] = repeated
        primitive.update(
            score_wheel_candidate(
                primitive,
                scene_bbox,
                repeated,
                config["wheel_detection"],
            )
        )

    cluster_primitives = build_spatial_wheel_clusters(
        primitives, mesh_internal, scene_bbox, config
    )
    if cluster_primitives:
        cluster_primitives.sort(key=lambda item: (-float(item["score"]), item["name"]))
        return cluster_primitives[: int(config["wheel_detection"]["max_candidates"])]

    passed_objects = {
        primitive["object_name"]
        for primitive in primitives
        if primitive["source_kind"] == "mesh_object"
        and primitive["passes_threshold"]
    }
    candidates = [
        primitive
        for primitive in primitives
        if primitive["passes_threshold"]
        and not (
            primitive["source_kind"] == "disconnected_component"
            and primitive["object_name"] in passed_objects
        )
    ]
    candidates.sort(key=lambda item: (-float(item["score"]), item["name"]))

    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        center = Vector(candidate["center"])
        scale = max(candidate["dimensions"])
        duplicate = False
        for existing in deduplicated:
            existing_center = Vector(existing["center"])
            existing_scale = max(existing["dimensions"])
            if (center - existing_center).length <= 0.08 * max(scale, existing_scale):
                if candidate["object_name"] == existing["object_name"]:
                    duplicate = True
                    break
        if not duplicate:
            deduplicated.append(candidate)
    return deduplicated[: int(config["wheel_detection"]["max_candidates"])]


def build_spatial_wheel_clusters(
    primitives: Sequence[Mapping[str, Any]],
    mesh_internal: Sequence[Mapping[str, Any]],
    scene_bbox: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Promote repeated hub-like components into six complete wheel assemblies."""
    components = [
        dict(primitive)
        for primitive in primitives
        if primitive["source_kind"] == "disconnected_component"
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in components:
        grouped[str(component["repeat_group"])].append(component)

    scene_dims = [float(value) for value in scene_bbox["dimensions"]]
    scene_min = [float(value) for value in scene_bbox["min"]]
    seed_groups: list[tuple[float, list[dict[str, Any]]]] = []
    for members in grouped.values():
        if not 5 <= len(members) <= 8:
            continue
        centers = [member["center"] for member in members]
        x_span = max(center[0] for center in centers) - min(
            center[0] for center in centers
        )
        y_span = max(center[1] for center in centers) - min(
            center[1] for center in centers
        )
        mean_z_fraction = sum(
            (center[2] - scene_min[2]) / max(scene_dims[2], 1e-12)
            for center in centers
        ) / len(centers)
        distributed_like_six_wheels = (
            x_span >= scene_dims[0] * 0.55
            and y_span >= scene_dims[1] * 0.40
            and mean_z_fraction <= 0.40
        )
        if not distributed_like_six_wheels:
            continue
        average_score = sum(float(member["score"]) for member in members) / len(
            members
        )
        average_cylinder = sum(
            float(member["signals"]["cylinder_like"]) for member in members
        ) / len(members)
        seed_groups.append(
            (average_score + 0.05 * average_cylinder, members)
        )
    if not seed_groups:
        return []

    _, seeds = max(seed_groups, key=lambda entry: entry[0])
    seeds = sorted(seeds, key=lambda item: (item["center"][1], item["center"][0]))
    maximum_seed_size = max(max(seed["dimensions"]) for seed in seeds)
    cluster_radius = max(maximum_seed_size * 2.5, scene_dims[2] * 0.15)
    cluster_radius = min(cluster_radius, min(scene_dims[0], scene_dims[1]) * 0.18)

    all_components: list[dict[str, Any]] = []
    seed_mesh_indices = {int(seed["mesh_index"]) for seed in seeds}
    for mesh_index in seed_mesh_indices:
        object_name = mesh_internal[mesh_index]["object"].name
        for component in mesh_internal[mesh_index]["all_components_serializable"]:
            all_components.append(
                {
                    **component,
                    "object_name": object_name,
                    "mesh_index": mesh_index,
                    "bbox": component["bbox"],
                    "source_kind": "disconnected_component",
                }
            )

    assignments: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seed_vectors = [Vector(seed["center"]) for seed in seeds]
    for component in all_components:
        if int(component.get("face_count", 0)) == 0:
            continue
        component_size = max(float(value) for value in component["dimensions"])
        center = Vector(component["center"])
        distances = [(center - seed).length for seed in seed_vectors]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        seed = seeds[nearest]
        if int(component["mesh_index"]) != int(seed["mesh_index"]):
            continue
        axle_axis = int(seed["signals"]["axle_axis"])
        radial_axes = [axis for axis in range(3) if axis != axle_axis]
        seed_dimensions = [float(value) for value in seed["dimensions"]]
        radial_limit = max(
            max(seed_dimensions) * 2.10,
            scene_dims[2] * 0.13,
        )
        radial_limit = min(
            radial_limit,
            min(scene_dims[0], scene_dims[1]) * 0.13,
        )
        axle_half_width = max(
            seed_dimensions[axle_axis] * 6.0,
            scene_dims[axle_axis] * 0.08,
        )
        axle_half_width = min(axle_half_width, radial_limit * 0.90)
        delta = center - seed_vectors[nearest]
        component_dimensions = [
            float(value) for value in component["dimensions"]
        ]
        axial_envelope = (
            abs(delta[axle_axis]) + component_dimensions[axle_axis] * 0.5
        )
        radial_envelope = math.sqrt(
            sum(
                (
                    abs(delta[axis])
                    + component_dimensions[axis] * 0.5
                )
                ** 2
                for axis in radial_axes
            )
        )
        radial_center_distance = math.sqrt(
            sum(delta[axis] ** 2 for axis in radial_axes)
        )
        if (
            abs(delta[axle_axis]) <= axle_half_width
            and radial_center_distance <= radial_limit
            and axial_envelope <= axle_half_width * 1.15
            and radial_envelope <= radial_limit
            and component_size <= radial_limit * 1.75
        ):
            assignments[nearest].append(component)

    clusters: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        members = assignments.get(seed_index, [])
        if len(members) < 8:
            continue
        member_bboxes = [member["bbox"] for member in members]
        cluster_bbox = merge_bboxes(member_bboxes)
        primary = max(
            members,
            key=lambda member: (
                int(member["vertex_count"]),
                int(member["face_count"]),
            ),
        )
        aggregate = {
            key: sum(int(member[key]) for member in members)
            for key in ("vertex_count", "edge_count", "face_count", "triangle_count")
        }
        cluster = {
            "name": f"{seed['object_name']} wheel assembly {seed_index + 1}",
            "object_name": seed["object_name"],
            "mesh_index": seed["mesh_index"],
            "source_kind": "disconnected_component_cluster",
            "component_index": None,
            "component_indices": sorted(
                int(member["component_index"]) for member in members
            ),
            "assembly_component_count": len(members),
            "seed_component_index": int(seed["component_index"]),
            "seed_repeat_group": seed["repeat_group"],
            "cluster_radius": round(cluster_radius, 8),
            "center": cluster_bbox["center"],
            "dimensions": cluster_bbox["dimensions"],
            "bbox": {
                "min": cluster_bbox["min"],
                "max": cluster_bbox["max"],
            },
            **aggregate,
            "primary_surface_component": {
                "component_index": int(primary["component_index"]),
                "vertex_count": int(primary["vertex_count"]),
                "edge_count": int(primary["edge_count"]),
                "face_count": int(primary["face_count"]),
                "dimensions": primary["dimensions"],
            },
            **{
                key: primary[key]
                for key in (
                    "loose_edge_count",
                    "boundary_edge_count",
                    "multi_face_edge_count",
                    "non_manifold_edge_count",
                    "inconsistent_orientation_edge_count",
                )
            },
        }
        scored = score_wheel_candidate(
            cluster,
            scene_bbox,
            repeat_count=len(seeds),
            config=config["wheel_detection"],
        )
        cluster.update(scored)
        cluster["signals"]["spatial_cluster"] = True
        cluster["signals"]["assembly_component_count"] = len(members)
        cluster["signals"]["seed_component_index"] = int(seed["component_index"])
        if cluster["passes_threshold"]:
            clusters.append(cluster)
    return clusters


def dependency_assessment(
    primitive: Mapping[str, Any], mesh_report: Mapping[str, Any]
) -> dict[str, Any]:
    dependencies = mesh_report["dependencies"]
    blocking = (
        dependencies["external_object_references"]
        or dependencies["has_drivers"]
        or dependencies["has_shape_keys"]
    )
    if primitive["source_kind"] in {
        "disconnected_component",
        "disconnected_component_cluster",
    }:
        return {
            "verdict": "yes",
            "reason": (
                "La componente può essere copiata in una nuova mesh preservando "
                "coordinate world e materiali; l’oggetto sorgente non viene alterato."
            ),
            "source_dependencies": dependencies,
        }
    if blocking:
        return {
            "verdict": "conditional",
            "reason": (
                "L’oggetto è separato, ma contiene riferimenti, driver o shape keys "
                "da verificare/bake prima della duplicazione."
            ),
            "source_dependencies": dependencies,
        }
    parent_note = (
        " Il parent può essere rimosso preservando matrix_world."
        if dependencies["parent"]
        else ""
    )
    return {
        "verdict": "yes",
        "reason": (
            "Mesh data duplicabile; non risultano dipendenze esterne bloccanti."
            + parent_note
        ),
        "source_dependencies": dependencies,
    }


def boolean_assessment(primitive: Mapping[str, Any]) -> dict[str, Any]:
    edge_count = max(int(primitive.get("edge_count", 0)), 1)
    non_manifold = int(primitive.get("non_manifold_edge_count", 0))
    inconsistent = int(primitive.get("inconsistent_orientation_edge_count", 0))
    issue_ratio = (non_manifold + inconsistent) / edge_count
    if primitive["source_kind"] == "disconnected_component_cluster":
        verdict = "conditional"
        reason = (
            "La ruota è un cluster di molte patch disconnesse: il solver EXACT può "
            "essere provato su una copia, ma prima serve unire/chiudere o ricostruire "
            "localmente la pelle interessata dal foro."
        )
    elif int(primitive.get("face_count", 0)) == 0:
        verdict = "no"
        reason = "La candidata non contiene facce."
    elif non_manifold == 0 and inconsistent == 0:
        verdict = "yes"
        reason = "La superficie è manifold, chiusa e con orientamento coerente."
    elif issue_ratio <= 0.03 and inconsistent == 0:
        verdict = "conditional"
        reason = (
            "Pochi edge aperti/non-manifold: Boolean Difference è plausibile "
            "dopo una chiusura o pulizia locale."
        )
    else:
        verdict = "conditional"
        reason = (
            "La topologia presenta edge non-manifold/orientamento incoerente; "
            "testare una copia e riparare la zona prima del Boolean."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "edge_issue_ratio": round(issue_ratio, 6),
        "non_manifold_edge_count": non_manifold,
        "inconsistent_orientation_edge_count": inconsistent,
        "solver_recommendation": "EXACT",
    }


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def estimate_thickness(
    primitive: Mapping[str, Any],
    mesh_entry: Mapping[str, Any],
    sample_count: int,
    minimum_ratio: float,
) -> dict[str, Any]:
    obj = mesh_entry["object"]
    mesh = obj.data
    if primitive["source_kind"] == "disconnected_component":
        component = mesh_entry["components"][int(primitive["component_index"])]
        face_indices = component["face_indices"]
    elif primitive["source_kind"] == "disconnected_component_cluster":
        primary_index = int(
            primitive["primary_surface_component"]["component_index"]
        )
        face_indices = mesh_entry["components"][primary_index]["face_indices"]
    else:
        face_indices = list(range(len(mesh.polygons)))
    if not face_indices:
        return {
            "verdict": "no",
            "reason": "Nessuna faccia disponibile per stimare lo spessore.",
            "method": "bidirectional normal ray-cast",
            "samples_requested": sample_count,
            "samples_hit": 0,
        }

    vertices = [vertex.co.copy() for vertex in mesh.vertices]
    polygons = [list(mesh.polygons[index].vertices) for index in face_indices]
    try:
        bvh = BVHTree.FromPolygons(vertices, polygons, all_triangles=False)
    except Exception as exc:
        return {
            "verdict": "uncertain",
            "reason": f"BVH non costruibile: {exc!r}",
            "method": "bidirectional normal ray-cast",
            "samples_requested": sample_count,
            "samples_hit": 0,
        }

    step = max(1, len(face_indices) // max(sample_count, 1))
    chosen = face_indices[::step][:sample_count]
    local_bbox_size = max(
        (
            max(vertex.co[axis] for vertex in mesh.vertices)
            - min(vertex.co[axis] for vertex in mesh.vertices)
        )
        for axis in range(3)
    )
    epsilon = max(local_bbox_size * 1e-5, 1e-7)
    max_distance = max(local_bbox_size * 2.0, epsilon * 100.0)
    world_basis = obj.matrix_world.to_3x3()
    distances: list[float] = []
    for face_index in chosen:
        polygon = mesh.polygons[face_index]
        center = sum(
            (mesh.vertices[index].co for index in polygon.vertices),
            Vector((0.0, 0.0, 0.0)),
        ) / max(len(polygon.vertices), 1)
        normal = polygon.normal.normalized()
        for sign in (-1.0, 1.0):
            direction = normal * sign
            origin = center + direction * epsilon
            hit, _, _, _ = bvh.ray_cast(origin, direction, max_distance)
            if hit is None:
                continue
            delta_local = hit - center
            distance_world = (world_basis @ delta_local).length
            if distance_world > epsilon * 5.0:
                distances.append(float(distance_world))
                break

    p10 = percentile(distances, 0.10)
    median = percentile(distances, 0.50)
    diameter = max(float(value) for value in primitive["dimensions"])
    relative = p10 / diameter if p10 is not None and diameter > 1e-12 else None
    hit_ratio = len(distances) / max(len(chosen), 1)
    if p10 is None or hit_ratio < 0.10:
        verdict = "uncertain"
        reason = (
            "Troppi pochi raggi incontrano una seconda superficie: la pelle può "
            "essere monostrato o aperta."
        )
    elif relative is not None and relative >= minimum_ratio:
        verdict = "yes"
        reason = (
            "La distanza robusta tra superfici supera la soglia relativa richiesta "
            "per un foro visibile."
        )
    else:
        verdict = "no"
        reason = (
            "Lo spessore stimato è inferiore alla soglia relativa; serve Solidify "
            "o una ricostruzione locale prima del Boolean."
        )
    result = {
        "verdict": verdict,
        "reason": reason,
        "method": "bidirectional normal ray-cast (10th percentile)",
        "samples_requested": sample_count,
        "samples_evaluated": len(chosen),
        "samples_hit": len(distances),
        "hit_ratio": round(hit_ratio, 6),
        "p10_world_units": round(p10, 8) if p10 is not None else None,
        "median_world_units": round(median, 8) if median is not None else None,
        "p10_to_diameter_ratio": round(relative, 8) if relative is not None else None,
        "minimum_visible_ratio": minimum_ratio,
        "limitation": (
            "Stima euristica su mesh triangolata/poligonale, non misura CAD garantita."
        ),
    }
    if primitive["source_kind"] == "disconnected_component_cluster":
        primary_size = max(
            float(value)
            for value in primitive["primary_surface_component"]["dimensions"]
        )
        wheel_size = max(float(value) for value in primitive["dimensions"])
        if primary_size < wheel_size * 0.55:
            result["verdict"] = "uncertain"
            result["reason"] = (
                "La ruota non contiene una singola pelle connessa: il ray-cast "
                "misura solo la maggiore patch locale e non dimostra uno spessore "
                "continuo sufficiente per il Boolean sull’intera ruota."
            )
            result["fragmented_skin"] = True
    return result


def bbox_contains_center(
    outer: Mapping[str, Sequence[float]],
    center: Sequence[float],
    margin: float,
) -> bool:
    dimensions = dimensions_from_bbox(outer["min"], outer["max"])
    return all(
        float(outer["min"][axis]) - dimensions[axis] * margin
        <= float(center[axis])
        <= float(outer["max"][axis]) + dimensions[axis] * margin
        for axis in range(3)
    )


def grouser_assessment(
    primitive: Mapping[str, Any],
    mesh_report: Mapping[str, Any],
    mesh_entry: Mapping[str, Any],
    all_mesh_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if primitive["source_kind"] == "disconnected_component_cluster":
        component_indices = [int(index) for index in primitive["component_indices"]]
        components = [
            mesh_entry["all_components_serializable"][index]
            for index in component_indices
        ]
        primary_index = int(
            primitive["primary_surface_component"]["component_index"]
        )
        smaller = [
            component
            for component in components
            if component["component_index"] != primary_index
            and component["vertex_count"]
            <= primitive["primary_surface_component"]["vertex_count"] * 0.35
        ]
        if len(smaller) >= 10:
            return {
                "verdict": "conditional",
                "organization": "fragmented_components_same_mesh",
                "component_fragment_count": len(smaller),
                "selectable_count_estimate": None,
                "reason": (
                    "Pelle e rilievi sono frammentati in molte componenti della stessa "
                    "mesh, ma una componente non equivale necessariamente a un "
                    "grouser. Il singolo grouser è separabile dopo clustering spaziale "
                    "o una verifica/selezione manuale minima."
                ),
            }
        return {
            "verdict": "uncertain",
            "organization": "clustered_components_not_fully_resolved",
            "selectable_count_estimate": len(smaller),
            "reason": (
                "La ruota è un cluster disconnesso, ma il numero/rapporto delle "
                "componenti piccole non dimostra singoli grousers con sufficiente confidenza."
            ),
        }

    candidate_vertices = max(int(primitive["vertex_count"]), 1)
    nearby_components = []
    for component in mesh_entry["all_components_serializable"]:
        if primitive["source_kind"] == "disconnected_component" and (
            component["component_index"] == primitive["component_index"]
        ):
            continue
        relative_size = component["vertex_count"] / candidate_vertices
        if (
            0.001 <= relative_size <= 0.25
            and bbox_contains_center(primitive["bbox"], component["center"], 0.10)
        ):
            nearby_components.append(component)
    if len(nearby_components) >= 6:
        return {
            "verdict": "yes",
            "organization": "disconnected_components_same_mesh",
            "selectable_count_estimate": len(nearby_components),
            "reason": (
                "Sono presenti numerose componenti piccole disconnesse entro il "
                "volume ruota; singoli grousers sono separabili per connettività."
            ),
        }

    nearby_objects = []
    for other in all_mesh_reports:
        if other["object_name"] == primitive["object_name"]:
            continue
        if not bbox_contains_center(
            primitive["bbox"], other["bbox_world"]["center"], 0.10
        ):
            continue
        relative_size = other["vertex_count"] / candidate_vertices
        if 0.001 <= relative_size <= 0.25:
            nearby_objects.append(other["object_name"])
    if len(nearby_objects) >= 6:
        return {
            "verdict": "yes",
            "organization": "separate_objects",
            "selectable_count_estimate": len(nearby_objects),
            "nearby_objects": nearby_objects,
            "reason": (
                "Molte mesh piccole autonome ricadono nel volume ruota; i grousers "
                "sono probabilmente oggetti separati."
            ),
        }
    if mesh_report["disconnected_component_count"] == 1:
        return {
            "verdict": "conditional",
            "organization": "integrated_same_connected_mesh",
            "selectable_count_estimate": 0,
            "reason": (
                "Pelle e rilievi appartengono alla stessa componente connessa. "
                "Un grouser va selezionato per facce/curvatura, non con Select Linked."
            ),
        }
    return {
        "verdict": "uncertain",
        "organization": "mixed_or_not_resolved",
        "selectable_count_estimate": len(nearby_components) + len(nearby_objects),
        "reason": (
            "La connettività mostra più parti, ma la sola geometria non identifica "
            "con sicurezza quali siano grousers."
        ),
    }


def remediation_assessment(
    boolean: Mapping[str, Any],
    thickness: Mapping[str, Any],
    grousers: Mapping[str, Any],
) -> dict[str, Any]:
    issue_ratio = float(boolean.get("edge_issue_ratio", 1.0))
    return {
        "voxel_remesh": {
            "needed": bool(issue_ratio > 0.10),
            "reason": (
                "Solo come ultima risorsa per topologia gravemente non-manifold."
                if issue_ratio > 0.10
                else "Non necessario; preserverebbe peggio i dettagli dei grousers."
            ),
        },
        "retopology": {
            "needed": False,
            "reason": "Non richiesta per l’audit; rivalutare solo dopo prove Boolean locali.",
        },
        "simplified_reconstruction": {
            "needed": thickness.get("verdict") == "no",
            "reason": (
                "Indicata se la pelle è realmente monostrato/senza spessore."
                if thickness.get("verdict") == "no"
                else "Non necessaria sulla base della stima corrente."
            ),
        },
        "grouser_manual_face_selection": {
            "needed": grousers.get("organization")
            in {
                "integrated_same_connected_mesh",
                "fragmented_components_same_mesh",
            },
            "reason": (
                "Necessaria per verificare/raggruppare le patch che appartengono "
                "a un singolo grouser."
                if grousers.get("organization")
                == "fragmented_components_same_mesh"
                else "Necessaria quando pelle e grouser condividono la stessa componente."
            ),
        },
    }


def assess_candidates(
    primitives: Sequence[dict[str, Any]],
    mesh_reports: Sequence[Mapping[str, Any]],
    mesh_internal: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    assessed: list[dict[str, Any]] = []
    for index, primitive in enumerate(primitives, start=1):
        mesh_index = int(primitive["mesh_index"])
        mesh_report = mesh_reports[mesh_index]
        mesh_entry = mesh_internal[mesh_index]
        duplication = dependency_assessment(primitive, mesh_report)
        boolean = boolean_assessment(primitive)
        thickness = estimate_thickness(
            primitive,
            mesh_entry,
            int(config["topology"]["thickness_samples"]),
            float(config["topology"]["minimum_visible_thickness_ratio"]),
        )
        grousers = grouser_assessment(
            primitive, mesh_report, mesh_entry, mesh_reports
        )
        is_component_source = primitive["source_kind"] in {
            "disconnected_component",
            "disconnected_component_cluster",
        }
        automatic_separation = {
            "verdict": (
                "yes"
                if is_component_source
                else "not_needed"
            ),
            "reason": (
                "La componente ha un insieme di vertici/facce disconnesso e può "
                "essere ricostruita automaticamente in una nuova mesh."
                if is_component_source
                else "La candidata coincide già con un oggetto mesh."
            ),
        }
        assessed.append(
            {
                "id": f"wheel_candidate_{index:02d}",
                "object_name": primitive["object_name"],
                "source_kind": primitive["source_kind"],
                "component_index": primitive["component_index"],
                "component_indices": primitive.get("component_indices"),
                "assembly_component_count": primitive.get("assembly_component_count"),
                "primary_surface_component": primitive.get(
                    "primary_surface_component"
                ),
                "score": primitive["score"],
                "signals": primitive["signals"],
                "bbox": primitive["bbox"],
                "center": primitive["center"],
                "dimensions": primitive["dimensions"],
                "vertex_count": primitive["vertex_count"],
                "edge_count": primitive["edge_count"],
                "face_count": primitive["face_count"],
                "triangle_count": primitive["triangle_count"],
                "topology": {
                    key: primitive[key]
                    for key in (
                        "loose_edge_count",
                        "boundary_edge_count",
                        "multi_face_edge_count",
                        "non_manifold_edge_count",
                        "inconsistent_orientation_edge_count",
                    )
                },
                "duplication": duplication,
                "automatic_separation": automatic_separation,
                "boolean_difference": boolean,
                "skin_thickness": thickness,
                "grousers": grousers,
                "remediation": remediation_assessment(
                    boolean, thickness, grousers
                ),
                "renders": {"closeups": [], "wireframe": None},
                "_primitive": primitive,
            }
        )
    return assessed


def configure_world(scene: bpy.types.Scene, rgba: Sequence[float]) -> None:
    world = bpy.data.worlds.new("AuditWorld") if scene.world is None else scene.world
    scene.world = world
    node_tree = getattr(world, "node_tree", None)
    if node_tree is None:
        # Compatibility path for Blender versions where nodes are still opt-in.
        world.use_nodes = True
        node_tree = world.node_tree
    background = node_tree.nodes.get("Background") if node_tree else None
    if background:
        background.inputs["Color"].default_value = tuple(rgba)
        background.inputs["Strength"].default_value = 0.35


def configure_render(
    scene: bpy.types.Scene, config: Mapping[str, Any], selected_engine: str
) -> None:
    render = config["render"]
    scene.render.engine = selected_engine
    scene.render.resolution_x = int(render["resolution_x"])
    scene.render.resolution_y = int(render["resolution_y"])
    scene.render.resolution_percentage = int(render["resolution_percentage"])
    scene.render.image_settings.file_format = str(render["image_format"])
    scene.render.film_transparent = bool(render["transparent"])
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    try:
        scene.render.engine = selected_engine
        if hasattr(scene, "eevee"):
            for attribute in ("taa_render_samples", "taa_samples"):
                if hasattr(scene.eevee, attribute):
                    setattr(scene.eevee, attribute, int(render["samples"]))
    except Exception:
        pass
    configure_world(scene, render["background_rgba"])


def create_camera_and_lights(
    scene: bpy.types.Scene, scene_bbox: Mapping[str, Any]
) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    camera_data = bpy.data.cameras.new("AuditCamera")
    camera_data.lens = 52.0
    camera = bpy.data.objects.new("AuditCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    scale = max(float(value) for value in scene_bbox["dimensions"])
    scale = max(scale, 1.0)
    center = Vector(scene_bbox["center"])
    lights: list[bpy.types.Object] = []
    light_specs = [
        ("AuditKey", Vector((1.3, -1.1, 1.4)), 1100.0, 0.60),
        ("AuditFill", Vector((-1.2, -0.4, 0.8)), 700.0, 0.50),
        ("AuditRim", Vector((0.2, 1.4, 1.1)), 900.0, 0.45),
    ]
    for name, offset, energy, size_ratio in light_specs:
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy * max(scale * scale, 0.2)
        data.shape = "DISK"
        data.size = scale * size_ratio
        light = bpy.data.objects.new(name, data)
        light.location = center + offset * scale
        light.rotation_euler = (
            center - light.location
        ).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)
        lights.append(light)
    return camera, lights


def position_camera(
    camera: bpy.types.Object,
    target: Sequence[float],
    dimensions: Sequence[float],
    direction: Sequence[float],
    margin: float,
) -> None:
    target_vector = Vector(target)
    direction_vector = Vector(direction).normalized()
    radius = max(Vector(dimensions).length * 0.5, 0.01)
    camera_data = camera.data
    minimum_angle = min(float(camera_data.angle_x), float(camera_data.angle_y))
    distance = radius / max(math.sin(minimum_angle * 0.5), 0.1) * margin
    camera.location = target_vector + direction_vector * distance
    camera.rotation_euler = (
        target_vector - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    camera_data.clip_start = max(distance / 10000.0, 0.0001)
    camera_data.clip_end = max(distance * 10.0, radius * 50.0)


def render_still(
    scene: bpy.types.Scene,
    path: Path,
    logger: logging.Logger,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path)
    logger.info("Rendering %s", path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Render did not produce a file: {path}")


def set_imported_visibility(
    imported_mesh_objects: Sequence[bpy.types.Object],
    visible: Sequence[bpy.types.Object] | None,
) -> dict[str, bool]:
    state = {obj.name: bool(obj.hide_render) for obj in imported_mesh_objects}
    visible_names = None if visible is None else {obj.name for obj in visible}
    for obj in imported_mesh_objects:
        obj.hide_render = visible_names is not None and obj.name not in visible_names
    return state


def restore_visibility(
    imported_mesh_objects: Sequence[bpy.types.Object], state: Mapping[str, bool]
) -> None:
    for obj in imported_mesh_objects:
        if obj.name in state:
            obj.hide_render = bool(state[obj.name])


def make_component_object(
    candidate: Mapping[str, Any],
    mesh_entry: Mapping[str, Any],
) -> bpy.types.Object:
    primitive = candidate["_primitive"]
    if primitive["source_kind"] == "disconnected_component_cluster":
        component_indices = [int(index) for index in primitive["component_indices"]]
    else:
        component_indices = [int(primitive["component_index"])]
    source_obj = mesh_entry["object"]
    source_mesh = source_obj.data
    components = [mesh_entry["components"][index] for index in component_indices]
    vertex_indices = sorted(
        {
            vertex_index
            for component in components
            for vertex_index in component["vertex_indices"]
        }
    )
    face_indices = sorted(
        {
            face_index
            for component in components
            for face_index in component["face_indices"]
        }
    )
    remap = {old: new for new, old in enumerate(vertex_indices)}
    vertices = [source_mesh.vertices[index].co.copy() for index in vertex_indices]
    faces = [
        [remap[int(index)] for index in source_mesh.polygons[face_index].vertices]
        for face_index in face_indices
    ]
    mesh = bpy.data.meshes.new(f"{candidate['id']}_diagnostic_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{candidate['id']}_diagnostic", mesh)
    obj.matrix_world = source_obj.matrix_world.copy()
    bpy.context.scene.collection.objects.link(obj)
    for material in source_mesh.materials:
        mesh.materials.append(material)
    for new_polygon, source_face_index in zip(mesh.polygons, face_indices):
        source_material_index = source_mesh.polygons[source_face_index].material_index
        if source_material_index < len(mesh.materials):
            new_polygon.material_index = source_material_index
    return obj


def remove_temporary_object(obj: bpy.types.Object) -> None:
    mesh = obj.data if obj.type == "MESH" else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def candidate_directions(
    candidate: Mapping[str, Any], scene_bbox: Mapping[str, Any]
) -> list[tuple[str, Vector]]:
    scene_dimensions = [float(value) for value in scene_bbox["dimensions"]]
    scene_center = [float(value) for value in scene_bbox["center"]]
    lateral_axis = 0 if scene_dimensions[0] <= scene_dimensions[1] else 1
    longitudinal_axis = 1 - lateral_axis
    axes = [
        Vector((1.0, 0.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
        Vector((0.0, 0.0, 1.0)),
    ]
    lateral_sign = (
        1.0
        if float(candidate["center"][lateral_axis]) >= scene_center[lateral_axis]
        else -1.0
    )
    longitudinal_sign = (
        1.0
        if float(candidate["center"][longitudinal_axis])
        >= scene_center[longitudinal_axis]
        else -1.0
    )
    outboard = axes[lateral_axis] * lateral_sign
    longitudinal = axes[longitudinal_axis] * longitudinal_sign
    oblique = (outboard + longitudinal * 0.45 + axes[2] * 0.30).normalized()
    return [
        ("outboard", outboard),
        ("longitudinal", longitudinal),
        ("oblique", oblique),
    ]


def render_wireframe(
    scene: bpy.types.Scene,
    path: Path,
    logger: logging.Logger,
    available_engines: Sequence[str],
) -> None:
    original_engine = scene.render.engine
    original_type = getattr(scene.display.shading, "type", None)
    original_light = getattr(scene.display.shading, "light", None)
    original_color_type = getattr(scene.display.shading, "color_type", None)
    if "BLENDER_WORKBENCH" not in available_engines:
        logger.warning(
            "Workbench unavailable; topology diagnostic falls back to Eevee solid"
        )
        render_still(scene, path, logger)
        return
    try:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.type = "SOLID"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "SINGLE"
        scene.display.shading.single_color = (0.20, 0.48, 0.82)
        scene.display.shading.background_type = "WORLD"
        scene.display.shading.show_shadows = False
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "BOTH"
        scene.display.shading.curvature_ridge_factor = 2.0
        scene.display.shading.curvature_valley_factor = 1.5
        if hasattr(scene.display.shading, "show_object_outline"):
            scene.display.shading.show_object_outline = True
        render_still(scene, path, logger)
    finally:
        scene.render.engine = original_engine
        if original_type is not None:
            scene.display.shading.type = original_type
        if original_light is not None:
            scene.display.shading.light = original_light
        if original_color_type is not None:
            scene.display.shading.color_type = original_color_type


def render_diagnostics(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    scene_bbox: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    mesh_internal: Sequence[Mapping[str, Any]],
    imported_mesh_objects: Sequence[bpy.types.Object],
    paths: Mapping[str, Path],
    output_root: Path,
    config: Mapping[str, Any],
    engines: Sequence[str],
    logger: logging.Logger,
) -> dict[str, list[str]]:
    margin = float(config["render"]["camera_margin"])
    rover_views = [
        ("positive_x", (1.0, 0.0, 0.0)),
        ("negative_x", (-1.0, 0.0, 0.0)),
        ("positive_y", (0.0, 1.0, 0.0)),
        ("negative_y", (0.0, -1.0, 0.0)),
        ("top", (0.0, 0.0, 1.0)),
        ("bottom", (0.0, 0.0, -1.0)),
    ]
    rover_renders: list[str] = []
    for name, direction in rover_views:
        position_camera(
            camera,
            scene_bbox["center"],
            scene_bbox["dimensions"],
            direction,
            margin,
        )
        path = paths["rover_renders"] / f"rover_{name}.png"
        render_still(scene, path, logger)
        rover_renders.append(safe_relative(path, output_root))

    topology_renders: list[str] = []
    position_camera(
        camera,
        scene_bbox["center"],
        scene_bbox["dimensions"],
        (1.0, -1.0, 0.65),
        margin,
    )
    general_wire = paths["topology_renders"] / "rover_wireframe_oblique.png"
    render_wireframe(scene, general_wire, logger, engines)
    topology_renders.append(safe_relative(general_wire, output_root))

    for candidate in candidates:
        primitive = candidate["_primitive"]
        mesh_entry = mesh_internal[int(primitive["mesh_index"])]
        temporary = None
        contextual_cluster = (
            candidate["source_kind"] == "disconnected_component_cluster"
        )
        if candidate["source_kind"] in {
            "disconnected_component",
            "disconnected_component_cluster",
        }:
            temporary = make_component_object(candidate, mesh_entry)
            visible = [mesh_entry["object"]] if contextual_cluster else []
        else:
            visible = [mesh_entry["object"]]
        visibility = set_imported_visibility(imported_mesh_objects, visible)
        if temporary:
            temporary.hide_render = contextual_cluster
        else:
            mesh_entry["object"].hide_render = False
        candidate_dir = paths["candidate_renders"] / candidate["id"]
        try:
            for view_name, direction in candidate_directions(
                candidate, scene_bbox
            ):
                position_camera(
                    camera,
                    candidate["center"],
                    candidate["dimensions"],
                    direction,
                    margin * 1.15,
                )
                path = candidate_dir / f"{candidate['id']}_{view_name}.png"
                render_still(scene, path, logger)
                candidate["renders"]["closeups"].append(
                    safe_relative(path, output_root)
                )
            if bool(config["diagnostics"]["save_candidate_wireframes"]):
                if contextual_cluster and temporary:
                    for imported in imported_mesh_objects:
                        imported.hide_render = True
                    temporary.hide_render = False
                _, wire_direction = candidate_directions(
                    candidate, scene_bbox
                )[-1]
                position_camera(
                    camera,
                    candidate["center"],
                    candidate["dimensions"],
                    wire_direction,
                    margin,
                )
                wire_path = candidate_dir / f"{candidate['id']}_wireframe.png"
                render_wireframe(scene, wire_path, logger, engines)
                relative = safe_relative(wire_path, output_root)
                candidate["renders"]["wireframe"] = relative
                topology_renders.append(relative)
        finally:
            restore_visibility(imported_mesh_objects, visibility)
            if temporary:
                remove_temporary_object(temporary)
    return {
        "rover_renders": rover_renders,
        "topology_renders": topology_renders,
    }


def write_reports(
    report: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> None:
    json_path = paths["reports"] / "audit.json"
    markdown_path = paths["reports"] / "audit.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    with markdown_path.open("w", encoding="utf-8") as stream:
        stream.write(make_markdown_report(report))


def strip_internal_candidate_fields(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    serializable = []
    for candidate in candidates:
        serializable.append(
            {key: value for key, value in candidate.items() if not key.startswith("_")}
        )
    return serializable


def failure_report(
    schema_version: str,
    asset: Path,
    started_at: str,
    error: str,
    warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "status": "failed",
        "started_utc": started_at,
        "completed_utc": utc_now(),
        "asset": {"path": str(asset)},
        "warnings": list(warnings),
        "errors": [error],
        "classification": {
            "category": "D",
            "title": "Asset non auditabile",
            "confidence": 1.0,
            "rationale": error,
        },
        "wheel_detection": {
            "candidate_count": 0,
            "candidates": [],
            "six_wheels_separate": "unknown",
            "organization": "unknown",
        },
        "scene": {"collections": [], "objects": []},
        "meshes": [],
        "materials": {"materials": [], "images": [], "missing_textures": []},
        "artifacts": {},
    }


def run_audit(args: argparse.Namespace) -> int:
    started_monotonic = time.perf_counter()
    started_at = utc_now()
    asset = args.asset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    paths = create_output_tree(output_dir)
    warnings: list[str] = []
    errors: list[str] = []

    try:
        config = load_config(args.config.expanduser().resolve())
    except Exception as exc:
        config = {"schema_version": "1.0", "logging": {"level": "INFO"}}
        logger = setup_logger(paths["logs"] / "audit.log", "INFO")
        error = f"Configuration load failed: {exc}"
        logger.exception(error)
        write_reports(
            failure_report("1.0", asset, started_at, error, warnings), paths
        )
        return 2

    logger = setup_logger(
        paths["logs"] / "audit.log", str(config["logging"]["level"])
    )
    logger.info("Starting Blender GLB audit")
    logger.info("Blender: %s", bpy.app.version_string)
    logger.info("Asset: %s", asset)
    logger.info("Output: %s", output_dir)

    if not asset.is_file():
        error = f"Asset not found: {asset}"
        logger.error(error)
        write_reports(
            failure_report(
                str(config["schema_version"]), asset, started_at, error, warnings
            ),
            paths,
        )
        return 2
    if asset.suffix.lower() != ".glb":
        error = f"Expected a GLB asset, got: {asset}"
        logger.error(error)
        write_reports(
            failure_report(
                str(config["schema_version"]), asset, started_at, error, warnings
            ),
            paths,
        )
        return 2

    checksum_before = sha256_file(asset)
    asset_stat_before = asset.stat()
    try:
        logger.info("Resetting Blender to an empty factory scene")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if not hasattr(bpy.ops.import_scene, "gltf"):
            raise RuntimeError(
                "bpy.ops.import_scene.gltf is unavailable in this Blender build"
            )
        logger.info("Importing GLB (read-only source)")
        result = bpy.ops.import_scene.gltf(filepath=str(asset))
        if "FINISHED" not in result:
            raise RuntimeError(f"GLB importer returned {result}")
    except Exception as exc:
        error = f"GLB import failed: {exc}"
        logger.exception(error)
        errors.append(error)
        report = failure_report(
            str(config["schema_version"]), asset, started_at, error, warnings
        )
        report["asset"].update(
            {
                "size_bytes": asset_stat_before.st_size,
                "sha256_before": checksum_before,
            }
        )
        write_reports(report, paths)
        return 3

    imported_objects = list(bpy.context.scene.objects)
    imported_mesh_objects = [obj for obj in imported_objects if obj.type == "MESH"]
    if not imported_mesh_objects:
        warning = "GLB import completed but produced no mesh objects"
        warnings.append(warning)
        logger.warning(warning)

    scene = bpy.context.scene
    engines = available_render_engines(scene)
    selected_engine = select_render_engine(
        scene, config["render"]["engine_preference"], engines
    )
    gpu_info = inspect_gpu()
    if not gpu_info["available"]:
        warning = (
            "No hardware GPU renderer detected; Eevee may be unavailable or use a "
            "software/fallback backend."
        )
        warnings.append(warning)
        logger.warning(warning)
    else:
        logger.info(
            "GPU: %s | %s | %s",
            gpu_info["backend"],
            gpu_info["vendor"],
            gpu_info["renderer"],
        )

    logger.info("Inspecting scene, meshes, materials and connectivity")
    scene_inventory = inspect_scene_objects(imported_objects)
    mesh_reports: list[dict[str, Any]] = []
    mesh_internal: list[dict[str, Any]] = []
    for obj in imported_mesh_objects:
        try:
            mesh_report, internal = analyze_mesh_object(obj, config)
            mesh_reports.append(mesh_report)
            mesh_internal.append(internal)
            logger.info(
                "Mesh %s: %d vertices, %d faces, %d components",
                obj.name,
                mesh_report["vertex_count"],
                mesh_report["face_count"],
                mesh_report["disconnected_component_count"],
            )
        except Exception as exc:
            message = f"Mesh analysis failed for {obj.name}: {exc}"
            warnings.append(message)
            logger.exception(message)

    scene_bbox = merge_bboxes(
        [mesh_report["bbox_world"] for mesh_report in mesh_reports]
    )
    scene_inventory["world_bbox"] = scene_bbox
    scene_inventory["counts"] = {
        object_type: sum(obj.type == object_type for obj in imported_objects)
        for object_type in sorted({obj.type for obj in imported_objects})
    }
    material_report = inspect_materials_and_textures()
    if material_report["missing_textures"]:
        warning = (
            f"{len(material_report['missing_textures'])} missing texture reference(s)"
        )
        warnings.append(warning)
        logger.warning(warning)

    primitives = build_candidate_primitives(
        mesh_reports, mesh_internal, scene_bbox, config
    )
    candidates = assess_candidates(
        primitives, mesh_reports, mesh_internal, config
    )
    if not candidates:
        warning = (
            "No wheel candidate passed the multi-signal geometric threshold; "
            "classification will be D."
        )
        warnings.append(warning)
        logger.warning(warning)
    else:
        logger.info("Detected %d wheel candidate(s)", len(candidates))

    distinct_object_candidates = {
        candidate["object_name"]
        for candidate in candidates
        if candidate["source_kind"] == "mesh_object"
    }
    component_candidates = [
        candidate
        for candidate in candidates
        if candidate["source_kind"]
        in {"disconnected_component", "disconnected_component_cluster"}
    ]
    if len(distinct_object_candidates) >= 6:
        six_separate = "yes"
        organization = "six_or_more_separate_mesh_objects"
    elif len(candidates) >= 6 and component_candidates:
        six_separate = "no"
        organization = "wheels_embedded_as_disconnected_components"
    else:
        six_separate = "uncertain"
        organization = "mixed_or_fewer_than_six_confident_candidates"

    configure_render(scene, config, selected_engine)
    camera, _ = create_camera_and_lights(scene, scene_bbox)
    render_artifacts = render_diagnostics(
        scene,
        camera,
        scene_bbox,
        candidates,
        mesh_internal,
        imported_mesh_objects,
        paths,
        output_dir,
        config,
        engines,
        logger,
    )

    diagnostic_blend = paths["diagnostics"] / "imported_asset.blend"
    if config["diagnostics"]["pack_resources_in_blend"]:
        try:
            bpy.ops.file.pack_all()
            logger.info("Packed imported resources into diagnostic .blend")
        except Exception as exc:
            warning = f"Could not pack all diagnostic resources: {exc}"
            warnings.append(warning)
            logger.warning(warning)
    scene.render.engine = selected_engine
    bpy.ops.wm.save_as_mainfile(filepath=str(diagnostic_blend), check_existing=False)
    logger.info("Saved diagnostic blend: %s", diagnostic_blend)

    checksum_after = sha256_file(asset)
    asset_stat_after = asset.stat()
    unchanged = (
        checksum_before == checksum_after
        and asset_stat_before.st_size == asset_stat_after.st_size
        and asset_stat_before.st_mtime_ns == asset_stat_after.st_mtime_ns
    )
    if not unchanged:
        error = "Source asset checksum/metadata changed during audit"
        errors.append(error)
        logger.error(error)

    serializable_candidates = strip_internal_candidate_fields(candidates)
    classification = classify_asset(serializable_candidates)
    gltf_needed = bool(material_report["missing_textures"])
    report = {
        "schema_version": str(config["schema_version"]),
        "status": "completed" if not errors else "completed_with_errors",
        "started_utc": started_at,
        "completed_utc": utc_now(),
        "duration_seconds": round(time.perf_counter() - started_monotonic, 3),
        "asset": {
            "path": str(asset),
            "name": asset.name,
            "size_bytes": asset_stat_before.st_size,
            "mtime_ns_before": asset_stat_before.st_mtime_ns,
            "mtime_ns_after": asset_stat_after.st_mtime_ns,
            "sha256_before": checksum_before,
            "sha256_after": checksum_after,
            "unchanged": unchanged,
            "source_writes_performed": False,
            "import_operator": "bpy.ops.import_scene.gltf",
        },
        "environment": {
            "blender_version": bpy.app.version_string,
            "blender_version_tuple": list(bpy.app.version),
            "blender_build_hash": bpy.app.build_hash.decode(
                "utf-8", errors="replace"
            )
            if isinstance(bpy.app.build_hash, bytes)
            else str(bpy.app.build_hash),
            "python_version": sys.version,
            "platform": platform.platform(),
            "render_engines_available": engines,
            "selected_render_engine": selected_engine,
            "render_resolution": {
                "width": int(config["render"]["resolution_x"]),
                "height": int(config["render"]["resolution_y"]),
                "aspect_ratio": "4:3",
            },
            "gpu": gpu_info,
        },
        "scene": scene_inventory,
        "meshes": mesh_reports,
        "materials": material_report,
        "wheel_detection": {
            "expected_wheel_count": 6,
            "candidate_count": len(serializable_candidates),
            "candidate_threshold": config["wheel_detection"]["candidate_threshold"],
            "six_wheels_separate": six_separate,
            "organization": organization,
            "strategy": [
                "object name keywords (non-exclusive)",
                "cylindrical bounding-box proportions",
                "low scene position",
                "peripheral XY position",
                "repeated geometry/topology signatures",
                "disconnected component analysis",
            ],
            "candidates": serializable_candidates,
        },
        "classification": classification,
        "glb_import_integrity": {
            "missing_texture_count": len(material_report["missing_textures"]),
            "material_count": len(material_report["materials"]),
            "image_count": len(material_report["images"]),
            "evidence_of_relevant_data_loss": gltf_needed,
            "try_gltf_required": gltf_needed,
            "gltf_reason": (
                "Una o più immagini esterne/non packed non sono risolvibili dal GLB "
                "importato. Il corrispondente glTF sarebbe necessario solo per "
                "ispezionare URI e buffer separati e stabilire quale risorsa manca; "
                "non viene convertito o importato automaticamente."
                if gltf_needed
                else "Nessuna perdita rilevante osservata; non è necessario provare il glTF."
            ),
        },
        "artifacts": {
            "json_report": "reports/audit.json",
            "markdown_report": "reports/audit.md",
            "diagnostic_blend": safe_relative(diagnostic_blend, output_dir),
            "log": "logs/audit.log",
            "topology_render_method": (
                "Blender Workbench solid cavity/outline; no geometry changes"
            ),
            **render_artifacts,
        },
        "warnings": warnings,
        "errors": errors,
    }
    write_reports(report, paths)
    logger.info(
        "Audit completed: category %s (%s)",
        classification["category"],
        classification["title"],
    )
    logger.info("JSON report: %s", paths["reports"] / "audit.json")
    return 0 if not errors else 5


def main() -> int:
    try:
        args = parse_args()
        return run_audit(args)
    except SystemExit:
        raise
    except Exception as exc:
        traceback.print_exc()
        print(f"FATAL: unhandled audit error: {exc}", file=sys.stderr)
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
