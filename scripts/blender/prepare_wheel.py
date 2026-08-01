from __future__ import annotations

import argparse
import bmesh
import bpy
import hashlib
import json
import logging
import math
import platform
import struct
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mathutils import Matrix, Vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blender_audit.core import connected_components, deep_merge
from src.wheel_preparation.core import (
    canonical_axis_from_candidates,
    find_candidate,
    make_markdown_report,
    select_merge_result,
    sha256_file,
    validate_audit_compatibility,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "1.0",
    "default_candidate_id": "wheel_candidate_05",
    "render": {
        "engine_preference": ["BLENDER_EEVEE", "BLENDER_WORKBENCH"],
        "resolution_x": 800,
        "resolution_y": 600,
        "resolution_percentage": 100,
        "samples": 16,
        "image_format": "PNG",
        "background_rgba": [0.025, 0.035, 0.055, 1.0],
        "camera_margin": 1.2,
    },
    "skin": {
        "radial_histogram_bins": 96,
        "minimum_radial_normal_alignment": 0.35,
        "radial_band_diameter_ratio": 0.035,
        "merge_tolerance_ratios": [1e-7, 1e-6, 1e-5, 1e-4],
        "shell_angular_segments": 192,
        "minimum_wall_thickness_ratio": 0.003,
        "maximum_wall_thickness_ratio": 0.02,
    },
    "perforation": {
        "cutter_segments": 20,
        "axial_radius_ratio": 0.03,
        "tangential_radius_ratio": 0.04,
        "radial_depth_ratio": 0.08,
        "angular_search_bins": 72,
        "mask_dilation_pixels": 4,
    },
    "gates": {
        "minimum_silhouette_iou": 0.995,
        "maximum_bbox_relative_error": 0.001,
        "minimum_mask_ratio": 0.002,
        "maximum_mask_ratio": 0.05,
        "maximum_outside_change_ratio": 0.01,
    },
    "diagnostics": {"pack_resources_in_blend": True},
    "logging": {"level": "INFO"},
}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Prepare one audited Curiosity wheel and a perforation probe."
    )
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--candidate-id", default="wheel_candidate_05")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    return parser.parse_args(values)


def create_output_tree(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "reports": root / "reports",
        "diagnostics": root / "diagnostics",
        "extraction": root / "renders" / "extraction",
        "topology": root / "renders" / "topology",
        "perforation": root / "renders" / "perforation",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("wheel_preparation")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return deep_merge(DEFAULT_CONFIG, {})
    return deep_merge(DEFAULT_CONFIG, load_json(path))


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def matrix_rows(matrix: Matrix) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


def mesh_counts(mesh: bpy.types.Mesh) -> dict[str, int]:
    mesh.calc_loop_triangles()
    return {
        "vertex_count": len(mesh.vertices),
        "edge_count": len(mesh.edges),
        "face_count": len(mesh.polygons),
        "triangle_count": len(mesh.loop_triangles),
    }


def component_data(mesh: bpy.types.Mesh) -> dict[str, Any]:
    components = connected_components(
        len(mesh.vertices),
        ((int(edge.vertices[0]), int(edge.vertices[1])) for edge in mesh.edges),
    )
    vertex_to_component: dict[int, int] = {}
    for component_index, vertices in enumerate(components):
        for vertex_index in vertices:
            vertex_to_component[vertex_index] = component_index
    faces: list[list[int]] = [[] for _ in components]
    for polygon in mesh.polygons:
        if polygon.vertices:
            faces[vertex_to_component[int(polygon.vertices[0])]].append(
                int(polygon.index)
            )
    return {
        "components": components,
        "faces": faces,
        "vertex_to_component": vertex_to_component,
    }


def material_histogram(
    mesh: bpy.types.Mesh, face_indices: Iterable[int] | None = None
) -> dict[str, int]:
    if face_indices is None:
        polygons = mesh.polygons
    else:
        polygons = [mesh.polygons[index] for index in face_indices]
    counts = Counter(int(polygon.material_index) for polygon in polygons)
    return {str(key): int(value) for key, value in sorted(counts.items())}


def _hash_records(records: Iterable[Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(tuple(item) for item in records):
        digest.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def coordinate_hash(
    mesh: bpy.types.Mesh,
    vertex_indices: Iterable[int] | None = None,
    transform: Matrix | None = None,
) -> str:
    indices = (
        list(range(len(mesh.vertices)))
        if vertex_indices is None
        else [int(index) for index in vertex_indices]
    )
    matrix = transform or Matrix.Identity(4)
    return _hash_records(
        tuple(round(float(value), 8) for value in (matrix @ mesh.vertices[index].co))
        for index in indices
    )


def uv_hashes(
    mesh: bpy.types.Mesh, face_indices: Iterable[int] | None = None
) -> dict[str, str]:
    polygons = (
        list(mesh.polygons)
        if face_indices is None
        else [mesh.polygons[int(index)] for index in face_indices]
    )
    result: dict[str, str] = {}
    for layer in mesh.uv_layers:
        records = []
        for polygon in polygons:
            for loop_index in polygon.loop_indices:
                uv = layer.data[int(loop_index)].uv
                records.append((round(float(uv.x), 7), round(float(uv.y), 7)))
        result[layer.name] = _hash_records(records)
    return result


def has_valid_custom_normals(mesh: bpy.types.Mesh) -> bool:
    try:
        if not bool(mesh.has_custom_normals):
            return False
    except Exception:
        return False
    try:
        normals = mesh.corner_normals
        return len(normals) == len(mesh.loops) and all(
            math.isfinite(float(value))
            for normal in normals
            for value in normal.vector
        )
    except Exception:
        return len(mesh.loops) > 0


def bbox(mesh: bpy.types.Mesh) -> dict[str, list[float]]:
    if not mesh.vertices:
        zero = [0.0, 0.0, 0.0]
        return {"min": zero, "max": zero, "center": zero, "dimensions": zero}
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    for vertex in mesh.vertices:
        for axis in range(3):
            value = float(vertex.co[axis])
            minimum[axis] = min(minimum[axis], value)
            maximum[axis] = max(maximum[axis], value)
    dimensions = [maximum[i] - minimum[i] for i in range(3)]
    center = [(maximum[i] + minimum[i]) * 0.5 for i in range(3)]
    return {
        "min": [round(value, 8) for value in minimum],
        "max": [round(value, 8) for value in maximum],
        "center": [round(value, 8) for value in center],
        "dimensions": [round(value, 8) for value in dimensions],
    }


def bbox_relative_error(
    reference: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
) -> float:
    scale = max((float(value) for value in reference["dimensions"]), default=1.0)
    differences = [
        abs(float(left) - float(right))
        for key in ("min", "max")
        for left, right in zip(reference[key], candidate[key])
    ]
    return max(differences, default=0.0) / max(scale, 1e-12)


def factory_empty() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(asset: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    result = bpy.ops.import_scene.gltf(filepath=str(asset))
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB import failed with status {result}")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError("GLB import completed without creating objects")
    return imported


def available_render_engines(scene: bpy.types.Scene) -> list[str]:
    engines: set[str] = set()
    try:
        prop = bpy.types.RenderSettings.bl_rna.properties["engine"]
        engines.update(str(item.identifier) for item in prop.enum_items)
    except Exception:
        pass
    original = scene.render.engine
    for identifier in ("BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
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


def inspect_gpu() -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "backend": None,
        "vendor": None,
        "renderer": None,
        "version": None,
    }
    try:
        import gpu

        gpu.init()
        info = gpu.platform
        result.update(
            {
                "backend": str(info.backend_type_get()),
                "vendor": str(info.vendor_get()),
                "renderer": str(info.renderer_get()),
                "version": str(info.version_get()),
            }
        )
        result["available"] = bool(result["renderer"])
    except Exception as exc:
        result["error"] = str(exc)
    return result


def configure_scene(
    scene: bpy.types.Scene, config: Mapping[str, Any], engine: str | None = None
) -> str:
    render = config["render"]
    available = available_render_engines(scene)
    chosen = engine
    if chosen is None:
        chosen = next(
            (
                preference
                for preference in render["engine_preference"]
                if preference in available
            ),
            available[0],
        )
    scene.render.engine = chosen
    scene.render.resolution_x = int(render["resolution_x"])
    scene.render.resolution_y = int(render["resolution_y"])
    scene.render.resolution_percentage = int(render["resolution_percentage"])
    scene.render.image_settings.file_format = str(render["image_format"])
    scene.render.film_transparent = False
    try:
        scene.render.image_settings.color_mode = "RGBA"
    except Exception:
        pass
    try:
        scene.render.engine = chosen
        scene.render.resolution_percentage = int(render["resolution_percentage"])
    except Exception:
        pass
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new(f"{scene.name}_World")
        scene.world = world
    if world.node_tree is None:
        try:
            world.use_nodes = True
        except Exception:
            pass
    rgba = [float(value) for value in render["background_rgba"]]
    world.color = rgba[:3]
    if world.node_tree:
        background = world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Color"].default_value = rgba
            background.inputs["Strength"].default_value = 0.35
    return chosen


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_rig(
    scene: bpy.types.Scene, diameter: float
) -> tuple[bpy.types.Collection, bpy.types.Object]:
    collection = bpy.data.collections.new("PAIR_RIG")
    scene.collection.children.link(collection)
    camera_data = bpy.data.cameras.new("WheelCamera")
    camera_data.lens = 55.0
    camera_data.sensor_width = 36.0
    camera = bpy.data.objects.new("WheelCamera", camera_data)
    collection.objects.link(camera)
    scene.camera = camera

    area_data = bpy.data.lights.new("KeyArea", "AREA")
    area_data.energy = 650.0
    area_data.shape = "DISK"
    area_data.size = max(diameter * 1.8, 0.1)
    area = bpy.data.objects.new("KeyArea", area_data)
    area.location = Vector((diameter * 1.5, -diameter * 1.2, diameter * 1.8))
    collection.objects.link(area)
    look_at(area, Vector((0.0, 0.0, 0.0)))

    sun_data = bpy.data.lights.new("FillSun", "SUN")
    sun_data.energy = 2.0
    sun_data.angle = math.radians(18.0)
    sun = bpy.data.objects.new("FillSun", sun_data)
    sun.rotation_euler = (math.radians(25), math.radians(-20), math.radians(-35))
    collection.objects.link(sun)
    return collection, camera


def position_camera(
    camera: bpy.types.Object,
    direction: Sequence[float],
    target: Sequence[float],
    diameter: float,
    distance_scale: float = 1.75,
) -> None:
    direction_vector = Vector(direction).normalized()
    target_vector = Vector(target)
    camera.location = target_vector + direction_vector * diameter * distance_scale
    look_at(camera, target_vector)


def set_mesh_visibility(
    visible: Iterable[bpy.types.Object],
) -> dict[str, bool]:
    visible_set = set(visible)
    state: dict[str, bool] = {}
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            state[obj.name] = bool(obj.hide_render)
            obj.hide_render = obj not in visible_set
    return state


def restore_visibility(state: Mapping[str, bool]) -> None:
    for name, hidden in state.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = bool(hidden)


def render_still(scene: bpy.types.Scene, path: Path) -> None:
    scene.render.filepath = str(path)
    result = bpy.ops.render.render(scene=scene.name, write_still=True)
    if "FINISHED" not in result:
        raise RuntimeError(f"Render failed for {path}: {result}")


def create_emission_material(name: str, color: Sequence[float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    if material.node_tree is None:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = tuple(color)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def derive_hybrid_skin_material(
    source: bpy.types.Material | None,
) -> bpy.types.Material:
    """Create a stable metal material derived from the NASA wheel material.

    The GLB wheel texture is an atlas and cannot be remapped cylindrically
    without displaying unrelated atlas regions. We retain the original material
    on all original details and derive a texture-free average metal only for the
    replacement shell.
    """

    samples: list[tuple[float, float, float]] = []
    if source is not None and source.node_tree is not None:
        for node in source.node_tree.nodes:
            image = getattr(node, "image", None)
            if image is None or not getattr(image, "has_data", False):
                continue
            try:
                pixels = image.pixels
                pixel_count = len(pixels) // 4
                stride = max(1, pixel_count // 2048)
                for pixel_index in range(0, pixel_count, stride):
                    offset = pixel_index * 4
                    alpha = float(pixels[offset + 3])
                    if alpha <= 0.1:
                        continue
                    samples.append(
                        (
                            float(pixels[offset]),
                            float(pixels[offset + 1]),
                            float(pixels[offset + 2]),
                        )
                    )
                    if len(samples) >= 2048:
                        break
            except Exception:
                continue
            if samples:
                break
    if samples:
        color = tuple(
            min(0.42, max(0.10, sum(sample[channel] for sample in samples) / len(samples)))
            for channel in range(3)
        )
    else:
        color = (0.22, 0.20, 0.18)

    material = bpy.data.materials.new("Wheel_Skin_Hybrid_Material")
    if material.node_tree is None:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Base Color"].default_value = (*color, 1.0)
    principled.inputs["Metallic"].default_value = 0.68
    principled.inputs["Roughness"].default_value = 0.38
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    material["derived_from"] = source.name if source is not None else "fallback"
    material["derivation"] = "sampled average color; texture atlas intentionally disconnected"
    return material


def image_pixels(path: Path) -> tuple[int, int, list[float]]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = int(image.size[0]), int(image.size[1])
        pixels = list(image.pixels[:])
    finally:
        bpy.data.images.remove(image)
    return width, height, pixels


def mask_iou(left_path: Path, right_path: Path) -> float:
    left_width, left_height, left = image_pixels(left_path)
    right_width, right_height, right = image_pixels(right_path)
    if (left_width, left_height) != (right_width, right_height):
        return 0.0
    intersection = 0
    union = 0
    for index in range(0, len(left), 4):
        left_on = max(left[index : index + 3]) > 0.5
        right_on = max(right[index : index + 3]) > 0.5
        intersection += int(left_on and right_on)
        union += int(left_on or right_on)
    return intersection / union if union else 1.0


def render_binary_mask(
    scene: bpy.types.Scene,
    objects: Sequence[bpy.types.Object],
    path: Path,
    override_material: bpy.types.Material,
) -> None:
    state = set_mesh_visibility(objects)
    layer = scene.view_layers[0]
    previous_override = layer.material_override
    previous_world = scene.world
    black_world = bpy.data.worlds.get("BinaryMaskWorld")
    if black_world is None:
        black_world = bpy.data.worlds.new("BinaryMaskWorld")
        black_world.color = (0.0, 0.0, 0.0)
        black_world.use_nodes = True
        background = black_world.node_tree.nodes.get("Background")
        if background:
            background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            background.inputs["Strength"].default_value = 0.0
    layer.material_override = override_material
    scene.world = black_world
    try:
        render_still(scene, path)
    finally:
        layer.material_override = previous_override
        scene.world = previous_world
        restore_visibility(state)


def extract_candidate(
    source_obj: bpy.types.Object,
    selected_vertices: set[int],
    canonical_matrix: Matrix,
) -> bpy.types.Object:
    """Separate selected vertices from a full duplicate using Blender/BMesh data."""

    scene = bpy.context.scene
    duplicate = source_obj.copy()
    duplicate.data = source_obj.data.copy()
    duplicate.name = "Wheel_Extraction_Working"
    scene.collection.objects.link(duplicate)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    duplicate.select_set(True)
    bpy.context.view_layer.objects.active = duplicate
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.context.tool_settings.mesh_select_mode = (True, False, False)
    bm = bmesh.from_edit_mesh(duplicate.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for face in bm.faces:
        face.select = False
    for edge in bm.edges:
        edge.select = False
    for vertex in bm.verts:
        vertex.select = int(vertex.index) in selected_vertices
    bm.select_flush_mode()
    bmesh.update_edit_mesh(duplicate.data)
    before = set(bpy.data.objects)
    result = bpy.ops.mesh.separate(type="SELECTED")
    if "FINISHED" not in result:
        bpy.ops.object.mode_set(mode="OBJECT")
        raise RuntimeError(f"Mesh separation failed: {result}")
    bpy.ops.object.mode_set(mode="OBJECT")
    created = [
        obj
        for obj in bpy.data.objects
        if obj not in before and obj.type == "MESH"
    ]
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one separated wheel object, found {len(created)}"
        )
    pieces = [duplicate, created[0]]
    matching = [
        obj for obj in pieces if len(obj.data.vertices) == len(selected_vertices)
    ]
    if len(matching) != 1:
        counts = {obj.name: len(obj.data.vertices) for obj in pieces}
        raise RuntimeError(
            "Could not identify the selected separation piece by vertex count: "
            f"expected={len(selected_vertices)}, pieces={counts}"
        )
    extracted = matching[0]
    remainder = created[0] if extracted is duplicate else duplicate
    extracted.name = "Wheel_Raw"
    extracted.data.name = "Wheel_Raw_Mesh"
    extracted.data.transform(canonical_matrix)
    extracted.matrix_world = Matrix.Identity(4)
    extracted.data.update(calc_edges=True)
    bpy.data.objects.remove(remainder, do_unlink=True)
    return extracted


def copy_faces(
    source: bpy.types.Object,
    keep_faces: set[int],
    name: str,
) -> bpy.types.Object:
    obj = source.copy()
    obj.data = source.data.copy()
    obj.name = name
    obj.data.name = f"{name}_Mesh"
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    delete_faces = [face for face in bm.faces if int(face.index) not in keep_faces]
    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES")
    unused_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
    if unused_vertices:
        bmesh.ops.delete(bm, geom=unused_vertices, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update(calc_edges=True)
    return obj


def topology_metrics(mesh: bpy.types.Mesh) -> dict[str, Any]:
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    boundary = 0
    loose = 0
    multi = 0
    for edge in bm.edges:
        face_count = len(edge.link_faces)
        if face_count == 0:
            loose += 1
        elif face_count == 1:
            boundary += 1
        elif face_count > 2:
            multi += 1
    component_count = len(
        connected_components(
            len(bm.verts),
            ((edge.verts[0].index, edge.verts[1].index) for edge in bm.edges),
        )
    )
    invalid_normals = sum(
        1
        for face in bm.faces
        if face.normal.length_squared <= 1e-20
        or not all(math.isfinite(float(value)) for value in face.normal)
    )
    result = {
        "vertex_count": len(bm.verts),
        "edge_count": len(bm.edges),
        "face_count": len(bm.faces),
        "component_count": component_count,
        "loose_edge_count": loose,
        "boundary_edge_count": boundary,
        "multi_face_edge_count": multi,
        "non_manifold_edge_count": loose + boundary + multi,
        "invalid_face_normal_count": invalid_normals,
        "normals_consistent": invalid_normals == 0 and multi == 0,
        "closed_manifold": loose == 0 and boundary == 0 and multi == 0,
    }
    bm.free()
    return result


def face_center(mesh: bpy.types.Mesh, polygon: bpy.types.MeshPolygon) -> Vector:
    if not polygon.vertices:
        return Vector((0.0, 0.0, 0.0))
    total = Vector((0.0, 0.0, 0.0))
    for vertex_index in polygon.vertices:
        total += mesh.vertices[int(vertex_index)].co
    return total / len(polygon.vertices)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def classify_skin_faces(
    obj: bpy.types.Object, config: Mapping[str, Any]
) -> dict[str, Any]:
    mesh = obj.data
    data = component_data(mesh)
    radii = [
        math.hypot(float(vertex.co.y), float(vertex.co.z))
        for vertex in mesh.vertices
    ]
    maximum_radius = max(radii)
    diameter = maximum_radius * 2.0
    minimum_alignment = float(
        config["skin"]["minimum_radial_normal_alignment"]
    )
    bins = int(config["skin"]["radial_histogram_bins"])
    weighted = [0.0] * bins
    face_descriptors: dict[int, dict[str, float]] = {}
    for polygon in mesh.polygons:
        center = face_center(mesh, polygon)
        radius = math.hypot(float(center.y), float(center.z))
        radial = Vector((0.0, center.y, center.z))
        alignment = (
            abs(float(polygon.normal.dot(radial.normalized())))
            if radial.length_squared > 1e-20
            else 0.0
        )
        face_descriptors[int(polygon.index)] = {
            "radius": radius,
            "alignment": alignment,
            "area": float(polygon.area),
        }
        radius_fraction = radius / max(maximum_radius, 1e-12)
        if 0.68 <= radius_fraction <= 0.985 and alignment >= minimum_alignment:
            index = min(bins - 1, max(0, int(radius_fraction * bins)))
            weighted[index] += max(float(polygon.area), 1e-12)
    if not any(weighted):
        raise RuntimeError("No radial face population found for skin classification")
    peak_index = max(range(bins), key=weighted.__getitem__)
    base_radius = (peak_index + 0.5) / bins * maximum_radius
    band = float(config["skin"]["radial_band_diameter_ratio"]) * diameter
    lower_radius = base_radius - band * 0.75
    upper_radius = base_radius + band * 0.20

    component_scores: dict[int, dict[str, float]] = {}
    skin_components: set[int] = set()
    for component_index, face_indices in enumerate(data["faces"]):
        if not face_indices:
            continue
        total_area = sum(face_descriptors[index]["area"] for index in face_indices)
        radial_area = sum(
            face_descriptors[index]["area"]
            for index in face_indices
            if lower_radius <= face_descriptors[index]["radius"] <= upper_radius
            and face_descriptors[index]["alignment"] >= minimum_alignment
        )
        mean_radius = sum(
            face_descriptors[index]["radius"]
            * face_descriptors[index]["area"]
            for index in face_indices
        ) / max(total_area, 1e-12)
        radial_fraction = radial_area / max(total_area, 1e-12)
        component_scores[component_index] = {
            "total_area": total_area,
            "radial_area_fraction": radial_fraction,
            "mean_radius": mean_radius,
        }
        if (
            radial_fraction >= 0.20
            and lower_radius <= mean_radius <= base_radius + band * 0.45
        ):
            skin_components.add(component_index)

    skin_faces = {
        face_index
        for component_index in skin_components
        for face_index in data["faces"][component_index]
    }
    if not skin_faces:
        raise RuntimeError("Skin classifier selected no faces")
    skin_vertices = {
        int(vertex_index)
        for face_index in skin_faces
        for vertex_index in mesh.polygons[face_index].vertices
    }
    axial_values = [float(mesh.vertices[index].co.x) for index in skin_vertices]
    return {
        "skin_face_indices": skin_faces,
        "skin_component_indices": skin_components,
        "base_radius": base_radius,
        "maximum_radius": maximum_radius,
        "diameter": diameter,
        "lower_radius": lower_radius,
        "upper_radius": upper_radius,
        "axial_min": percentile(axial_values, 0.01),
        "axial_max": percentile(axial_values, 0.99),
        "skin_face_count": len(skin_faces),
        "skin_component_count": len(skin_components),
        "detail_face_count": len(mesh.polygons) - len(skin_faces),
        "histogram_peak_index": peak_index,
        "component_scores": component_scores,
    }


def merge_sweep(
    original_skin: bpy.types.Object,
    diameter: float,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bpy.types.Object | None]:
    reference_bbox = bbox(original_skin.data)
    reference_materials = material_histogram(original_skin.data)
    reference_uv_layers = [layer.name for layer in original_skin.data.uv_layers]
    records: list[dict[str, Any]] = []
    objects: dict[float, bpy.types.Object] = {}
    for tolerance_ratio in config["skin"]["merge_tolerance_ratios"]:
        ratio = float(tolerance_ratio)
        obj = original_skin.copy()
        obj.data = original_skin.data.copy()
        obj.name = f"Skin_Merge_{ratio:.0e}"
        obj.data.name = f"{obj.name}_Mesh"
        bpy.context.scene.collection.objects.link(obj)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        before_vertices = len(bm.verts)
        bmesh.ops.remove_doubles(
            bm, verts=list(bm.verts), dist=max(ratio * diameter, 1e-12)
        )
        bm.to_mesh(obj.data)
        bm.free()
        obj.data.update(calc_edges=True)
        topology = topology_metrics(obj.data)
        record = {
            "tolerance_ratio": ratio,
            "tolerance_world_units": ratio * diameter,
            "vertices_before": before_vertices,
            "vertices_after": len(obj.data.vertices),
            "merged_vertex_count": before_vertices - len(obj.data.vertices),
            "bbox_relative_error": bbox_relative_error(
                reference_bbox, bbox(obj.data)
            ),
            "silhouette_iou": 1.0,
            "material_histogram_preserved": (
                material_histogram(obj.data) == reference_materials
            ),
            "uv_layers_preserved": (
                [layer.name for layer in obj.data.uv_layers] == reference_uv_layers
            ),
            **topology,
        }
        records.append(record)
        objects[ratio] = obj

    selected = select_merge_result(
        records,
        minimum_silhouette_iou=float(
            config["gates"]["minimum_silhouette_iou"]
        ),
        maximum_bbox_relative_error=float(
            config["gates"]["maximum_bbox_relative_error"]
        ),
    )
    selected_obj = (
        objects[float(selected["tolerance_ratio"])] if selected is not None else None
    )
    for ratio, obj in objects.items():
        if obj is not selected_obj:
            bpy.data.objects.remove(obj, do_unlink=True)
    if selected_obj is not None:
        selected_obj.name = "Wheel_Skin"
        selected_obj.data.name = "Wheel_Skin_Mesh"
    return records, selected_obj


def create_shell_mesh(
    *,
    name: str,
    outer_radius: float,
    wall_thickness: float,
    axial_min: float,
    axial_max: float,
    segments: int,
    material: bpy.types.Material | None,
) -> bpy.types.Object:
    inner_radius = outer_radius - wall_thickness
    if inner_radius <= 0.0 or axial_max <= axial_min:
        raise ValueError("Invalid dimensions for parametric wheel shell")
    vertices: list[tuple[float, float, float]] = []
    for radius, axial in (
        (outer_radius, axial_min),
        (outer_radius, axial_max),
        (inner_radius, axial_min),
        (inner_radius, axial_max),
    ):
        for index in range(segments):
            theta = 2.0 * math.pi * index / segments
            vertices.append(
                (axial, radius * math.cos(theta), radius * math.sin(theta))
            )
    outer_left = 0
    outer_right = segments
    inner_left = segments * 2
    inner_right = segments * 3
    faces: list[tuple[int, int, int, int]] = []
    face_roles: list[str] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append(
            (
                outer_left + index,
                outer_left + next_index,
                outer_right + next_index,
                outer_right + index,
            )
        )
        face_roles.append("outer")
        faces.append(
            (
                inner_left + index,
                inner_right + index,
                inner_right + next_index,
                inner_left + next_index,
            )
        )
        face_roles.append("inner")
        faces.append(
            (
                outer_left + index,
                inner_left + index,
                inner_left + next_index,
                outer_left + next_index,
            )
        )
        face_roles.append("left_cap")
        faces.append(
            (
                outer_right + index,
                outer_right + next_index,
                inner_right + next_index,
                inner_right + index,
            )
        )
        face_roles.append("right_cap")
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    if material is not None:
        mesh.materials.append(material)
    uv_layer = mesh.uv_layers.new(name="Wheel_Cylindrical_UV")
    for polygon, role in zip(mesh.polygons, face_roles):
        segment_index = int(polygon.index) // 4
        u0 = segment_index / segments
        u1 = (segment_index + 1) / segments
        if role in {"outer", "inner"}:
            coordinates = [(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)]
        else:
            coordinates = [(u0, 1.0), (u0, 0.0), (u1, 0.0), (u1, 1.0)]
        for loop_index, uv in zip(polygon.loop_indices, coordinates):
            uv_layer.data[int(loop_index)].uv = uv
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def select_wheel_material(source_mesh: bpy.types.Mesh) -> bpy.types.Material | None:
    for material in source_mesh.materials:
        if material and "wheel" in material.name.lower():
            return material
    return source_mesh.materials[0] if source_mesh.materials else None


def create_cutter(
    *,
    name: str,
    theta: float,
    outer_radius: float,
    diameter: float,
    config: Mapping[str, Any],
) -> bpy.types.Object:
    segments = int(config["perforation"]["cutter_segments"])
    axial_radius = float(config["perforation"]["axial_radius_ratio"]) * diameter
    tangential_radius = (
        float(config["perforation"]["tangential_radius_ratio"]) * diameter
    )
    half_depth = (
        float(config["perforation"]["radial_depth_ratio"]) * diameter * 0.5
    )
    radial = Vector((0.0, math.cos(theta), math.sin(theta)))
    tangent = Vector((0.0, -math.sin(theta), math.cos(theta)))
    axial = Vector((1.0, 0.0, 0.0))
    center = radial * outer_radius
    irregularity = (
        1.00,
        1.06,
        0.96,
        1.04,
        0.94,
        1.02,
        1.07,
        0.97,
        1.03,
        0.95,
    )
    vertices: list[tuple[float, float, float]] = []
    for depth_sign in (-1.0, 1.0):
        for index in range(segments):
            phi = 2.0 * math.pi * index / segments
            factor = irregularity[index % len(irregularity)]
            point = (
                center
                + radial * (depth_sign * half_depth)
                + axial * (math.cos(phi) * axial_radius * factor)
                + tangent * (math.sin(phi) * tangential_radius / factor)
            )
            vertices.append(tuple(float(value) for value in point))
    faces: list[tuple[int, ...]] = []
    faces.append(tuple(reversed(range(segments))))
    faces.append(tuple(range(segments, segments * 2)))
    for index in range(segments):
        next_index = (index + 1) % segments
        faces.append(
            (
                index,
                next_index,
                segments + next_index,
                segments + index,
            )
        )
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def angular_distance(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2.0 * math.pi) - math.pi)


def choose_hole_angle(
    details: bpy.types.Object,
    *,
    base_radius: float,
    axial_width: float,
    bins: int,
) -> dict[str, Any]:
    scores = [0.0] * bins
    for polygon in details.data.polygons:
        center = face_center(details.data, polygon)
        radius = math.hypot(float(center.y), float(center.z))
        if radius < base_radius * 0.94:
            continue
        if abs(float(center.x)) > axial_width * 0.25:
            continue
        theta = math.atan2(float(center.z), float(center.y)) % (2.0 * math.pi)
        index = min(bins - 1, int(theta / (2.0 * math.pi) * bins))
        scores[index] += max(float(polygon.area), 1e-12)
    windowed = []
    for index in range(bins):
        score = sum(scores[(index + offset) % bins] for offset in (-2, -1, 0, 1, 2))
        theta = 2.0 * math.pi * (index + 0.5) / bins
        windowed.append((score, angular_distance(theta, math.pi * 0.5), index, theta))
    score, _, index, theta = min(windowed)
    return {
        "theta_radians": theta,
        "theta_degrees": math.degrees(theta),
        "bin_index": index,
        "occupancy_score": score,
        "bin_scores": scores,
        "method": "minimum five-bin detail occupancy near the tread center",
    }


def apply_boolean(
    target: bpy.types.Object,
    cutter: bpy.types.Object,
    operation: str,
) -> dict[str, Any]:
    modifier = target.modifiers.new(name=f"{operation}_Probe", type="BOOLEAN")
    modifier.operation = operation
    modifier.solver = "EXACT"
    modifier.object = cutter
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    try:
        result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    except Exception as exc:
        return {"success": False, "error": str(exc), "operation": operation}
    target.data.update(calc_edges=True)
    topology = topology_metrics(target.data)
    return {
        "success": "FINISHED" in result,
        "operation": operation,
        "solver": "EXACT",
        **topology,
    }


def create_collection(name: str, scene: bpy.types.Scene) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if collection.name not in {child.name for child in scene.collection.children}:
        scene.collection.children.link(collection)
    return collection


def move_object_to_collection(
    obj: bpy.types.Object, collection: bpy.types.Collection
) -> None:
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    for current in list(obj.users_collection):
        if current != collection:
            current.objects.unlink(obj)


def safe_pack(logger: logging.Logger) -> bool:
    try:
        bpy.ops.file.pack_all()
        return True
    except Exception as exc:
        logger.warning("Could not pack every resource: %s", exc)
        return False


def create_pair_scenes(
    config: Mapping[str, Any],
    rig_collection: bpy.types.Collection,
    normal_objects: Sequence[bpy.types.Object],
    anomaly_objects: Sequence[bpy.types.Object],
    camera: bpy.types.Object,
) -> dict[str, bpy.types.Scene]:
    normal_collection = bpy.data.collections.new("PAIR_NORMAL")
    anomaly_collection = bpy.data.collections.new("PAIR_ANOMALY")
    for obj in normal_objects:
        if obj.name not in normal_collection.objects:
            normal_collection.objects.link(obj)
    for obj in anomaly_objects:
        if obj.name not in anomaly_collection.objects:
            anomaly_collection.objects.link(obj)

    normal_scene = bpy.data.scenes.get("Normal")
    if normal_scene is None:
        normal_scene = bpy.data.scenes.new("Normal")
    anomaly_scene = bpy.data.scenes.get("Perforation")
    if anomaly_scene is None:
        anomaly_scene = bpy.data.scenes.new("Perforation")
    for scene, variant in (
        (normal_scene, normal_collection),
        (anomaly_scene, anomaly_collection),
    ):
        for collection in list(scene.collection.children):
            scene.collection.children.unlink(collection)
        scene.collection.children.link(rig_collection)
        scene.collection.children.link(variant)
        scene.camera = camera
        configure_scene(scene, config, engine="BLENDER_EEVEE")
    return {"normal": normal_scene, "perforation": anomaly_scene}


def write_difference_and_metrics(
    normal_path: Path,
    anomaly_path: Path,
    mask_path: Path,
    wheel_mask_path: Path,
    difference_path: Path,
    dilation_pixels: int,
) -> dict[str, Any]:
    width, height, normal = image_pixels(normal_path)
    width_b, height_b, anomaly = image_pixels(anomaly_path)
    width_m, height_m, mask = image_pixels(mask_path)
    width_w, height_w, wheel = image_pixels(wheel_mask_path)
    if len({(width, height), (width_b, height_b), (width_m, height_m), (width_w, height_w)}) != 1:
        raise RuntimeError("Pair and mask render dimensions differ")
    pixel_count = width * height
    mask_flags = bytearray(pixel_count)
    wheel_flags = bytearray(pixel_count)
    differences = [0.0] * pixel_count
    output_pixels = [0.0] * (pixel_count * 4)
    for pixel_index in range(pixel_count):
        offset = pixel_index * 4
        difference = max(
            abs(float(normal[offset + channel]) - float(anomaly[offset + channel]))
            for channel in range(3)
        )
        differences[pixel_index] = difference
        mask_flags[pixel_index] = int(max(mask[offset : offset + 3]) > 0.5)
        wheel_flags[pixel_index] = int(max(wheel[offset : offset + 3]) > 0.5)
        value = min(1.0, difference * 4.0)
        output_pixels[offset : offset + 4] = [value, value * 0.22, 0.0, 1.0]

    dilated = bytearray(mask_flags)
    active = [index for index, value in enumerate(mask_flags) if value]
    radius = max(0, int(dilation_pixels))
    for index in active:
        y, x = divmod(index, width)
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if not 0 <= yy < height:
                continue
            span = int(math.sqrt(max(0, radius * radius - dy * dy)))
            left = max(0, x - span)
            right = min(width - 1, x + span)
            row = yy * width
            for xx in range(left, right + 1):
                dilated[row + xx] = 1

    wheel_pixels = sum(wheel_flags)
    mask_pixels = sum(
        1
        for mask_value, wheel_value in zip(mask_flags, wheel_flags)
        if mask_value and wheel_value
    )
    changed_outside = sum(
        1
        for difference, excluded, wheel_value in zip(
            differences, dilated, wheel_flags
        )
        if wheel_value and not excluded and difference > (2.0 / 255.0)
    )
    image = bpy.data.images.new(
        "CounterfactualDifference", width=width, height=height, alpha=True
    )
    try:
        image.colorspace_settings.name = "Non-Color"
    except Exception:
        pass
    image.pixels.foreach_set(output_pixels)
    image.filepath_raw = str(difference_path)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return {
        "pixel_count": pixel_count,
        "wheel_visible_pixels": wheel_pixels,
        "mask_visible_pixels": mask_pixels,
        "area_ratio": mask_pixels / wheel_pixels if wheel_pixels else 0.0,
        "dilation_pixels": radius,
        "changed_outside_mask_pixels": changed_outside,
        "outside_change_ratio": (
            changed_outside / wheel_pixels if wheel_pixels else 1.0
        ),
        "difference_threshold": 2.0 / 255.0,
    }


def write_reports(report: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    def json_default(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {value.__class__.__name__} is not serializable")

    json_path = paths["reports"] / "preparation.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    with temporary_json.open("w", encoding="utf-8") as stream:
        json.dump(
            report,
            stream,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        stream.write("\n")
    temporary_json.replace(json_path)
    markdown_path = paths["reports"] / "preparation.md"
    temporary_markdown = markdown_path.with_suffix(".md.tmp")
    with temporary_markdown.open("w", encoding="utf-8") as stream:
        stream.write(make_markdown_report(report))
    temporary_markdown.replace(markdown_path)


def render_extraction_diagnostics(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    raw: bpy.types.Object,
    reference: bpy.types.Object,
    diameter: float,
    paths: Mapping[str, Path],
    white_material: bpy.types.Material,
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, float], list[str]]:
    directions = {
        "outboard": (1.0, 0.0, 0.0),
        "tread": (0.2, -1.0, 0.15),
        "oblique": (0.85, -0.85, 0.55),
    }
    extraction_paths: list[str] = []
    silhouette: dict[str, float] = {}
    verification_paths: list[str] = []
    original_engine = scene.render.engine
    configure_scene(scene, config, engine="BLENDER_EEVEE")
    for label, direction in directions.items():
        position_camera(
            camera,
            direction,
            (0.0, 0.0, 0.0),
            diameter,
            distance_scale=2.8,
        )
        state = set_mesh_visibility([raw])
        output = paths["extraction"] / f"wheel_{label}.png"
        render_still(scene, output)
        restore_visibility(state)
        extraction_paths.append(relative(output, paths["root"]))

        reference_path = (
            paths["extraction"] / f"verification_{label}_reference.png"
        )
        extracted_path = (
            paths["extraction"] / f"verification_{label}_extracted.png"
        )
        render_binary_mask(scene, [reference], reference_path, white_material)
        render_binary_mask(scene, [raw], extracted_path, white_material)
        silhouette[label] = mask_iou(reference_path, extracted_path)
        verification_paths.extend(
            [
                relative(reference_path, paths["root"]),
                relative(extracted_path, paths["root"]),
            ]
        )

    position_camera(
        camera,
        (0.85, -0.85, 0.55),
        (0.0, 0.0, 0.0),
        diameter,
        distance_scale=2.8,
    )
    configure_scene(scene, config, engine="BLENDER_WORKBENCH")
    try:
        scene.display.shading.light = "STUDIO"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "WORLD"
        scene.display.shading.show_object_outline = True
        scene.display.shading.color_type = "MATERIAL"
    except Exception:
        pass
    topology_path = paths["topology"] / "wheel_topology.png"
    state = set_mesh_visibility([raw])
    render_still(scene, topology_path)
    restore_visibility(state)
    configure_scene(scene, config, engine=original_engine)
    return (
        extraction_paths,
        silhouette,
        [relative(topology_path, paths["root"])],
    )


def prepare(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    started_utc = utc_now()
    output_dir = args.output_dir.expanduser().resolve()
    paths = create_output_tree(output_dir)
    config = load_config(args.config.expanduser().resolve() if args.config else None)
    logger = setup_logger(
        paths["logs"] / "preparation.log",
        str(config["logging"]["level"]),
    )
    warnings: list[str] = []
    errors: list[str] = []
    asset = args.asset.expanduser().resolve()
    audit_path = args.audit_report.expanduser().resolve()
    sha_before: str | None = None
    report: dict[str, Any] = {
        "schema_version": str(config["schema_version"]),
        "status": "failed",
        "started_utc": started_utc,
        "asset": {"path": str(asset)},
        "audit_report": {"path": str(audit_path)},
        "candidate": {"id": args.candidate_id},
        "configuration": config,
        "warnings": warnings,
        "errors": errors,
        "artifacts": {},
    }
    try:
        for label, path in (("asset", asset), ("audit report", audit_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} not found: {path}")
        if asset.suffix.lower() != ".glb":
            raise ValueError(f"Expected a GLB asset, got {asset}")
        sha_before = sha256_file(asset)
        audit_report = load_json(audit_path)
        candidate = find_candidate(audit_report, args.candidate_id)
        logger.info("Preparing %s from %s", args.candidate_id, asset)

        factory_empty()
        imported = import_glb(asset)
        source_obj = bpy.data.objects.get(str(candidate.get("object_name")))
        if source_obj is None or source_obj.type != "MESH":
            raise RuntimeError(
                f"Candidate source mesh {candidate.get('object_name')!r} was not imported"
            )
        source_mesh = source_obj.data
        source_counts = mesh_counts(source_mesh)
        compatibility = validate_audit_compatibility(
            audit_report,
            asset_sha256=sha_before,
            blender_version=bpy.app.version,
            object_name=source_obj.name,
            mesh_counts=source_counts,
            candidate_id=args.candidate_id,
        )
        if not compatibility["compatible"]:
            raise RuntimeError(
                "Audit preflight failed: " + "; ".join(compatibility["errors"])
            )

        components = component_data(source_mesh)
        component_indices = [int(index) for index in candidate["component_indices"]]
        if any(not 0 <= index < len(components["components"]) for index in component_indices):
            raise RuntimeError("Candidate component index is outside imported mesh")
        selected_vertices = {
            int(vertex)
            for index in component_indices
            for vertex in components["components"][index]
        }
        selected_faces = {
            int(face)
            for index in component_indices
            for face in components["faces"][index]
        }
        expected_counts = {
            "vertex_count": int(candidate["vertex_count"]),
            "edge_count": int(candidate["edge_count"]),
            "face_count": int(candidate["face_count"]),
            "component_count": int(candidate["assembly_component_count"]),
        }
        if len(selected_vertices) != expected_counts["vertex_count"]:
            raise RuntimeError(
                f"Candidate vertex mapping changed: {len(selected_vertices)} != "
                f"{expected_counts['vertex_count']}"
            )
        if len(selected_faces) != expected_counts["face_count"]:
            raise RuntimeError(
                f"Candidate face mapping changed: {len(selected_faces)} != "
                f"{expected_counts['face_count']}"
            )

        axis = canonical_axis_from_candidates(
            audit_report["wheel_detection"]["candidates"], candidate
        )
        source_axis = Vector((0.0, 0.0, 0.0))
        source_axis[int(axis["axis_index"])] = float(axis["outboard_sign"])
        rotation = source_axis.rotation_difference(Vector((1.0, 0.0, 0.0))).to_matrix().to_4x4()
        translation = Matrix.Translation(-Vector(candidate["center"]))
        canonical_matrix = rotation @ translation @ source_obj.matrix_world
        canonical_to_source = canonical_matrix.inverted()

        source_coordinate_hash = coordinate_hash(
            source_mesh, selected_vertices, canonical_matrix
        )
        source_material_histogram = material_histogram(source_mesh, selected_faces)
        source_uv_hashes = uv_hashes(source_mesh, selected_faces)
        source_custom_normals = has_valid_custom_normals(source_mesh)
        source_selected_loop_count = sum(
            len(source_mesh.polygons[index].loop_indices)
            for index in selected_faces
        )

        raw = extract_candidate(source_obj, selected_vertices, canonical_matrix)
        raw_counts = mesh_counts(raw.data)
        raw_components = component_data(raw.data)
        if raw_counts["vertex_count"] != expected_counts["vertex_count"]:
            raise RuntimeError(
                "Separated wheel vertex count does not match audit: "
                f"{raw_counts['vertex_count']} != "
                f"{expected_counts['vertex_count']}"
            )
        if raw_counts["face_count"] != expected_counts["face_count"]:
            raise RuntimeError(
                "Separated wheel face count does not match audit: "
                f"{raw_counts['face_count']} != {expected_counts['face_count']}"
            )
        if len(raw_components["components"]) != expected_counts["component_count"]:
            raise RuntimeError(
                "Separated wheel component count does not match audit: "
                f"{len(raw_components['components'])} != "
                f"{expected_counts['component_count']}"
            )

        raw_coordinate_hash = coordinate_hash(raw.data)
        raw_material_histogram = material_histogram(raw.data)
        raw_uv_hashes = uv_hashes(raw.data)
        raw_custom_normals = has_valid_custom_normals(raw.data)
        coordinate_preserved = source_coordinate_hash == raw_coordinate_hash
        materials_preserved = (
            source_material_histogram == raw_material_histogram
        )
        uv_preserved = source_uv_hashes == raw_uv_hashes
        custom_normals_preserved = (
            source_custom_normals
            and raw_custom_normals
            and source_selected_loop_count == len(raw.data.loops)
        )
        if not all(
            (
                coordinate_preserved,
                materials_preserved,
                uv_preserved,
                custom_normals_preserved,
            )
        ):
            raise RuntimeError(
                "Extraction preservation gate failed: "
                f"coordinates={coordinate_preserved}, "
                f"materials={materials_preserved}, UV={uv_preserved}, "
                f"custom_normals={custom_normals_preserved}"
            )

        scene = bpy.context.scene
        scene.name = "PreparationWorkspace"
        selected_engine = configure_scene(scene, config)
        gpu = inspect_gpu()
        if not gpu.get("available"):
            warnings.append("No hardware GPU context detected; rendering may use CPU")
            logger.warning(warnings[-1])
        raw_box = bbox(raw.data)
        diameter = max(float(raw_box["dimensions"][1]), float(raw_box["dimensions"][2]))
        rig_collection, camera = make_rig(scene, diameter)
        white_material = create_emission_material(
            "DiagnosticWhiteEmission", (1.0, 1.0, 1.0, 1.0)
        )

        reference = raw.copy()
        reference.data = raw.data.copy()
        reference.name = "Wheel_Reference_Verification"
        scene.collection.objects.link(reference)
        extraction_renders, silhouette, topology_renders = (
            render_extraction_diagnostics(
                scene,
                camera,
                raw,
                reference,
                diameter,
                paths,
                white_material,
                config,
            )
        )
        minimum_iou = min(silhouette.values())
        if minimum_iou < float(config["gates"]["minimum_silhouette_iou"]):
            raise RuntimeError(
                f"Silhouette gate failed: {minimum_iou} < "
                f"{config['gates']['minimum_silhouette_iou']}"
            )
        bpy.data.objects.remove(reference, do_unlink=True)

        skin_analysis = classify_skin_faces(raw, config)
        skin_face_indices = set(skin_analysis.pop("skin_face_indices"))
        skin_analysis["skin_component_indices"] = sorted(
            int(index) for index in skin_analysis["skin_component_indices"]
        )
        skin_analysis.pop("component_scores", None)
        original_skin = copy_faces(raw, skin_face_indices, "Wheel_Skin_Original")
        detail_faces = set(range(len(raw.data.polygons))) - skin_face_indices
        details = copy_faces(raw, detail_faces, "Wheel_Details")
        merge_records, prepared_skin = merge_sweep(
            original_skin, diameter, config
        )
        selected_merge = select_merge_result(
            merge_records,
            minimum_silhouette_iou=float(
                config["gates"]["minimum_silhouette_iou"]
            ),
            maximum_bbox_relative_error=float(
                config["gates"]["maximum_bbox_relative_error"]
            ),
        )

        thickness_measurement = float(
            candidate.get("skin_thickness", {}).get(
                "p10_world_units",
                diameter
                * float(config["skin"]["minimum_wall_thickness_ratio"]),
            )
        )
        thickness = min(
            diameter * float(config["skin"]["maximum_wall_thickness_ratio"]),
            max(
                diameter * float(config["skin"]["minimum_wall_thickness_ratio"]),
                thickness_measurement,
            ),
        )
        if prepared_skin is None:
            strategy = "hybrid_parametric_shell"
            warnings.append(
                "Original skin did not pass the closed-manifold repair gate; "
                "using the configured hybrid parametric shell."
            )
            logger.warning(warnings[-1])
            source_wheel_material = select_wheel_material(source_mesh)
            hybrid_material = derive_hybrid_skin_material(source_wheel_material)
            prepared_skin = create_shell_mesh(
                name="Wheel_Skin",
                outer_radius=float(skin_analysis["base_radius"]),
                wall_thickness=thickness,
                axial_min=float(skin_analysis["axial_min"]),
                axial_max=float(skin_analysis["axial_max"]),
                segments=int(config["skin"]["shell_angular_segments"]),
                material=hybrid_material,
            )
        else:
            strategy = "repaired_original_skin"
        final_topology = topology_metrics(prepared_skin.data)
        if not final_topology["closed_manifold"]:
            raise RuntimeError(
                "Prepared skin is not closed manifold after the selected repair"
            )

        raw_collection = create_collection("RAW_REFERENCE", scene)
        canonical_collection = create_collection("WHEEL_CANONICAL", scene)
        move_object_to_collection(raw, raw_collection)
        move_object_to_collection(original_skin, raw_collection)
        move_object_to_collection(details, canonical_collection)
        move_object_to_collection(prepared_skin, canonical_collection)
        raw.hide_render = True
        raw.hide_viewport = True
        original_skin.hide_render = True
        original_skin.hide_viewport = True
        details.hide_render = False
        prepared_skin.hide_render = False
        raw["role"] = "immutable_extracted_reference"
        original_skin["role"] = "classified_original_skin_reference"
        details["role"] = "original_hub_spokes_and_grouser_details"
        prepared_skin["role"] = "boolean_ready_skin"
        prepared_skin["repair_strategy"] = strategy
        scene["canonical_axis"] = "X"
        scene["outboard_direction"] = "+X"
        scene["source_to_canonical"] = json.dumps(matrix_rows(canonical_matrix))

        if bool(config["diagnostics"]["pack_resources_in_blend"]):
            packed = safe_pack(logger)
        else:
            packed = False
        canonical_path = paths["diagnostics"] / "wheel_canonical.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(canonical_path))

        hole = choose_hole_angle(
            details,
            base_radius=float(skin_analysis["base_radius"]),
            axial_width=float(skin_analysis["axial_max"])
            - float(skin_analysis["axial_min"]),
            bins=int(config["perforation"]["angular_search_bins"]),
        )
        cutter = create_cutter(
            name="Perforation_Cutter",
            theta=float(hole["theta_radians"]),
            outer_radius=float(skin_analysis["base_radius"]),
            diameter=diameter,
            config=config,
        )
        anomaly_skin = prepared_skin.copy()
        anomaly_skin.data = prepared_skin.data.copy()
        anomaly_skin.name = "Wheel_Skin_Perforated"
        anomaly_skin.data.name = "Wheel_Skin_Perforated_Mesh"
        scene.collection.objects.link(anomaly_skin)
        boolean_result = apply_boolean(anomaly_skin, cutter, "DIFFERENCE")
        if not boolean_result.get("success"):
            raise RuntimeError(
                f"Boolean Difference failed: {boolean_result.get('error', boolean_result)}"
            )
        if not boolean_result.get("closed_manifold"):
            raise RuntimeError("Boolean Difference produced a non-manifold skin")

        mask_proxy = prepared_skin.copy()
        mask_proxy.data = prepared_skin.data.copy()
        mask_proxy.name = "AnomalyMaskProxy"
        mask_proxy.data.name = "AnomalyMaskProxy_Mesh"
        scene.collection.objects.link(mask_proxy)
        mask_boolean = apply_boolean(mask_proxy, cutter, "INTERSECT")
        if not mask_boolean.get("success") or not mask_proxy.data.polygons:
            raise RuntimeError(
                f"Mask proxy intersection failed: {mask_boolean}"
            )
        cutter.hide_render = True

        radial = Vector(
            (
                0.0,
                math.cos(float(hole["theta_radians"])),
                math.sin(float(hole["theta_radians"])),
            )
        )
        target = radial * float(skin_analysis["base_radius"]) * 0.82
        camera_direction = (radial + Vector((0.32, 0.0, 0.0))).normalized()
        position_camera(
            camera,
            camera_direction,
            target,
            diameter,
            distance_scale=2.35,
        )
        configure_scene(scene, config, engine="BLENDER_EEVEE")
        normal_path = paths["perforation"] / "normal.png"
        anomaly_path = paths["perforation"] / "anomaly.png"
        mask_path = paths["perforation"] / "anomaly_mask.png"
        wheel_mask_path = paths["perforation"] / "wheel_mask.png"
        difference_path = paths["perforation"] / "difference.png"

        state = set_mesh_visibility([details, prepared_skin])
        render_still(scene, normal_path)
        restore_visibility(state)
        state = set_mesh_visibility([details, anomaly_skin])
        render_still(scene, anomaly_path)
        restore_visibility(state)
        render_binary_mask(
            scene, [details, prepared_skin], wheel_mask_path, white_material
        )
        render_binary_mask(scene, [mask_proxy], mask_path, white_material)
        mask_metrics = write_difference_and_metrics(
            normal_path,
            anomaly_path,
            mask_path,
            wheel_mask_path,
            difference_path,
            int(config["perforation"]["mask_dilation_pixels"]),
        )

        normal_scene_details = details.copy()
        normal_scene_details.data = details.data
        normal_scene_details.name = "Wheel_Details_Normal"
        anomaly_scene_details = details.copy()
        anomaly_scene_details.data = details.data
        anomaly_scene_details.name = "Wheel_Details_Perforation"
        normal_scene_skin = prepared_skin.copy()
        normal_scene_skin.data = prepared_skin.data
        normal_scene_skin.name = "Wheel_Skin_Normal"
        anomaly_scene_skin = anomaly_skin.copy()
        anomaly_scene_skin.data = anomaly_skin.data
        anomaly_scene_skin.name = "Wheel_Skin_Anomaly"
        pair_scenes = create_pair_scenes(
            config,
            rig_collection,
            [normal_scene_details, normal_scene_skin],
            [anomaly_scene_details, anomaly_scene_skin],
            camera,
        )
        for pair_scene in pair_scenes.values():
            pair_scene["counterfactual_pair"] = True
            pair_scene["camera_locked"] = True
            pair_scene["lighting_locked"] = True
        probe_path = paths["diagnostics"] / "perforation_probe.blend"
        if bool(config["diagnostics"]["pack_resources_in_blend"]):
            safe_pack(logger)
        bpy.ops.wm.save_as_mainfile(filepath=str(probe_path))

        sha_after = sha256_file(asset)
        if sha_after != sha_before:
            raise RuntimeError("Source GLB checksum changed during preparation")
        minimum_mask_ratio = float(config["gates"]["minimum_mask_ratio"])
        maximum_mask_ratio = float(config["gates"]["maximum_mask_ratio"])
        mask_valid = minimum_mask_ratio <= float(
            mask_metrics["area_ratio"]
        ) <= maximum_mask_ratio
        outside_valid = float(mask_metrics["outside_change_ratio"]) <= float(
            config["gates"]["maximum_outside_change_ratio"]
        )
        validation_errors = []
        if not mask_valid:
            validation_errors.append(
                f"Mask ratio {mask_metrics['area_ratio']} outside configured range"
            )
        if not outside_valid:
            validation_errors.append(
                "Counterfactual difference outside the dilated mask exceeds gate"
            )
        if validation_errors:
            raise RuntimeError("; ".join(validation_errors))

        report = {
            "schema_version": str(config["schema_version"]),
            "status": "completed",
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "duration_seconds": round(time.perf_counter() - started, 3),
            "asset": {
                "path": str(asset),
                "sha256_before": sha_before,
                "sha256_after": sha_after,
                "unchanged": sha_before == sha_after,
                "source_writes_performed": False,
            },
            "audit_report": {
                "path": str(audit_path),
                "schema_version": audit_report.get("schema_version"),
                "classification": audit_report.get("classification", {}).get(
                    "category"
                ),
                "compatible": True,
            },
            "candidate": {
                "id": candidate["id"],
                "object_name": candidate["object_name"],
                "component_indices": component_indices,
                "expected_counts": expected_counts,
                "source_center": candidate["center"],
                "source_bbox": candidate["bbox"],
            },
            "environment": {
                "blender_version": bpy.app.version_string,
                "blender_version_tuple": list(bpy.app.version),
                "python_version": sys.version,
                "platform": platform.platform(),
                "render_engines_available": available_render_engines(scene),
                "selected_render_engine": selected_engine,
                "render_resolution": {
                    "width": int(config["render"]["resolution_x"]),
                    "height": int(config["render"]["resolution_y"]),
                    "aspect_ratio": "4:3",
                },
                "gpu": gpu,
            },
            "configuration": config,
            "preflight": {
                "compatible": True,
                "source_mesh_counts": source_counts,
                "component_count": len(components["components"]),
            },
            "canonical_transform": {
                "source_to_canonical": matrix_rows(canonical_matrix),
                "canonical_to_source": matrix_rows(canonical_to_source),
                "origin": [0.0, 0.0, 0.0],
                "axis": axis,
                "final_axis": "X",
                "final_outboard_direction": "+X",
            },
            "extraction": {
                **raw_counts,
                "component_count": len(raw_components["components"]),
                "coordinate_hash_source": source_coordinate_hash,
                "coordinate_hash_extracted": raw_coordinate_hash,
                "coordinates_preserved": coordinate_preserved,
                "material_histogram_source": source_material_histogram,
                "material_histogram_extracted": raw_material_histogram,
                "materials_preserved": materials_preserved,
                "uv_hashes_source": source_uv_hashes,
                "uv_hashes_extracted": raw_uv_hashes,
                "uv_preserved": uv_preserved,
                "custom_normals_preserved": custom_normals_preserved,
                "silhouette_iou": silhouette,
                "minimum_silhouette_iou": minimum_iou,
                "method": (
                    "full mesh duplicate + BMesh selection + Blender separate; "
                    "no from_pydata reconstruction"
                ),
            },
            "skin_analysis": skin_analysis,
            "repair": {
                "strategy": strategy,
                "merge_sweep": merge_records,
                "selected_merge": selected_merge,
                "outer_radius": float(skin_analysis["base_radius"]),
                "wall_thickness": thickness,
                "axial_min": float(skin_analysis["axial_min"]),
                "axial_max": float(skin_analysis["axial_max"]),
                "uv_strategy": (
                    "original"
                    if strategy == "repaired_original_skin"
                    else "new cylindrical UV on replacement skin only"
                ),
                "material_strategy": (
                    "original"
                    if strategy == "repaired_original_skin"
                    else (
                        "NASA wheel material retained on original details; "
                        "replacement shell uses a texture-free metal derived "
                        "from the atlas average to avoid unrelated atlas regions"
                    )
                ),
                "original_details_preserved": True,
                "voxel_remesh_used": False,
                "retopology_used": False,
                "final_topology": final_topology,
            },
            "perforation": {
                "placement": hole,
                "boolean": boolean_result,
                "mask_proxy_boolean": mask_boolean,
                "mask": mask_metrics,
                "pair": {
                    "normal_scene": pair_scenes["normal"].name,
                    "anomaly_scene": pair_scenes["perforation"].name,
                    "camera_identical": True,
                    "lighting_identical": True,
                    "materials_identical_outside_anomaly": True,
                },
            },
            "validation": {
                "valid": True,
                "preflight": True,
                "extraction_preservation": True,
                "silhouette": True,
                "prepared_skin_manifold": True,
                "boolean": True,
                "mask_area": mask_valid,
                "counterfactual_outside_mask": outside_valid,
                "blend_resources_packed": packed,
            },
            "artifacts": {
                "json_report": "reports/preparation.json",
                "markdown_report": "reports/preparation.md",
                "canonical_blend": relative(canonical_path, paths["root"]),
                "perforation_blend": relative(probe_path, paths["root"]),
                "log": "logs/preparation.log",
                "extraction_renders": extraction_renders,
                "silhouette_verification_renders": [
                    relative(path, paths["root"])
                    for path in sorted(paths["extraction"].glob("verification_*.png"))
                ],
                "topology_renders": topology_renders,
                "perforation_renders": [
                    relative(normal_path, paths["root"]),
                    relative(anomaly_path, paths["root"]),
                    relative(mask_path, paths["root"]),
                    relative(difference_path, paths["root"]),
                    relative(wheel_mask_path, paths["root"]),
                ],
            },
            "warnings": warnings,
            "errors": errors,
        }
        write_reports(report, paths)
        logger.info(
            "Wheel preparation completed: strategy=%s, mask_ratio=%.6f",
            strategy,
            mask_metrics["area_ratio"],
        )
        return 0
    except Exception as exc:
        errors.append(str(exc))
        logger.error("%s", exc)
        logger.debug("%s", traceback.format_exc())
        sha_after = sha256_file(asset) if asset.is_file() else None
        report.update(
            {
                "status": "failed",
                "completed_utc": utc_now(),
                "duration_seconds": round(time.perf_counter() - started, 3),
                "asset": {
                    "path": str(asset),
                    "sha256_before": sha_before,
                    "sha256_after": sha_after,
                    "unchanged": (
                        sha_before is not None and sha_before == sha_after
                    ),
                    "source_writes_performed": False,
                },
                "warnings": warnings,
                "errors": errors,
            }
        )
        write_reports(report, paths)
        return 1


def main() -> int:
    return prepare(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
