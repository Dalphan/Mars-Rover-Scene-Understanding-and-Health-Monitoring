"""Validate the persisted Level-2 instanced clast scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.build_report.resolve(), args.output_report.resolve()


def _read_attribute(mesh, name: str, width: int, property_name: str, dtype):
    attribute = mesh.attributes.get(name)
    if attribute is None or attribute.domain != "POINT":
        return None
    values = np.empty(len(attribute.data) * width, dtype=dtype)
    attribute.data.foreach_get(property_name, values)
    return values.reshape(-1, width) if width > 1 else values


def _family_signature(obj: bpy.types.Object) -> tuple[str | None, dict]:
    count = len(obj.data.vertices)
    positions = np.empty(count * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", positions)
    positions = positions.reshape(-1, 3)
    rotations = _read_attribute(obj.data, "instance_rotation", 3, "vector", np.float32)
    scales = _read_attribute(obj.data, "instance_scale", 3, "vector", np.float32)
    sizes = _read_attribute(obj.data, "characteristic_size_m", 1, "value", np.float32)
    burial = _read_attribute(obj.data, "burial_fraction", 1, "value", np.float32)
    classes = _read_attribute(obj.data, "size_class_index", 1, "value", np.int32)
    indices = _read_attribute(obj.data, "prototype_index", 1, "value", np.int32)
    arrays = (positions, rotations, scales, sizes, burial, classes, indices)
    if any(values is None for values in arrays):
        return None, {}
    signature = hashlib.sha256()
    for values in arrays:
        signature.update(values.tobytes(order="C"))
    return signature.hexdigest(), {
        "positions": positions,
        "rotations": rotations,
        "scales": scales,
        "sizes": sizes,
        "burial": burial,
        "classes": classes,
        "indices": indices,
    }


def _bilinear(axis_x, axis_y, values, x, y):
    columns = np.interp(x, axis_x, np.arange(len(axis_x)))
    rows = np.interp(y, axis_y, np.arange(len(axis_y)))
    c0, r0 = np.floor(columns).astype(np.int32), np.floor(rows).astype(np.int32)
    c1, r1 = np.minimum(c0 + 1, len(axis_x) - 1), np.minimum(r0 + 1, len(axis_y) - 1)
    wc, wr = columns - c0, rows - r0
    return values[r0, c0] * (1 - wc) * (1 - wr) + values[r0, c1] * wc * (1 - wr) + values[r1, c0] * (1 - wc) * wr + values[r1, c1] * wc * wr


def validate(config_path: Path, build_report_path: Path, output_report_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.build_clast_library import FAMILY_COLLECTIONS, LIBRARY_COLLECTION
    from scripts.blender.microterrain.build_clast_scatter import SCATTER_OBJECTS
    from src.microterrain.clasts import FAMILIES, generate_clast_scatter, validate_level2_config
    from src.microterrain.core import generate_meso_relief
    from src.microterrain.level2_validation import validate_level2_report

    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(build_report_path.read_text(encoding="utf-8"))
    contract = validate_level2_config(config)
    errors = validate_level2_report(config, report)
    patch = bpy.data.objects.get("MicroterrainPatch_L1")
    if patch is None or patch.type != "MESH":
        errors.append("Persisted Level-1 patch is missing")
        surface_z = None
    else:
        expected_topology = report["level1"]["patch_topology"]
        if [len(patch.data.vertices), len(patch.data.polygons)] != expected_topology:
            errors.append("Level-1 patch topology changed")
        if patch.get("displacement_signature_sha256") != report["level1"]["displacement_signature_sha256"]:
            errors.append("Level-1 patch signature changed")
        coordinates = np.empty(len(patch.data.vertices) * 3, dtype=np.float32)
        patch.data.vertices.foreach_get("co", coordinates)
        surface_z = coordinates.reshape(-1, 3)[:, 2].reshape(contract["samples"], contract["samples"])

    family_signatures = {}
    point_count = 0
    no_float_max_bottom = -math.inf
    center_x, center_y = map(float, json.loads(Path(report["level1"]["report"]).read_text(encoding="utf-8"))["patch"]["center_xy_m"])
    half_size = float(config["microterrain"]["patch_size_m"]) / 2.0
    x_axis = np.linspace(center_x - half_size, center_x + half_size, contract["samples"], dtype=np.float32)
    y_axis = np.linspace(center_y - half_size, center_y + half_size, contract["samples"], dtype=np.float32)
    for family in FAMILIES:
        obj = bpy.data.objects.get(SCATTER_OBJECTS[family])
        if obj is None or obj.type != "MESH":
            errors.append(f"Missing scatter object for {family}")
            continue
        point_count += len(obj.data.vertices)
        if len(obj.data.vertices) != contract["counts"][family]:
            errors.append(f"Point count mismatch for {family}")
        modifiers = [modifier for modifier in obj.modifiers if modifier.type == "NODES" and modifier.node_group]
        if len(modifiers) != 1:
            errors.append(f"Expected one Geometry Nodes modifier for {family}")
        else:
            node_types = {node.bl_idname for node in modifiers[0].node_group.nodes}
            if "GeometryNodeInstanceOnPoints" not in node_types or "GeometryNodeCollectionInfo" not in node_types or "GeometryNodeRealizeInstances" in node_types:
                errors.append(f"Invalid non-realized instancing graph for {family}")
        signature, arrays = _family_signature(obj)
        family_signatures[family] = signature
        if signature != report["clasts"]["family_signatures_sha256"][family] or obj.get("scatter_signature_sha256") != signature:
            errors.append(f"Persisted scatter signature mismatch for {family}")
        if arrays and surface_z is not None:
            if int(arrays["indices"].min()) < 0 or int(arrays["indices"].max()) >= contract["variants_per_family"][family]:
                errors.append(f"Prototype index outside source library for {family}")
            positions, rotations, scales = arrays["positions"], arrays["rotations"], arrays["scales"]
            sizes, burial, classes = arrays["sizes"], arrays["burial"], arrays["classes"]
            cx, sx = np.cos(rotations[:, 0]), np.sin(rotations[:, 0])
            cy, sy = np.cos(rotations[:, 1]), np.sin(rotations[:, 1])
            vertical_radius = 0.5 * np.sqrt((sy * scales[:, 0]) ** 2 + (sx * cy * scales[:, 1]) ** 2 + (cx * cy * scales[:, 2]) ** 2)
            terrain = _bilinear(x_axis, y_axis, surface_z, positions[:, 0], positions[:, 1])
            bottom_above_surface = positions[:, 2] - vertical_radius - terrain
            no_float_max_bottom = max(no_float_max_bottom, float(bottom_above_surface.max()))
            if np.any(bottom_above_surface > 1e-7):
                errors.append(f"Floating clast estimate detected for {family}")
            radius = 0.5 * sizes
            edge_margin = float(config["microterrain"]["clasts"]["edge_margin_m"])
            outside_patch = (
                (positions[:, 0] - radius < x_axis[0] + edge_margin)
                | (positions[:, 0] + radius > x_axis[-1] - edge_margin)
                | (positions[:, 1] - radius < y_axis[0] + edge_margin)
                | (positions[:, 1] + radius > y_axis[-1] - edge_margin)
            )
            if np.any(outside_patch):
                errors.append(f"Size-aware patch boundary violated by {family}")
            for zone in report["clasts"]["exclusion_zones"]:
                inside = ((positions[:, 0] - zone[0]) / (zone[2] + radius)) ** 2 + ((positions[:, 1] - zone[1]) / (zone[3] + radius)) ** 2 < 1.0
                if np.any(inside):
                    errors.append(f"Size-aware wheel exclusion violated by {family}")
                    break
            if family == "coarse_clasts" and contract["coarse_class_counts"]:
                configured_classes = config["microterrain"]["clasts"][family]["size_classes"]
                actual_counts = {entry["name"]: int(np.count_nonzero(classes == index)) for index, entry in enumerate(configured_classes)}
                if actual_counts != contract["coarse_class_counts"]:
                    errors.append("Persisted coarse size-class counts changed")
                for index, entry in enumerate(configured_classes):
                    selected = classes == index
                    low, high = map(float, entry["buried_fraction"])
                    if np.any(burial[selected] < low - 1e-7) or np.any(burial[selected] > high + 1e-7):
                        errors.append(f"Burial range violated by coarse class {entry['name']}")
                spacing_factor = float(config["microterrain"]["clasts"][family]["minimum_center_spacing_factor"])
                dx = positions[:, None, 0] - positions[None, :, 0]
                dy = positions[:, None, 1] - positions[None, :, 1]
                required = spacing_factor * (sizes[:, None] + sizes[None, :])
                overlap = dx * dx + dy * dy < required * required
                np.fill_diagonal(overlap, False)
                if np.any(overlap):
                    errors.append("Coarse clast minimum spacing was violated")
            elif family != "coarse_clasts":
                low, high = contract["burial_ranges"][family]
                if np.any(burial < low - 1e-7) or np.any(burial > high + 1e-7):
                    errors.append(f"Burial range violated by {family}")

    combined = hashlib.sha256("".join(family_signatures.get(family) or "" for family in FAMILIES).encode("ascii")).hexdigest()
    if combined != report["clasts"]["signature_sha256"]:
        errors.append("Combined persisted scatter signature mismatch")
    family_collections = [bpy.data.collections.get(FAMILY_COLLECTIONS[family]) for family in FAMILIES]
    if any(collection is None for collection in family_collections):
        errors.append("One or more source clast family collections are missing")
    else:
        source_objects = [obj for collection in family_collections for obj in collection.objects]
        if len(source_objects) != report["library"]["source_mesh_count"]:
            errors.append("Source library object count mismatch")
        if any(not obj.data.materials for obj in source_objects):
            errors.append("Source prototype without clast material")
        scene_collections = set(bpy.context.scene.collection.children_recursive)
        if any(collection in scene_collections for collection in family_collections):
            errors.append("Source family collections must not be directly linked into the rendered scene")

    if surface_z is not None:
        displacement, l1_metrics = generate_meso_relief(x_axis, y_axis, config, np)
        regenerated, regenerated_metrics = generate_clast_scatter(x_axis, y_axis, surface_z, displacement, config, np, report["clasts"]["exclusion_zones"])
        if l1_metrics["signature_sha256"] != report["level1"]["displacement_signature_sha256"]:
            errors.append("Level-1 signature cannot be regenerated")
        if regenerated_metrics["signature_sha256"] != report["clasts"]["signature_sha256"]:
            errors.append("Level-2 scatter is not deterministic")
    else:
        regenerated_metrics = {"signature_sha256": None}
    for path in report["renders"].values():
        if not Path(path).is_file():
            errors.append(f"Missing render file: {path}")
    result = {
        "ok": not errors,
        "level": 2,
        "blend": bpy.data.filepath,
        "errors": errors,
        "deterministic_signature_sha256": regenerated_metrics["signature_sha256"],
        "persisted_signature_sha256": combined,
        "point_instance_count": point_count,
        "maximum_estimated_clast_bottom_above_surface_m": no_float_max_bottom,
        "geometry_nodes_non_realized": not any("instancing" in error.lower() for error in errors),
        "scope": {"level3_shading_enabled": False},
        "renders": report["renders"],
    }
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError("Microterrain Level 2 validation failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    validate(*_arguments())
