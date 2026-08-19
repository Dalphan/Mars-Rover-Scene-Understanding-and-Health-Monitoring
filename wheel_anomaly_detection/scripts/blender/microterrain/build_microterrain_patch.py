"""Construct the dense Level-1 patch and non-destructive macroterrain opening."""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np


PATCH_OBJECT = "MicroterrainPatch_L1"
MACRO_PROXY_OBJECT = "GaleTerrainVisual_L1"
COLLECTION_NAME = "Microterrain_L1"


def _remove_existing() -> None:
    for name in (PATCH_OBJECT, MACRO_PROXY_OBJECT):
        obj = bpy.data.objects.get(name)
        if obj:
            data = obj.data if obj.type == "MESH" else None
            bpy.data.objects.remove(obj, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)
    collection = bpy.data.collections.get(COLLECTION_NAME)
    if collection:
        bpy.data.collections.remove(collection)


def _dense_grid_mesh(name: str, x_axis, y_axis, z, macro_bounds) -> bpy.types.Mesh:
    rows, columns = z.shape
    vertex_count = rows * columns
    face_count = (rows - 1) * (columns - 1)
    grid_x, grid_y = np.meshgrid(x_axis.astype(np.float32), y_axis.astype(np.float32))
    coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel(), z.astype(np.float32).ravel())).astype(np.float32, copy=False)
    face_rows, face_columns = np.meshgrid(np.arange(rows - 1, dtype=np.int32), np.arange(columns - 1, dtype=np.int32), indexing="ij")
    lower_left = (face_rows * columns + face_columns).ravel()
    loop_vertices = np.column_stack((lower_left, lower_left + 1, lower_left + columns + 1, lower_left + columns)).astype(np.int32, copy=False).ravel()

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(vertex_count)
    mesh.vertices.foreach_set("co", coordinates.ravel())
    mesh.loops.add(face_count * 4)
    mesh.loops.foreach_set("vertex_index", loop_vertices)
    mesh.polygons.add(face_count)
    mesh.polygons.foreach_set("loop_start", np.arange(face_count, dtype=np.int32) * 4)
    mesh.polygons.foreach_set("loop_total", np.full(face_count, 4, dtype=np.int32))
    mesh.polygons.foreach_set("use_smooth", np.ones(face_count, dtype=np.bool_))

    left, bottom, right, top = map(float, macro_bounds)
    vertex_uv = np.column_stack(((coordinates[:, 0] - left) / (right - left), (coordinates[:, 1] - bottom) / (top - bottom))).astype(np.float32)
    uv_layer = mesh.uv_layers.new(name="GeoreferencedUV")
    uv_layer.uv.foreach_set("vector", vertex_uv[loop_vertices].ravel())
    mesh.update(calc_edges=True)
    return mesh


def _masked_macro_material(source: bpy.types.Material, bounds: tuple[float, float, float, float]) -> bpy.types.Material:
    material = source.copy()
    material.name = "GaleTerrainMacroAlbedo_L1_Hole"
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        raise RuntimeError("Macroterrain material has no Principled BSDF")
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.name = "MicroterrainHolePosition"
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Position"], separate.inputs["Vector"])
    comparisons = []
    for operation, coordinate, threshold, name in (
        ("GREATER_THAN", "X", bounds[0], "InsideLeft"),
        ("LESS_THAN", "X", bounds[2], "InsideRight"),
        ("GREATER_THAN", "Y", bounds[1], "InsideBottom"),
        ("LESS_THAN", "Y", bounds[3], "InsideTop"),
    ):
        node = nodes.new("ShaderNodeMath")
        node.operation = operation
        node.name = name
        node.inputs[1].default_value = float(threshold)
        links.new(separate.outputs[coordinate], node.inputs[0])
        comparisons.append(node)
    multiply_x = nodes.new("ShaderNodeMath"); multiply_x.operation = "MULTIPLY"
    multiply_y = nodes.new("ShaderNodeMath"); multiply_y.operation = "MULTIPLY"
    inside = nodes.new("ShaderNodeMath"); inside.operation = "MULTIPLY"
    links.new(comparisons[0].outputs[0], multiply_x.inputs[0]); links.new(comparisons[1].outputs[0], multiply_x.inputs[1])
    links.new(comparisons[2].outputs[0], multiply_y.inputs[0]); links.new(comparisons[3].outputs[0], multiply_y.inputs[1])
    links.new(multiply_x.outputs[0], inside.inputs[0]); links.new(multiply_y.outputs[0], inside.inputs[1])
    alpha = nodes.new("ShaderNodeMath"); alpha.operation = "SUBTRACT"; alpha.name = "OutsideMicroterrainPatch"; alpha.inputs[0].default_value = 1.0
    links.new(inside.outputs[0], alpha.inputs[1]); links.new(alpha.outputs[0], principled.inputs["Alpha"])
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    material["microterrain_hole_bounds_m"] = list(map(float, bounds))
    material["source_material"] = source.name
    return material


def build_patch(x_axis, y_axis, base_z, displacement, macro_bounds, metrics: dict) -> tuple[bpy.types.Object, bpy.types.Object]:
    _remove_existing()
    original = bpy.data.objects.get("GaleTerrainVisual")
    if original is None or original.type != "MESH" or not original.data.materials:
        raise RuntimeError("Missing valid GaleTerrainVisual")
    collection = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)
    bounds = (float(x_axis[0]), float(y_axis[0]), float(x_axis[-1]), float(y_axis[-1]))

    source_material = original.data.materials[0]
    macro_proxy = original.copy()
    macro_proxy.data = original.data.copy()
    macro_proxy.name = MACRO_PROXY_OBJECT
    collection.objects.link(macro_proxy)
    masked_material = _masked_macro_material(source_material, bounds)
    if len(macro_proxy.data.materials):
        macro_proxy.data.materials[0] = masked_material
    else:
        macro_proxy.data.materials.append(masked_material)
    macro_proxy["semantic_role"] = "macroterrain_visual_with_microterrain_opening"
    macro_proxy["source_object"] = original.name
    original.hide_render = True

    mesh = _dense_grid_mesh(f"{PATCH_OBJECT}_mesh", x_axis, y_axis, base_z + displacement, macro_bounds)
    patch = bpy.data.objects.new(PATCH_OBJECT, mesh)
    collection.objects.link(patch)
    patch.data.materials.append(bpy.data.materials.get("GaleTerrainMacroAlbedo") or source_material)
    patch["semantic_role"] = "microterrain_visual_and_local_contact_surface"
    patch["microterrain_level"] = 1
    patch["seed"] = int(metrics["seed"])
    patch["displacement_signature_sha256"] = metrics["signature_sha256"]
    patch["patch_bounds_m"] = list(bounds)
    patch["grid_spacing_m"] = float(x_axis[1] - x_axis[0])
    patch["maximum_absolute_displacement_m"] = float(metrics["maximum_absolute_displacement_m"])
    return patch, macro_proxy
