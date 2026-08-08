from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Apply Milestone 6 feather to patch appearance, gravel, and displacement.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-label", default="rover_simulator_m5_lighting.blend")
    parser.add_argument("--patches", default="A,B,C", help="Comma-separated patch suffixes to process")
    return parser.parse_args(values)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def feather_at(x: float, y: float, center_x: float, center_y: float, patch: dict) -> float:
    half = patch["size_m"] * 0.5
    edge = min(half - abs(x - center_x), half - abs(y - center_y))
    return smoothstep(edge / max(1e-6, patch["edge_feather_m"]))


def surface_bvh(obj: bpy.types.Object) -> BVHTree:
    return BVHTree.FromObject(obj, bpy.context.evaluated_depsgraph_get())


def surface_hit(obj: bpy.types.Object, bvh: BVHTree, x: float, y: float) -> tuple[Vector, Vector]:
    inverse = obj.matrix_world.inverted()
    origin = inverse @ Vector((x, y, 10.0))
    direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, normal, _, _ = bvh.ray_cast(origin, direction, 20.0)
    if hit is None:
        raise RuntimeError(f"Terrain raycast missed at ({x:.5f}, {y:.5f})")
    return obj.matrix_world @ hit, (obj.matrix_world.to_3x3() @ normal).normalized()


def safe_surface_hit(obj: bpy.types.Object, bvh: BVHTree, x: float, y: float) -> tuple[Vector, Vector] | None:
    """Return the base surface, probing a small neighborhood across source holes."""
    offsets = [(0.0, 0.0), (0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02),
               (0.05, 0.0), (-0.05, 0.0), (0.0, 0.05), (0.0, -0.05),
               (0.10, 0.0), (-0.10, 0.0), (0.0, 0.10), (0.0, -0.10)]
    for dx, dy in offsets:
        try:
            return surface_hit(obj, bvh, x + dx, y + dy)
        except RuntimeError:
            continue
    return None


def patch_center(obj: bpy.types.Object) -> tuple[float, float]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return sum(point.x for point in points) / len(points), sum(point.y for point in points) / len(points)


def local_fallback_plane(terrain: bpy.types.Object, terrain_bvh: BVHTree, center_x: float,
                         center_y: float) -> tuple[float, float, float, int]:
    """Fit a gentle local plane when the imported terrain has a source hole."""
    samples: list[tuple[float, float, float]] = []
    for dx, dy in ((-4.0, 0.0), (4.0, 0.0), (0.0, -4.0), (0.0, 4.0),
                   (-4.0, -4.0), (-4.0, 4.0), (4.0, -4.0), (4.0, 4.0),
                   (-6.0, 0.0), (6.0, 0.0), (0.0, -6.0), (0.0, 6.0)):
        hit = safe_surface_hit(terrain, terrain_bvh, center_x + dx, center_y + dy)
        if hit is not None:
            point, _ = hit
            samples.append((center_x + dx, center_y + dy, point.z))
    if len(samples) < 2:
        return 0.0, 0.0, 0.0, len(samples)
    left = min(samples, key=lambda item: item[0])
    right = max(samples, key=lambda item: item[0])
    down = min(samples, key=lambda item: item[1])
    up = max(samples, key=lambda item: item[1])
    dx = max(1e-6, right[0] - left[0])
    dy = max(1e-6, up[1] - down[1])
    slope_x = (right[2] - left[2]) / dx
    slope_y = (up[2] - down[2]) / dy
    mean_z = sum(item[2] for item in samples) / len(samples)
    intercept = mean_z - slope_x * center_x - slope_y * center_y
    return slope_x, slope_y, intercept, len(samples)


def node(nodes: bpy.types.Nodes, name: str, node_type: str) -> bpy.types.Node:
    existing = nodes.get(name)
    if existing is not None:
        nodes.remove(existing)
    created = nodes.new(node_type)
    created.name = name
    created.label = name
    return created


def linked_source(socket: bpy.types.NodeSocket) -> bpy.types.NodeSocket | None:
    return socket.links[0].from_socket if socket.links else None


def clear_socket(socket: bpy.types.NodeSocket) -> None:
    for link in list(socket.links):
        socket.id_data.links.remove(link)


def feather_material(material: bpy.types.Material, patch_id: str) -> dict:
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF") or next(item for item in nodes if item.type == "BSDF_PRINCIPLED")
    patch_color = linked_source(principled.inputs["Base Color"])
    patch_roughness = linked_source(principled.inputs["Roughness"])
    patch_normal = linked_source(principled.inputs["Normal"])
    if patch_color is None or patch_roughness is None or patch_normal is None:
        raise RuntimeError(f"{material.name} does not have a complete M5 shader")
    # At feather=0 the patch must reproduce the underlying terrain material.
    # The multiplier node is intentionally darker than the base terrain and
    # would therefore leave a rectangular albedo seam even with a smooth
    # geometric transition.  Use the original diffuse texture directly;
    # both the terrain and the patches carry the same world UV map.
    original_color = nodes.get("Diffuse").outputs.get("Color") if nodes.get("Diffuse") else None
    original_roughness = nodes.get("Roughness").outputs.get("Color") if nodes.get("Roughness") else None
    if original_color is None or original_roughness is None:
        raise RuntimeError(f"{material.name} is missing original terrain texture sources")

    prefix = f"{patch_id}_M6_"
    for item in list(nodes):
        if item.name.startswith(prefix):
            nodes.remove(item)
    feather = node(nodes, prefix + "PatchFeather", "ShaderNodeAttribute")
    feather.attribute_name = "PatchFeather"

    color_mix = node(nodes, prefix + "ColorFeatherMix", "ShaderNodeMixRGB")
    color_mix.blend_type = "MIX"
    links.new(feather.outputs["Fac"], color_mix.inputs["Fac"])
    links.new(original_color, color_mix.inputs[1])
    links.new(patch_color, color_mix.inputs[2])
    clear_socket(principled.inputs["Base Color"])
    links.new(color_mix.outputs["Color"], principled.inputs["Base Color"])

    rough_mix = node(nodes, prefix + "RoughnessFeatherMix", "ShaderNodeMixRGB")
    rough_mix.blend_type = "MIX"
    links.new(feather.outputs["Fac"], rough_mix.inputs["Fac"])
    links.new(original_roughness, rough_mix.inputs[1])
    links.new(patch_roughness, rough_mix.inputs[2])
    clear_socket(principled.inputs["Roughness"])
    links.new(rough_mix.outputs["Color"], principled.inputs["Roughness"])

    bump_source_node = nodes.get(f"{patch_id}_M2_GranularBump") or nodes.get(f"{patch_id}_Regolith_Bump")
    if bump_source_node is None or "Strength" not in bump_source_node.inputs:
        raise RuntimeError(f"{material.name} is missing the M2 bump node")
    old_strength = float(bump_source_node.inputs["Strength"].default_value)
    strength = node(nodes, prefix + "BumpStrength", "ShaderNodeMath")
    strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = old_strength
    links.new(feather.outputs["Fac"], strength.inputs[0])
    clear_socket(bump_source_node.inputs["Strength"])
    links.new(strength.outputs["Value"], bump_source_node.inputs["Strength"])

    material["milestone_6_feather"] = True
    material["feather_width_m"] = 0.35
    material["feather_channels"] = "color,roughness,bump"
    return {
        "material": material.name,
        "channels": ["color", "roughness", "bump"],
        "bump_node": bump_source_node.name,
        "feather_attribute": "PatchFeather",
    }


def connected_components(mesh: bpy.types.Mesh) -> list[list[int]]:
    adjacency = [[] for _ in mesh.vertices]
    for edge in mesh.edges:
        a, b = edge.vertices
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in range(len(adjacency)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append(component)
    return [component for component in components if component]


def rebuild_gravel_with_feather(gravel: bpy.types.Object, patch_obj: bpy.types.Object, patch: dict,
                                settings: dict) -> dict:
    mesh = gravel.data
    center_x, center_y = patch_center(patch_obj)
    components = connected_components(mesh)
    polygon_by_vertex: dict[int, list[int]] = {index: [] for index in range(len(mesh.vertices))}
    for polygon_index, polygon in enumerate(mesh.polygons):
        for vertex_index in polygon.vertices:
            polygon_by_vertex[vertex_index].append(polygon_index)
    kept_components: list[list[int]] = []
    feather_values: list[float] = []
    rng = random.Random(patch["seed"] + settings["density_seed_offset"])
    dropped = 0
    for component_index, component in enumerate(components):
        centroid_local = sum((mesh.vertices[index].co for index in component), Vector()) / len(component)
        centroid_world = gravel.matrix_world @ centroid_local
        feather = feather_at(centroid_world.x, centroid_world.y, center_x, center_y, patch)
        keep_probability = max(0.0, min(1.0, feather))
        if feather < settings["edge_threshold"] or rng.random() > keep_probability:
            dropped += 1
            continue
        kept_components.append(component)
        feather_values.append(feather)

    new_vertices: list[tuple[float, float, float]] = []
    new_faces: list[tuple[int, ...]] = []
    new_material_indices: list[int] = []
    for component, feather in zip(kept_components, feather_values):
        centroid = sum((mesh.vertices[index].co for index in component), Vector()) / len(component)
        scale = float(settings["gravel_scale_floor"]) + (1.0 - float(settings["gravel_scale_floor"])) * feather
        remap = {}
        for vertex_index in component:
            co = centroid + (mesh.vertices[vertex_index].co - centroid) * scale
            remap[vertex_index] = len(new_vertices)
            new_vertices.append(tuple(co))
        polygon_indices = sorted({polygon_index for vertex_index in component for polygon_index in polygon_by_vertex[vertex_index]})
        component_set = set(component)
        for polygon_index in polygon_indices:
            polygon = mesh.polygons[polygon_index]
            if not all(vertex_index in component_set for vertex_index in polygon.vertices):
                continue
            new_faces.append(tuple(remap[vertex_index] for vertex_index in polygon.vertices))
            new_material_indices.append(int(polygon.material_index))

    old_mesh = gravel.data
    old_materials = list(old_mesh.materials)
    new_mesh = bpy.data.meshes.new(f"{patch['id']}_Gravel_M6_Feather_Mesh")
    new_mesh.from_pydata(new_vertices, [], new_faces)
    new_mesh.update()
    gravel.data = new_mesh
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    for material in old_materials:
        new_mesh.materials.append(material)
    for polygon, material_index in zip(new_mesh.polygons, new_material_indices):
        polygon.material_index = material_index
    gravel["milestone_6_feather"] = True
    gravel["feather_width_m"] = float(patch["edge_feather_m"])
    gravel["density_fade"] = True
    return {
        "components_before": len(components),
        "components_after": len(kept_components),
        "components_dropped": dropped,
        "feather_width_m": float(patch["edge_feather_m"]),
        "scale_floor": float(settings["gravel_scale_floor"]),
    }


def feather_displacement(patch_obj: bpy.types.Object, terrain: bpy.types.Object, patch: dict,
                         terrain_bvh: BVHTree) -> dict:
    inverse = patch_obj.matrix_world.inverted()
    before_max = 0.0
    after_max = 0.0
    misses = 0
    center_x, center_y = patch_center(patch_obj)
    slope_x, slope_y, intercept, plane_samples = local_fallback_plane(terrain, terrain_bvh, center_x, center_y)
    center_hit = safe_surface_hit(terrain, terrain_bvh, center_x, center_y)
    use_plane = center_hit is None
    for vertex in patch_obj.data.vertices:
        world = patch_obj.matrix_world @ vertex.co
        base_hit = None if use_plane else safe_surface_hit(terrain, terrain_bvh, world.x, world.y)
        if base_hit is None:
            # The imported terrain has a source hole under the patch. Use a
            # locally fitted plane so the 35 cm transition remains continuous
            # instead of leaving a rectangular vertical wall.
            misses += 1
            base = Vector((world.x, world.y, slope_x * world.x + slope_y * world.y + intercept))
        else:
            base, _ = base_hit
        before = abs(world.z - base.z)
        feather = feather_at(world.x, world.y, *patch_center(patch_obj), patch)
        # Keep a tiny positive clearance in the feather-zero overlap band to
        # avoid z-fighting where the patch closes the source terrain hole.
        edge_clearance = float(patch.get("edge_offset_m", 0.0))
        blended_z = base.z + (world.z - base.z) * feather + edge_clearance * (1.0 - feather)
        after = abs(blended_z - base.z)
        before_max = max(before_max, before)
        after_max = max(after_max, after)
        vertex.co = inverse @ Vector((world.x, world.y, blended_z))
    patch_obj.data.update()
    patch_obj["milestone_6_feather"] = True
    patch_obj["feather_width_m"] = float(patch["edge_feather_m"])
    patch_obj["displacement_feathered"] = True
    return {"max_displacement_before_m": before_max, "max_displacement_after_m": after_max, "raycast_fallback_vertices": misses, "fallback_plane_samples": plane_samples, "fallback_plane_used": use_plane, "feather_width_m": float(patch["edge_feather_m"])}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    settings = dict(config["feather_m6"])
    requested = {f"TerrainPatch_{suffix.strip().upper()}" for suffix in args.patches.split(",") if suffix.strip()}
    patches = [item for item in config["patches"] if item["id"] in requested]
    if not patches:
        raise RuntimeError(f"No requested patches found: {sorted(requested)}")
    terrain = bpy.data.objects.get(config["terrain_object"])
    if terrain is None:
        raise RuntimeError("Missing MartianTerrain base mesh")
    terrain_bvh = surface_bvh(terrain)
    reports = []
    for patch in patches:
        patch_id = patch["id"]
        patch_obj = bpy.data.objects.get(patch_id)
        material = bpy.data.materials.get(f"{patch_id}_Material")
        gravel = bpy.data.objects.get(f"{patch_id}_Gravel")
        if patch_obj is None or material is None or gravel is None:
            raise RuntimeError(f"Missing M5 prerequisites for {patch_id}")
        material_report = feather_material(material, patch_id)
        gravel_report = rebuild_gravel_with_feather(gravel, patch_obj, patch, settings)
        displacement_report = feather_displacement(patch_obj, terrain, patch, terrain_bvh)
        reports.append({"patch_id": patch_id, "material": material_report, "gravel": gravel_report, "displacement": displacement_report})

    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            scene["terrain_detail_milestone"] = "6-feather-transition"
            scene["milestone_6_feather"] = True
            scene["collision_enabled"] = False
    root = bpy.data.objects.get(config["rover_root"])
    if root is not None:
        root["milestone_6_feather"] = True
        root["milestone_6_source_milestone"] = "5-lighting-render_from_m3"

    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "milestone": "6-feather-transition",
        "source_scene": args.source_label,
        "output_blend": str(output_blend),
        "scope": [entry["patch_id"] for entry in reports],
        "settings": settings,
        "feather_width_m": {entry["patch_id"]: next(item["edge_feather_m"] for item in patches if item["id"] == entry["patch_id"]) for entry in reports},
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
    print(f"Built Milestone 6 feather transition: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
