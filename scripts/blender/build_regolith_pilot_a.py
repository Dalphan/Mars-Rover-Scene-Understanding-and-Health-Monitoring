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


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build a reversible regolith realism pilot on TerrainPatch_A.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--patches", default="A", help="Comma-separated patch suffixes to process, for example A,B,C")
    parser.add_argument("--source-label", default="rover_simulator_detailed.blend")
    parser.add_argument("--revert-backup", default="outputs/rover_simulator/curiosity_middle_right/diagnostics/rover_simulator_detailed_backup_before_regolith_pilotA.blend")
    return parser.parse_args(values)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def smoothstep_range(edge0: float, edge1: float, value: float) -> float:
    if abs(edge1 - edge0) < 1e-8:
        return 1.0 if value >= edge1 else 0.0
    return smoothstep((value - edge0) / (edge1 - edge0))


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


def append_boulder(
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    smooth_flags: list[bool],
    center: Vector,
    normal: Vector,
    radius: float,
    family: str,
    rng: random.Random,
) -> None:
    yaw = rng.uniform(0.0, 2.0 * math.pi)
    tangent_a, tangent_b = tangent_basis(normal, yaw)
    if family == "flat":
        rx, ry, height, smooth = radius * 1.45, radius * 0.82, radius * 0.38, False
    elif family == "rounded":
        rx, ry, height, smooth = radius * 1.00, radius * 0.90, radius * 0.82, True
    elif family == "shard":
        rx, ry, height, smooth = radius * 1.75, radius * 0.45, radius * 0.54, False
    else:
        rx, ry, height, smooth = radius * 1.05, radius * 0.82, radius * 1.05, False

    ring_count = 7 if family != "rounded" else 8
    base = len(vertices)
    bottom = center - normal * (radius * rng.uniform(0.12, 0.24))
    vertices.append(tuple(bottom))
    lower_indices: list[int] = []
    upper_indices: list[int] = []
    for ring in range(ring_count):
        angle = 2.0 * math.pi * ring / ring_count
        jitter = rng.uniform(0.78, 1.20)
        offset = tangent_a * (math.cos(angle) * rx * jitter) + tangent_b * (math.sin(angle) * ry * jitter)
        lower = center + offset + normal * rng.uniform(-0.005, 0.025) * radius
        upper = center + offset * rng.uniform(0.55, 0.78) + normal * height * rng.uniform(0.42, 0.70)
        lower_indices.append(len(vertices))
        vertices.append(tuple(lower))
        upper_indices.append(len(vertices))
        vertices.append(tuple(upper))
    top = len(vertices)
    vertices.append(tuple(center + normal * height * rng.uniform(0.88, 1.12)))
    for ring in range(ring_count):
        nxt = (ring + 1) % ring_count
        faces.append((base, lower_indices[nxt], lower_indices[ring]))
        smooth_flags.append(False)
        faces.append((lower_indices[ring], lower_indices[nxt], upper_indices[nxt]))
        smooth_flags.append(smooth)
        faces.append((lower_indices[ring], upper_indices[nxt], upper_indices[ring]))
        smooth_flags.append(smooth)
        faces.append((upper_indices[ring], upper_indices[nxt], top))
        smooth_flags.append(smooth)


def mesh_from_parts(name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], smooth_flags: list[bool]) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon, smooth in zip(mesh.polygons, smooth_flags):
        polygon.use_smooth = smooth
    return mesh


def replace_rock_mesh(rocks: bpy.types.Object, patch_obj: bpy.types.Object, patch: dict, center_x: float, center_y: float, wheel_yaw: float) -> dict:
    pilot = patch["regolith_pilot"]
    rng = random.Random(patch["seed"] + 900)
    bvh = surface_bvh(patch_obj)
    points = poisson_points(
        center_x,
        center_y,
        patch["size_m"],
        pilot.get("rock_min_distance_m", 0.16),
        pilot["rock_family_count"],
        patch["seed"] + 901,
        (center_x, center_y, 1.0),
    )
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    smooth_flags: list[bool] = []
    face_family: list[int] = []
    families = ("angular", "flat", "rounded", "shard")
    for index, (x, y) in enumerate(points):
        point, normal = surface_hit(patch_obj, bvh, x, y)
        radius = rng.uniform(
            patch["rock_radius_min_m"] * pilot.get("rock_min_scale", 1.0),
            patch["rock_radius_max_m"] * pilot.get("rock_max_scale", 0.82),
        )
        if rng.random() < patch["large_rock_fraction"]:
            radius *= rng.uniform(1.15, 1.55)
        family = families[index % len(families)]
        before = len(faces)
        append_boulder(vertices, faces, smooth_flags, point, normal, radius, family, rng)
        face_family.extend([index % len(families)] * (len(faces) - before))
    new_mesh = mesh_from_parts(f"{patch['id']}_Rocks_Families_Mesh", vertices, faces, smooth_flags)
    old_mesh = rocks.data
    rocks.data = new_mesh
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    for modifier in list(rocks.modifiers):
        rocks.modifiers.remove(modifier)
    for material in list(rocks.data.materials):
        rocks.data.materials.pop(index=0)
    materials = rock_family_materials(patch["id"])
    for material in materials:
        rocks.data.materials.append(material)
    for polygon, material_index in zip(rocks.data.polygons, face_family):
        polygon.material_index = material_index
    subdivision = rocks.modifiers.new(f"{patch['id']}_RockSubdivision", "SUBSURF")
    subdivision.subdivision_type = "SIMPLE"
    subdivision.levels = 1
    subdivision.render_levels = 1
    texture = bpy.data.textures.get(f"{patch['id']}_FamilyDisplacement") or bpy.data.textures.new(f"{patch['id']}_FamilyDisplacement", type="CLOUDS")
    texture.noise_scale = 0.04
    texture.noise_depth = 2
    displacement = rocks.modifiers.new(f"{patch['id']}_FamilyDisplacement", "DISPLACE")
    displacement.texture = texture
    displacement.texture_coords = "GLOBAL"
    displacement.strength = 0.0045
    displacement.mid_level = 0.5
    rocks["detail_role"] = "rock_families"
    rocks["deterministic_seed"] = patch["seed"] + 900
    return {"rock_count": len(points), "families": list(families)}


def rock_family_materials(patch_id: str) -> list[bpy.types.Material]:
    base = bpy.data.materials.get(f"{patch_id}_Rock_Material")
    if base is None:
        raise RuntimeError(f"Missing base rock material for {patch_id}")
    palettes = [
        ((0.10, 0.042, 0.016, 1.0), (0.34, 0.14, 0.050, 1.0)),
        ((0.13, 0.055, 0.020, 1.0), (0.43, 0.19, 0.070, 1.0)),
        ((0.075, 0.032, 0.014, 1.0), (0.29, 0.12, 0.042, 1.0)),
        ((0.16, 0.070, 0.024, 1.0), (0.48, 0.22, 0.075, 1.0)),
    ]
    materials: list[bpy.types.Material] = []
    for index, palette in enumerate(palettes):
        name = f"{patch_id}_RockFamily_{index + 1:02d}"
        material = bpy.data.materials.get(name) or base.copy()
        material.name = name
        ramp = material.node_tree.nodes.get(f"{patch_id}_RockColorRamp")
        if ramp is not None:
            ramp.color_ramp.elements[0].color = palette[0]
            ramp.color_ramp.elements[1].color = palette[1]
        materials.append(material)
    return materials


def create_gravel(patch_obj: bpy.types.Object, patch: dict, center_x: float, center_y: float) -> dict:
    pilot = patch["regolith_pilot"]
    bvh = surface_bvh(patch_obj)
    rng = random.Random(patch["seed"] + 1200)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    smooth_flags: list[bool] = []
    total = 0
    for band_index, band in enumerate(pilot["gravel_bands"]):
        points = poisson_points(
            center_x,
            center_y,
            patch["size_m"],
            band["min_distance_m"],
            band["count"],
            patch["seed"] + 1201 + band_index,
            (center_x, center_y, 0.82),
        )
        for x, y in points:
            point, normal = surface_hit(patch_obj, bvh, x, y)
            radius = rng.uniform(band["radius_min_m"], band["radius_max_m"])
            family = "flat" if band_index == 2 else ("rounded" if rng.random() < 0.35 else "angular")
            before = len(faces)
            append_boulder(vertices, faces, smooth_flags, point, normal, radius, family, rng)
            total += 1
    gravel = bpy.data.objects.get(f"{patch['id']}_Gravel")
    if gravel is None:
        gravel = bpy.data.objects.new(f"{patch['id']}_Gravel", mesh_from_parts(f"{patch['id']}_Gravel_Mesh", vertices, faces, smooth_flags))
        patch_collection = bpy.data.collections[patch["id"]]
        patch_collection.objects.link(gravel)
    else:
        old_mesh = gravel.data
        gravel.data = mesh_from_parts(f"{patch['id']}_Gravel_Mesh", vertices, faces, smooth_flags)
        if old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    material = bpy.data.materials.get(f"{patch['id']}_Gravel_Material")
    if material is None:
        base = bpy.data.materials.get(f"{patch['id']}_RockFamily_02") or bpy.data.materials.get(f"{patch['id']}_Rock_Material")
        if base is None:
            raise RuntimeError(f"Missing rock material for gravel {patch['id']}")
        material = base.copy()
        material.name = f"{patch['id']}_Gravel_Material"
        principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        principled.inputs["Roughness"].default_value = 0.96
    gravel.data.materials.clear()
    gravel.data.materials.append(material)
    gravel["detail_role"] = "poisson_gravel"
    gravel["deterministic_seed"] = patch["seed"] + 1200
    gravel["band_count"] = len(pilot["gravel_bands"])
    gravel["total_instances_as_mesh_islands"] = total
    return {"gravel_count": total, "bands": pilot["gravel_bands"]}


def update_contact_mask_and_imprint(patch_obj: bpy.types.Object, patch: dict, center_x: float, center_y: float, wheel_yaw: float) -> dict:
    pilot = patch["regolith_pilot"]
    mesh = patch_obj.data
    contact = mesh.attributes.get("ContactDust") or mesh.attributes.new("ContactDust", type="FLOAT", domain="POINT")
    length = patch["contact_length_m"] * 0.5
    width = patch["contact_width_m"] * 0.5
    inv = patch_obj.matrix_world.inverted().to_3x3()
    for index, vertex in enumerate(mesh.vertices):
        point = patch_obj.matrix_world @ vertex.co
        dx = point.x - center_x
        dy = point.y - center_y
        u = math.cos(wheel_yaw) * dx + math.sin(wheel_yaw) * dy
        v = -math.sin(wheel_yaw) * dx + math.cos(wheel_yaw) * dy
        ellipse = math.sqrt((u / max(length, 1e-5)) ** 2 + (v / max(width, 1e-5)) ** 2)
        inner = 1.0 - smoothstep_range(0.55, 1.0, ellipse)
        rim = smoothstep_range(0.82, 1.02, ellipse) * (1.0 - smoothstep_range(1.02, 1.48, ellipse))
        deterministic_variation = 0.82 + 0.18 * (0.5 + 0.5 * math.sin((point.x + 3.0 * point.y) * 37.0 + patch["seed"]))
        dust_value = max(0.0, min(1.0, (0.65 * rim + 0.15 * inner) * deterministic_variation * pilot["contact_dust_strength"]))
        contact.data[index].value = dust_value
        feather_attribute = mesh.attributes.get("PatchFeather")
        feather = feather_attribute.data[index].value if feather_attribute else 1.0
        delta_world = (-pilot["contact_extra_depression_m"] * inner + pilot["contact_extra_berm_m"] * rim) * feather
        vertex.co += inv @ Vector((0.0, 0.0, delta_world))
    patch_obj["contact_imprint_enabled"] = True
    patch_obj["contact_dust_attribute"] = "ContactDust"
    patch_obj["contact_imprint_yaw_degrees"] = math.degrees(wheel_yaw)
    return {"attribute": "ContactDust", "extra_depression_m": pilot["contact_extra_depression_m"], "dust_strength": pilot["contact_dust_strength"]}


def update_regolith_material(material: bpy.types.Material, patch: dict) -> dict:
    pilot = patch["regolith_pilot"]
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    diffuse = nodes.get("Diffuse")
    roughness = nodes.get("Roughness")
    normal_map = nodes.get("Normal Map")
    if any(node is None for node in (diffuse, roughness, normal_map)):
        raise RuntimeError("Terrain material texture nodes are incomplete")
    for socket_name in ("Base Color", "Roughness", "Normal"):
        for link in list(principled.inputs[socket_name].links):
            links.remove(link)
    prefix = f"{patch['id']}_Regolith_"
    texcoord = nodes.get(prefix + "TextureCoordinate") or nodes.new("ShaderNodeTexCoord")
    texcoord.name = prefix + "TextureCoordinate"
    mapping = nodes.get(prefix + "Mapping") or nodes.new("ShaderNodeMapping")
    mapping.name = prefix + "Mapping"
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    layers = []
    for label, scale, detail, rough in (
        ("Macro", pilot["macro_noise_scale"], 4.0, 0.78),
        ("Meso", pilot["meso_noise_scale"], 3.0, 0.72),
        ("Micro", pilot["micro_noise_scale"], 2.0, 0.64),
    ):
        noise = nodes.get(prefix + label + "Noise") or nodes.new("ShaderNodeTexNoise")
        noise.name = prefix + label + "Noise"
        noise.inputs["Scale"].default_value = scale
        noise.inputs["Detail"].default_value = detail
        noise.inputs["Roughness"].default_value = rough
        for link in list(noise.inputs["Vector"].links):
            links.remove(link)
        links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
        layers.append(noise)

    macro_ramp = nodes.get(prefix + "MacroColor") or nodes.new("ShaderNodeValToRGB")
    macro_ramp.name = prefix + "MacroColor"
    macro_ramp.color_ramp.elements[0].color = (0.78, 0.64, 0.48, 1.0)
    macro_ramp.color_ramp.elements[1].color = (1.00, 0.90, 0.70, 1.0)
    links.new(layers[0].outputs["Fac"], macro_ramp.inputs["Fac"])
    macro_mix = nodes.get(prefix + "MacroColorMix") or nodes.new("ShaderNodeMixRGB")
    macro_mix.name = prefix + "MacroColorMix"
    macro_mix.blend_type = "MULTIPLY"
    macro_mix.inputs["Fac"].default_value = 0.28
    links.new(diffuse.outputs["Color"], macro_mix.inputs[1])
    links.new(macro_ramp.outputs["Color"], macro_mix.inputs[2])
    meso_ramp = nodes.get(prefix + "MesoColor") or nodes.new("ShaderNodeValToRGB")
    meso_ramp.name = prefix + "MesoColor"
    meso_ramp.color_ramp.elements[0].color = (0.84, 0.70, 0.52, 1.0)
    meso_ramp.color_ramp.elements[1].color = (1.00, 0.92, 0.76, 1.0)
    links.new(layers[1].outputs["Fac"], meso_ramp.inputs["Fac"])
    meso_mix = nodes.get(prefix + "MesoColorMix") or nodes.new("ShaderNodeMixRGB")
    meso_mix.name = prefix + "MesoColorMix"
    meso_mix.blend_type = "MULTIPLY"
    meso_mix.inputs["Fac"].default_value = 0.22
    links.new(macro_mix.outputs["Color"], meso_mix.inputs[1])
    links.new(meso_ramp.outputs["Color"], meso_mix.inputs[2])

    contact = nodes.get(prefix + "ContactDust") or nodes.new("ShaderNodeAttribute")
    contact.name = prefix + "ContactDust"
    contact.attribute_name = "ContactDust"
    dust_factor = nodes.get(prefix + "DustFactor") or nodes.new("ShaderNodeMath")
    dust_factor.name = prefix + "DustFactor"
    dust_factor.operation = "MULTIPLY"
    dust_factor.inputs[1].default_value = 0.52
    links.new(contact.outputs["Fac"], dust_factor.inputs[0])
    dust_mix = nodes.get(prefix + "DustColorMix") or nodes.new("ShaderNodeMixRGB")
    dust_mix.name = prefix + "DustColorMix"
    dust_mix.blend_type = "MIX"
    dust_mix.inputs[2].default_value = (0.48, 0.28, 0.16, 1.0)
    links.new(dust_factor.outputs["Value"], dust_mix.inputs["Fac"])
    links.new(meso_mix.outputs["Color"], dust_mix.inputs[1])
    links.new(dust_mix.outputs["Color"], principled.inputs["Base Color"])

    macro_rough = nodes.get(prefix + "MacroRoughness") or nodes.new("ShaderNodeMapRange")
    macro_rough.name = prefix + "MacroRoughness"
    macro_rough.inputs["To Min"].default_value = 0.79
    macro_rough.inputs["To Max"].default_value = 0.94
    links.new(layers[0].outputs["Fac"], macro_rough.inputs["Value"])
    meso_rough = nodes.get(prefix + "MesoRoughness") or nodes.new("ShaderNodeMapRange")
    meso_rough.name = prefix + "MesoRoughness"
    meso_rough.inputs["To Min"].default_value = 0.84
    meso_rough.inputs["To Max"].default_value = 0.99
    links.new(layers[1].outputs["Fac"], meso_rough.inputs["Value"])
    rough_mix = nodes.get(prefix + "RoughnessMix") or nodes.new("ShaderNodeMixRGB")
    rough_mix.name = prefix + "RoughnessMix"
    rough_mix.blend_type = "MIX"
    rough_mix.inputs["Fac"].default_value = 0.35
    links.new(roughness.outputs["Color"], rough_mix.inputs[1])
    links.new(macro_rough.outputs["Result"], rough_mix.inputs[2])
    rough_mix_2 = nodes.get(prefix + "RoughnessMixMeso") or nodes.new("ShaderNodeMixRGB")
    rough_mix_2.name = prefix + "RoughnessMixMeso"
    rough_mix_2.blend_type = "MIX"
    rough_mix_2.inputs["Fac"].default_value = 0.45
    links.new(rough_mix.outputs["Color"], rough_mix_2.inputs[1])
    links.new(meso_rough.outputs["Result"], rough_mix_2.inputs[2])
    links.new(rough_mix_2.outputs["Color"], principled.inputs["Roughness"])

    macro_height = nodes.get(prefix + "MacroHeight") or nodes.new("ShaderNodeMath")
    macro_height.name = prefix + "MacroHeight"
    macro_height.operation = "MULTIPLY"
    macro_height.inputs[1].default_value = 0.35
    links.new(layers[0].outputs["Fac"], macro_height.inputs[0])
    meso_height = nodes.get(prefix + "MesoHeight") or nodes.new("ShaderNodeMath")
    meso_height.name = prefix + "MesoHeight"
    meso_height.operation = "MULTIPLY"
    meso_height.inputs[1].default_value = 0.50
    links.new(layers[1].outputs["Fac"], meso_height.inputs[0])
    micro_height = nodes.get(prefix + "MicroHeight") or nodes.new("ShaderNodeMath")
    micro_height.name = prefix + "MicroHeight"
    micro_height.operation = "MULTIPLY"
    micro_height.inputs[1].default_value = 0.22
    links.new(layers[2].outputs["Fac"], micro_height.inputs[0])
    height_sum = nodes.get(prefix + "HeightSum") or nodes.new("ShaderNodeMath")
    height_sum.name = prefix + "HeightSum"
    height_sum.operation = "ADD"
    links.new(macro_height.outputs["Value"], height_sum.inputs[0])
    links.new(meso_height.outputs["Value"], height_sum.inputs[1])
    height_sum_2 = nodes.get(prefix + "HeightSumMicro") or nodes.new("ShaderNodeMath")
    height_sum_2.name = prefix + "HeightSumMicro"
    height_sum_2.operation = "ADD"
    links.new(height_sum.outputs["Value"], height_sum_2.inputs[0])
    links.new(micro_height.outputs["Value"], height_sum_2.inputs[1])
    feather = nodes.get(prefix + "PatchFeather") or nodes.new("ShaderNodeAttribute")
    feather.name = prefix + "PatchFeather"
    feather.attribute_name = "PatchFeather"
    bump_strength = nodes.get(prefix + "BumpStrength") or nodes.new("ShaderNodeMath")
    bump_strength.name = prefix + "BumpStrength"
    bump_strength.operation = "MULTIPLY"
    bump_strength.inputs[1].default_value = pilot["bump_strength"]
    links.new(feather.outputs["Fac"], bump_strength.inputs[0])
    bump = nodes.get(prefix + "Bump") or nodes.new("ShaderNodeBump")
    bump.name = prefix + "Bump"
    bump.inputs["Distance"].default_value = pilot["bump_distance_m"]
    links.new(height_sum_2.outputs["Value"], bump.inputs["Height"])
    links.new(bump_strength.outputs["Value"], bump.inputs["Strength"])
    links.new(normal_map.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    material["regolith_layers"] = "macro_meso_micro"
    material["regolith_noise_scales"] = json.dumps([pilot["macro_noise_scale"], pilot["meso_noise_scale"], pilot["micro_noise_scale"]])
    material["contact_dust_attribute"] = "ContactDust"
    return {"noise_scales": [pilot["macro_noise_scale"], pilot["meso_noise_scale"], pilot["micro_noise_scale"]], "bump_strength": pilot["bump_strength"]}


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    requested = {f"TerrainPatch_{suffix.strip().upper()}" for suffix in args.patches.split(",") if suffix.strip()}
    patches = [item for item in config["patches"] if item["id"] in requested]
    if not patches:
        raise RuntimeError(f"No requested patches found: {sorted(requested)}")
    terrain = bpy.data.objects.get(config["terrain_object"])
    root = bpy.data.objects.get(config["rover_root"])
    if terrain is None or root is None:
        raise RuntimeError("The existing detailed scene is missing terrain or RoverRoot")
    wheel_yaw = root.rotation_euler.z
    patch_reports = []
    for patch in patches:
        if "regolith_pilot" not in patch:
            raise RuntimeError(f"{patch['id']} has no regolith_pilot configuration")
        patch_obj = bpy.data.objects.get(patch["id"])
        rocks = bpy.data.objects.get(f"{patch['id']}_Rocks")
        material = bpy.data.materials.get(f"{patch['id']}_Material")
        if any(value is None for value in (patch_obj, rocks, material)):
            raise RuntimeError(f"The existing detailed scene is missing prerequisites for {patch['id']}")
        center_x, center_y = patch_center(patch_obj)
        contact_report = update_contact_mask_and_imprint(patch_obj, patch, center_x, center_y, wheel_yaw)
        material_report = update_regolith_material(material, patch)
        rock_report = replace_rock_mesh(rocks, patch_obj, patch, center_x, center_y, wheel_yaw)
        gravel_report = create_gravel(patch_obj, patch, center_x, center_y)
        patch_obj["regolith_pilot_enabled"] = True
        patch_obj["regolith_pilot_features"] = "three_scale_material,poisson_gravel,rock_families,contact_dust_imprint"
        patch_reports.append({
            "patch_id": patch["id"],
            "center_world": [center_x, center_y],
            "regolith_material": material_report,
            "poisson_gravel": gravel_report,
            "rock_families": rock_report,
            "contact_dust_imprint": contact_report,
        })
    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            scene["regolith_pilot_a"] = True
    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "source_scene": args.source_label,
        "output_blend": str(output_blend),
        "scope": ",".join(entry["patch_id"] for entry in patch_reports),
        "revert_backup": args.revert_backup,
        "patches": patch_reports,
        "pair_context_unchanged": True,
        "collision_enabled": False,
        "external_resources": [],
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built reversible regolith pilot A: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
