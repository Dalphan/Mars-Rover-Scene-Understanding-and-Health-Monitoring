from __future__ import annotations

import argparse
import bpy
import json
import math
import random
import sys
from pathlib import Path

from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build the deterministic pilot terrain detail patch.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--render", choices=("none", "all", "overview", "zenith", "normal", "perforation"), default="none")
    return parser.parse_args(values)


def configure_scene(scene: bpy.types.Scene) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 40
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def terrain_bvh(terrain: bpy.types.Object) -> BVHTree:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(terrain, depsgraph)


def terrain_hit(terrain: bpy.types.Object, bvh: BVHTree, x: float, y: float) -> tuple[float, Vector]:
    inverse = terrain.matrix_world.inverted()
    origin_world = Vector((x, y, 10.0))
    origin = inverse @ origin_world
    direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, normal, _, _ = bvh.ray_cast(origin, direction, 20.0)
    if hit is None:
        raise RuntimeError(f"Terrain raycast missed at ({x:.4f}, {y:.4f})")
    hit_world = terrain.matrix_world @ hit
    normal_world = (terrain.matrix_world.to_3x3() @ normal).normalized()
    return hit_world.z, normal_world


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def patch_feather(local_x: float, local_y: float, patch: dict) -> float:
    half = patch["size_m"] * 0.5
    edge = min(half - abs(local_x), half - abs(local_y))
    return smoothstep(edge / patch["edge_feather_m"])


def terrain_noise(x: float, y: float, seed: int, scale: float) -> float:
    return math.sin((x + seed * 0.071) * scale) * math.cos((y - seed * 0.113) * scale * 0.83)


def patch_displacement(local_x: float, local_y: float, patch: dict, seed: int) -> float:
    feather = patch_feather(local_x, local_y, patch)
    broad = terrain_noise(local_x, local_y, seed, 7.5) * patch["broad_amplitude_m"]
    medium = terrain_noise(local_x, local_y, seed + 31, 27.0) * patch["medium_amplitude_m"]
    displacement = (broad + medium) * feather

    # Static contact depression aligned with the observed wheel footprint.
    length = patch["contact_length_m"] * 0.5
    width = patch["contact_width_m"] * 0.5
    ellipse = (local_x / max(length, 1e-6)) ** 2 + (local_y / max(width, 1e-6)) ** 2
    if ellipse < 1.0:
        depression = (1.0 - smoothstep(ellipse)) * patch["contact_depression_m"]
        berm = smoothstep(max(0.0, ellipse - 1.0) / 1.2) * patch["contact_berm_m"]
        displacement -= depression
        displacement += berm * feather
    return displacement + patch["edge_offset_m"]


def create_patch_mesh(
    terrain: bpy.types.Object,
    bvh: BVHTree,
    patch: dict,
    center_x: float,
    center_y: float,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    size = patch["size_m"]
    step = patch["grid_step_m"]
    count = int(round(size / step))
    vertices: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    feather_values: list[float] = []
    half = size * 0.5
    for row in range(count + 1):
        y = center_y - half + row * step
        for col in range(count + 1):
            x = center_x - half + col * step
            z, _ = terrain_hit(terrain, bvh, x, y)
            z += patch_displacement(x - center_x, y - center_y, patch, patch["seed"])
            vertices.append((x, y, z))
            uvs.append(((x + 25.0) / 50.0, (y + 25.0) / 50.0))
            feather_values.append(patch_feather(x - center_x, y - center_y, patch))
    faces: list[tuple[int, int, int, int]] = []
    for row in range(count):
        for col in range(count):
            a = row * (count + 1) + col
            faces.append((a, a + 1, a + count + 2, a + count + 1))
    mesh = bpy.data.meshes.new(f"{patch['id']}_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            uv_layer.data[loop_index].uv = uvs[mesh.loops[loop_index].vertex_index]
    feather_attribute = mesh.attributes.new(name="PatchFeather", type="FLOAT", domain="POINT")
    for index, value in enumerate(feather_values):
        feather_attribute.data[index].value = value
    obj = bpy.data.objects.new(patch["id"], mesh)
    collection.objects.link(obj)
    return obj


def remove_base_faces_under_patch(terrain: bpy.types.Object, patch: dict, center_x: float, center_y: float) -> None:
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(terrain.data)
    # Keep one base-terrain cell under the patch perimeter so the overlay
    # cannot expose a square hole or a visible seam.
    half = patch["size_m"] * 0.5 - patch["grid_step_m"] * 0.5
    inverse = terrain.matrix_world.inverted()
    for face in list(bm.faces):
        center = terrain.matrix_world @ face.calc_center_median()
        if abs(center.x - center_x) <= half and abs(center.y - center_y) <= half:
            bm.faces.remove(face)
    bm.to_mesh(terrain.data)
    bm.free()
    terrain.data.update()


def create_rock_mesh(terrain: bpy.types.Object, bvh: BVHTree, patch: dict, center_x: float, center_y: float, collection: bpy.types.Collection) -> bpy.types.Object:
    rng = random.Random(patch["seed"])
    half = patch["size_m"] * 0.5
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    wheel_x, wheel_y = center_x, center_y
    rock_count = patch["rock_count"]
    for index in range(rock_count):
        for _ in range(200):
            x = center_x + rng.uniform(-half + 0.10, half - 0.10)
            y = center_y + rng.uniform(-half + 0.10, half - 0.10)
            if ((x - wheel_x) / 0.48) ** 2 + ((y - wheel_y) / 0.38) ** 2 < 1.0:
                continue
            break
        z, normal = terrain_hit(terrain, bvh, x, y)
        radius = rng.uniform(patch["rock_radius_min_m"], patch["rock_radius_max_m"])
        if rng.random() < patch["large_rock_fraction"]:
            radius *= 1.8
        start = len(verts)
        ring_count = 6
        top = start + ring_count
        verts.append((x, y, z + radius * rng.uniform(0.55, 0.9)))
        for ring in range(ring_count):
            angle = 2.0 * math.pi * ring / ring_count
            rr = radius * rng.uniform(0.75, 1.15)
            verts.append((x + rr * math.cos(angle), y + rr * math.sin(angle), z + radius * rng.uniform(0.03, 0.18)))
        bottom = len(verts)
        verts.append((x, y, z - radius * rng.uniform(0.03, 0.12)))
        for ring in range(ring_count):
            nxt = (ring + 1) % ring_count
            faces.append((top, start + ring + 1, start + nxt + 1))
            faces.append((bottom, start + nxt + 1, start + ring + 1))
    mesh = bpy.data.meshes.new(f"{patch['id']}_Rocks_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{patch['id']}_Rocks", mesh)
    collection.objects.link(obj)
    return obj


def patch_material(base_material: bpy.types.Material, patch_id: str, bump_strength: float, bump_distance: float) -> bpy.types.Material:
    material = base_material.copy()
    material.name = f"{patch_id}_Material"
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    diffuse = nodes.get("Diffuse")
    roughness = nodes.get("Roughness")
    normal_map = nodes.get("Normal Map")
    if any(node is None for node in (diffuse, roughness, normal_map)):
        raise RuntimeError("Terrain material texture nodes are incomplete")
    for link in list(principled.inputs["Base Color"].links):
        links.remove(link)
    for link in list(principled.inputs["Roughness"].links):
        links.remove(link)
    for link in list(principled.inputs["Normal"].links):
        links.remove(link)
    texcoord = nodes.get(f"{patch_id}_TextureCoordinate") or nodes.new("ShaderNodeTexCoord")
    texcoord.name = f"{patch_id}_TextureCoordinate"
    mapping = nodes.get(f"{patch_id}_Mapping") or nodes.new("ShaderNodeMapping")
    mapping.name = f"{patch_id}_Mapping"
    coarse_noise = nodes.get(f"{patch_id}_CoarseNoise") or nodes.new("ShaderNodeTexNoise")
    coarse_noise.name = f"{patch_id}_CoarseNoise"
    coarse_noise.inputs["Scale"].default_value = 7.0
    coarse_noise.inputs["Detail"].default_value = 5.0
    coarse_noise.inputs["Roughness"].default_value = 0.72
    fine_noise = nodes.get(f"{patch_id}_FineNoise") or nodes.new("ShaderNodeTexNoise")
    fine_noise.name = f"{patch_id}_FineNoise"
    fine_noise.inputs["Scale"].default_value = 55.0
    fine_noise.inputs["Detail"].default_value = 3.0
    fine_noise.inputs["Roughness"].default_value = 0.68
    for node in (coarse_noise, fine_noise):
        for link in list(node.inputs["Vector"].links):
            links.remove(link)
        links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

    color_variation = nodes.get(f"{patch_id}_ColorVariation") or nodes.new("ShaderNodeValToRGB")
    color_variation.name = f"{patch_id}_ColorVariation"
    color_variation.color_ramp.elements[0].color = (0.95, 0.95, 0.95, 1.0)
    color_variation.color_ramp.elements[1].color = (1.05, 1.05, 1.05, 1.0)
    links.new(coarse_noise.outputs["Fac"], color_variation.inputs["Fac"])
    base_mix = nodes.get(f"{patch_id}_BaseColorMix") or nodes.new("ShaderNodeMixRGB")
    base_mix.name = f"{patch_id}_BaseColorMix"
    base_mix.blend_type = "MULTIPLY"
    base_mix.inputs["Fac"].default_value = 0.35
    links.new(diffuse.outputs["Color"], base_mix.inputs[1])
    links.new(color_variation.outputs["Color"], base_mix.inputs[2])
    links.new(base_mix.outputs["Color"], principled.inputs["Base Color"])

    roughness_variation = nodes.get(f"{patch_id}_RoughnessVariation") or nodes.new("ShaderNodeMapRange")
    roughness_variation.name = f"{patch_id}_RoughnessVariation"
    roughness_variation.inputs["From Min"].default_value = 0.0
    roughness_variation.inputs["From Max"].default_value = 1.0
    roughness_variation.inputs["To Min"].default_value = 0.88
    roughness_variation.inputs["To Max"].default_value = 0.98
    links.new(fine_noise.outputs["Fac"], roughness_variation.inputs["Value"])
    rough_mix = nodes.get(f"{patch_id}_RoughnessMix") or nodes.new("ShaderNodeMixRGB")
    rough_mix.name = f"{patch_id}_RoughnessMix"
    rough_mix.blend_type = "MIX"
    rough_mix.inputs["Fac"].default_value = 0.45
    links.new(roughness.outputs["Color"], rough_mix.inputs[1])
    links.new(roughness_variation.outputs["Result"], rough_mix.inputs[2])
    links.new(rough_mix.outputs["Color"], principled.inputs["Roughness"])

    bump = nodes.get(f"{patch_id}_Bump") or nodes.new("ShaderNodeBump")
    bump.name = f"{patch_id}_Bump"
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = bump_distance
    feather = nodes.get(f"{patch_id}_Feather") or nodes.new("ShaderNodeAttribute")
    feather.name = f"{patch_id}_Feather"
    feather.attribute_name = "PatchFeather"
    strength = nodes.get(f"{patch_id}_BumpStrength") or nodes.new("ShaderNodeMath")
    strength.name = f"{patch_id}_BumpStrength"
    strength.operation = "MULTIPLY"
    strength.inputs[1].default_value = bump_strength
    links.new(fine_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(feather.outputs["Fac"], strength.inputs[0])
    links.new(strength.outputs["Value"], bump.inputs["Strength"])
    links.new(normal_map.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def rock_material(base_material: bpy.types.Material, patch_id: str) -> bpy.types.Material:
    material = base_material.copy()
    material.name = f"{patch_id}_Rock_Material"
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    diffuse = nodes.get("Diffuse")
    normal_map = nodes.get("Normal Map")
    if diffuse is None or normal_map is None:
        raise RuntimeError("Rock material base texture nodes are incomplete")
    for link in list(principled.inputs["Base Color"].links):
        links.remove(link)
    for link in list(principled.inputs["Roughness"].links):
        links.remove(link)
    for link in list(principled.inputs["Normal"].links):
        links.remove(link)
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = f"{patch_id}_RockTextureCoordinate"
    mapping = nodes.new("ShaderNodeMapping")
    mapping.name = f"{patch_id}_RockMapping"
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = f"{patch_id}_RockNoise"
    noise.inputs["Scale"].default_value = 9.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.78
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.name = f"{patch_id}_RockVoronoi"
    voronoi.distance = "EUCLIDEAN"
    voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.inputs["Scale"].default_value = 18.0
    links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
    rock_color = nodes.new("ShaderNodeValToRGB")
    rock_color.name = f"{patch_id}_RockColorRamp"
    rock_color.color_ramp.elements[0].color = (0.10, 0.045, 0.018, 1.0)
    rock_color.color_ramp.elements[1].color = (0.36, 0.16, 0.07, 1.0)
    links.new(noise.outputs["Fac"], rock_color.inputs["Fac"])
    color_mix = nodes.new("ShaderNodeMixRGB")
    color_mix.name = f"{patch_id}_RockColorMix"
    color_mix.blend_type = "MULTIPLY"
    color_mix.inputs["Fac"].default_value = 0.55
    links.new(diffuse.outputs["Color"], color_mix.inputs[1])
    links.new(rock_color.outputs["Color"], color_mix.inputs[2])
    links.new(color_mix.outputs["Color"], principled.inputs["Base Color"])
    roughness = nodes.new("ShaderNodeMapRange")
    roughness.name = f"{patch_id}_RockRoughness"
    roughness.inputs["To Min"].default_value = 0.76
    roughness.inputs["To Max"].default_value = 0.98
    links.new(voronoi.outputs["Distance"], roughness.inputs["Value"])
    links.new(roughness.outputs["Result"], principled.inputs["Roughness"])
    bump = nodes.new("ShaderNodeBump")
    bump.name = f"{patch_id}_RockBump"
    bump.inputs["Strength"].default_value = 0.32
    bump.inputs["Distance"].default_value = 0.0025
    links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    links.new(normal_map.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return material


def add_rock_micro_geometry(rocks: bpy.types.Object, patch_id: str) -> None:
    subdivision = rocks.modifiers.get(f"{patch_id}_RockSubdivision") or rocks.modifiers.new(f"{patch_id}_RockSubdivision", "SUBSURF")
    subdivision.subdivision_type = "SIMPLE"
    subdivision.levels = 1
    subdivision.render_levels = 1
    texture = bpy.data.textures.get(f"{patch_id}_RockDisplacementTexture") or bpy.data.textures.new(f"{patch_id}_RockDisplacementTexture", type="CLOUDS")
    texture.noise_scale = 0.055
    texture.noise_depth = 2
    displacement = rocks.modifiers.get(f"{patch_id}_RockDisplacement") or rocks.modifiers.new(f"{patch_id}_RockDisplacement", "DISPLACE")
    displacement.texture = texture
    displacement.texture_coords = "GLOBAL"
    displacement.strength = 0.006
    displacement.mid_level = 0.5


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = config["terrain_detail"]
    output_dir = args.output_dir.expanduser().resolve()
    renders_dir = output_dir / "renders"
    reports_dir = output_dir / "reports"
    renders_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    simulator = bpy.data.scenes.get("Simulator")
    normal = bpy.data.scenes.get("Normal")
    perforation = bpy.data.scenes.get("Perforation")
    terrain = bpy.data.objects.get(settings["terrain_object"])
    root = bpy.data.objects.get(settings["rover_root"])
    wheel = bpy.data.objects.get(settings["observed_wheel"])
    sun = bpy.data.objects.get(settings["sun_object"])
    world = bpy.data.worlds.get(settings["world_object"])
    wheel_camera = bpy.data.objects.get("WheelCamera")
    chase = bpy.data.objects.get("SimulatorCamera")
    if any(value is None for value in (simulator, normal, perforation, terrain, root, wheel, sun, world, wheel_camera, chase)):
        raise RuntimeError("Detailed terrain pilot prerequisites are incomplete")
    scenes = [simulator, normal, perforation]
    for scene in scenes:
        configure_scene(scene)
        scene.world = world
        if sun.name not in scene.objects:
            scene.collection.objects.link(sun)
        for obj in scene.objects:
            if obj.type == "LIGHT" and obj != sun:
                obj.hide_render = True
    base_material = terrain.data.materials[0]
    bvh = terrain_bvh(terrain)
    patches = settings["patches"]
    altitude = math.radians(settings["solar_altitude_degrees"])
    azimuth = math.radians(settings["solar_azimuth_degrees"])
    horizontal_distance = 50.0
    sun.location = Vector((
        horizontal_distance * math.cos(azimuth),
        horizontal_distance * math.sin(azimuth),
        horizontal_distance * math.tan(altitude),
    ))
    sun.data.energy = settings["solar_energy"]
    sun.data.angle = math.radians(settings["solar_angle_degrees"])
    look_at(sun, Vector((0.0, 0.0, 0.0)))
    bbox = [wheel.matrix_world @ Vector(corner) for corner in wheel.bound_box]
    anchor_x = sum(point.x for point in bbox) / len(bbox)
    anchor_y = sum(point.y for point in bbox) / len(bbox)
    anchor_surface, _ = terrain_hit(terrain, bvh, anchor_x, anchor_y)
    root_base_location = root.location.copy()
    root["terrain_detail_pilot_pose"] = json.dumps({"x": anchor_x, "y": anchor_y, "yaw_degrees": math.degrees(root.rotation_euler.z)}, sort_keys=True)
    shared_lighting = bpy.data.collections.get("MARS_CLEAR_DAY_LIGHTING") or bpy.data.collections.new("MARS_CLEAR_DAY_LIGHTING")
    if shared_lighting.name not in simulator.collection.children:
        simulator.collection.children.link(shared_lighting)
    simulator_lighting = bpy.data.collections.get("SIMULATOR_SOLAR_LIGHTING")
    for scene in scenes:
        if shared_lighting.name not in scene.collection.children:
            scene.collection.children.link(shared_lighting)
        if simulator_lighting is not None and simulator_lighting.name in scene.collection.children:
            scene.collection.children.unlink(simulator_lighting)
    if sun.name not in shared_lighting.objects:
        shared_lighting.objects.link(sun)

    detail_root = bpy.data.collections.get("TERRAIN_DETAIL_PATCHES") or bpy.data.collections.new("TERRAIN_DETAIL_PATCHES")
    if detail_root.name not in simulator.collection.children:
        simulator.collection.children.link(detail_root)
    for scene in (normal, perforation):
        if detail_root.name not in scene.collection.children:
            scene.collection.children.link(detail_root)
    patch_results = []
    for patch in patches:
        if "center_world" in patch:
            center_x, center_y = patch["center_world"]
        else:
            center_x, center_y = anchor_x, anchor_y
        surface_z, surface_normal = terrain_hit(terrain, bvh, center_x, center_y)
        patch_collection = bpy.data.collections.get(patch["id"]) or bpy.data.collections.new(patch["id"])
        if patch_collection.name not in detail_root.children:
            detail_root.children.link(patch_collection)
        remove_base_faces_under_patch(terrain, patch, center_x, center_y)
        patch_obj = create_patch_mesh(terrain, bvh, patch, center_x, center_y, patch_collection)
        patch_obj.data.materials.append(patch_material(base_material, patch["id"], patch["bump_strength"], patch["bump_distance_m"]))
        rocks = create_rock_mesh(terrain, bvh, patch, center_x, center_y, patch_collection)
        rocks.data.materials.append(rock_material(base_material, patch["id"]))
        add_rock_micro_geometry(rocks, patch["id"])
        patch_results.append({
            "patch": patch,
            "patch_obj": patch_obj,
            "rocks": rocks,
            "center_x": center_x,
            "center_y": center_y,
            "surface_z": surface_z,
            "surface_normal": surface_normal,
        })
    simulator["terrain_detail_status"] = "multi_patch_built_pending_visual_approval"
    simulator["terrain_detail_patch_id"] = ",".join(result["patch"]["id"] for result in patch_results)
    simulator["terrain_detail_patch_seed"] = ",".join(str(result["patch"]["seed"]) for result in patch_results)
    simulator["terrain_shared_with_pair_scenes"] = True
    simulator["collision_enabled"] = False
    simulator["terrain_collision_enabled"] = False
    bpy.context.view_layer.update()
    # Rendering of individual patches is handled by render_terrain_detail_batches.py
    # so that the build remains short and reproducible on constrained Blender runs.
    simulator.camera = chase
    output_blend = output_dir / "diagnostics" / "rover_simulator_detailed.blend"
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "patches": [
            {
                "patch_id": result["patch"]["id"],
                "profile": result["patch"]["profile"],
                "seed": result["patch"]["seed"],
                "size_m": result["patch"]["size_m"],
                "grid_step_m": result["patch"]["grid_step_m"],
                "center_world": [result["center_x"], result["center_y"]],
                "surface_z": result["surface_z"],
                "surface_normal": list(result["surface_normal"]),
                "objects": {"patch": result["patch_obj"].name, "rocks": result["rocks"].name, "terrain": terrain.name},
            }
            for result in patch_results
        ],
        "pair_context_identical": True,
        "collision_enabled": False,
        "lighting": {
            "solar_altitude_degrees": settings["solar_altitude_degrees"],
            "solar_azimuth_degrees": settings["solar_azimuth_degrees"],
            "solar_energy": settings["solar_energy"],
            "solar_angle_degrees": settings["solar_angle_degrees"],
        },
        "render_pattern": "patch_{A|B|C}_{overview|zenith|normal|perforation}.png",
        "output_blend": str(output_blend),
    }
    (reports_dir / "terrain_detail_pilot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built terrain detail pilot at ({center_x:.4f}, {center_y:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
