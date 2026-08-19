"""Build the metric Gale terrain, place the semantic rover and render QA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _arguments() -> tuple[Path, Path, Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--geospatial-dir", type=Path)
    parser.add_argument("--source-blend", type=Path)
    parser.add_argument("--output-dir", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    values = (
        globals().get("CONFIG_PATH") or args.config,
        globals().get("GEOSPATIAL_DIR") or args.geospatial_dir,
        globals().get("SOURCE_BLEND") or args.source_blend,
        globals().get("OUTPUT_DIR") or args.output_dir,
    )
    if any(value is None for value in values):
        raise RuntimeError("CONFIG_PATH, GEOSPATIAL_DIR, SOURCE_BLEND and OUTPUT_DIR are required")
    return tuple(Path(value).resolve() for value in values)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mesh_from_heightfield(name: str, x, y, z, with_uv: bool) -> bpy.types.Object:
    height, width = z.shape
    if len(x) != width or len(y) != height:
        raise RuntimeError(f"Heightfield axes do not match {name}")
    vertices = [(float(x[column]), float(y[row]), float(z[row, column])) for row in range(height) for column in range(width)]
    faces = []
    for row in range(height - 1):
        for column in range(width - 1):
            index = row * width + column
            faces.append((index, index + 1, index + width + 1, index + width))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    if with_uv:
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    if with_uv:
        uv_layer = mesh.uv_layers.new(name="GeoreferencedUV")
        for polygon in mesh.polygons:
            for loop_index, vertex_index in zip(polygon.loop_indices, polygon.vertices):
                column = vertex_index % width
                row = vertex_index // width
                uv_layer.data[loop_index].uv = (column / (width - 1), row / (height - 1))
    return obj


def _terrain_material(texture_path: Path, roughness: float, texture_selection: dict) -> bpy.types.Material:
    material = bpy.data.materials.new("GaleTerrainMacroAlbedo")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(str(texture_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "CLIP"
    principled.inputs["Roughness"].default_value = roughness
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    material["role"] = "HiRISE_georeferenced_color"
    material["texture_product"] = texture_selection["chosen"]
    material["color_type"] = texture_selection["color_type"]
    material["texture_path"] = str(texture_path)
    material["microtexture_enabled"] = False
    return material


def _sample_height(x_axis, y_axis, z, x: float, y: float) -> float:
    column = float(np.interp(x, x_axis, np.arange(len(x_axis))))
    row = float(np.interp(y, y_axis, np.arange(len(y_axis))))
    c0, r0 = int(math.floor(column)), int(math.floor(row))
    c1, r1 = min(c0 + 1, len(x_axis) - 1), min(r0 + 1, len(y_axis) - 1)
    wc, wr = column - c0, row - r0
    return float(z[r0, c0] * (1 - wc) * (1 - wr) + z[r0, c1] * wc * (1 - wr) + z[r1, c0] * (1 - wc) * wr + z[r1, c1] * wc * wr)


def _bbox_min_z(obj: bpy.types.Object) -> float:
    return min((obj.matrix_world @ Vector(corner)).z for corner in obj.bound_box)


def _bbox_center_xy(obj: bpy.types.Object) -> tuple[float, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    return center.x, center.y


def _place_rover(config: dict, x_axis, y_axis, z) -> dict:
    root = bpy.data.objects.get(config["rover"]["root_object"])
    if root is None:
        raise RuntimeError(f"Missing rover root {config['rover']['root_object']}")
    wheel_names = sorted(obj.name for obj in bpy.data.objects if obj.name.startswith("wheel_"))
    if len(wheel_names) != 6:
        raise RuntimeError(f"Expected six semantic wheels, found {wheel_names}")
    clearance = float(config["rover"]["clearance_m"])
    contact_samples = []
    for name in wheel_names:
        wheel = bpy.data.objects[name]
        x, y = _bbox_center_xy(wheel)
        contact_samples.append((x, y, _sample_height(x_axis, y_axis, z, x, y)))
    design = np.asarray([[x, y, 1.0] for x, y, _ in contact_samples], dtype=np.float64)
    elevations = np.asarray([height for _, _, height in contact_samples], dtype=np.float64)
    slope_x, slope_y, intercept = np.linalg.lstsq(design, elevations, rcond=None)[0]
    normal = Vector((-float(slope_x), -float(slope_y), 1.0)).normalized()
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(normal)
    bpy.context.view_layer.update()
    requirements = []
    for name in wheel_names:
        wheel = bpy.data.objects[name]
        x, y = _bbox_center_xy(wheel)
        requirements.append(_sample_height(x_axis, y_axis, z, x, y) + clearance - _bbox_min_z(wheel))
    root.location.z += max(requirements)
    bpy.context.view_layer.update()
    clearances = {}
    for name in wheel_names:
        wheel = bpy.data.objects[name]
        x, y = _bbox_center_xy(wheel)
        clearances[name] = _bbox_min_z(wheel) - _sample_height(x_axis, y_axis, z, x, y)
    return {
        "root_translation_z_m": root.location.z,
        "wheel_clearances_m": clearances,
        "placement_method": "least-squares local plane alignment then max wheel-center clearance",
        "terrain_plane": {"z_equals_ax_plus_by_plus_c": [float(slope_x), float(slope_y), float(intercept)], "normal": list(normal), "tilt_deg": math.degrees(normal.angle(Vector((0.0, 0.0, 1.0))))},
        "dynamic_pose_solver_enabled": False,
    }


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _render_overview(scene: bpy.types.Scene, output_path: Path) -> None:
    data = bpy.data.cameras.new("TerrainOverviewCamera_TEMP_data")
    camera = bpy.data.objects.new("TerrainOverviewCamera_TEMP", data)
    scene.collection.objects.link(camera)
    previous = scene.camera
    camera.location = (175.0, -175.0, 150.0)
    _look_at(camera, Vector((0, 0, 0)))
    data.lens = 52.0
    scene.camera = camera
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 800
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)
    scene.camera = previous
    bpy.data.objects.remove(camera, do_unlink=True)


def build(config_path: Path, geospatial_dir: Path, source_blend: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = json.loads((geospatial_dir / "metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("validation", {}).get("ok"):
        raise RuntimeError("Geospatial metadata report is not valid")
    if not source_blend.is_file():
        raise FileNotFoundError(source_blend)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_blend = Path(bpy.data.filepath).resolve() if bpy.data.filepath else None
    if current_blend != source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))

    heightfield = np.load(geospatial_dir / "terrain_heightfield.npz")
    x_axis = heightfield["x_local"]
    y_axis = heightfield["y_local"]
    z = heightfield["z_local"]
    visual = _mesh_from_heightfield(config["terrain"]["visual_object"], x_axis, y_axis, z, with_uv=True)
    texture_selection = metadata["texture_selection"]
    texture_path = Path(metadata["outputs"]["texture_png"])
    if not texture_path.is_file():
        raise FileNotFoundError(texture_path)
    visual.data.materials.append(_terrain_material(texture_path, float(config["terrain"]["roughness"]), texture_selection))
    visual["semantic_role"] = "visual_terrain"
    visual["mesh_step_m"] = float(config["terrain"]["visual_step_m"])
    visual["vertical_exaggeration"] = 1.0
    visual["projected_origin"] = list(map(float, heightfield["projected_origin"]))
    visual["projected_bounds"] = list(map(float, heightfield["projected_bounds"]))

    collision_step = int(round(float(config["terrain"]["collision_step_m"]) / float(config["terrain"]["visual_step_m"])))
    collision = _mesh_from_heightfield(config["terrain"]["collision_object"], x_axis[::collision_step], y_axis[::collision_step], z[::collision_step, ::collision_step], with_uv=False)
    collision.hide_render = True
    collision.display_type = "WIRE"
    collision["semantic_role"] = "collision_terrain"
    collision["mesh_step_m"] = float(config["terrain"]["collision_step_m"])
    collision["vertical_exaggeration"] = 1.0

    terrain_collection = bpy.data.collections.new("GaleTerrain")
    bpy.context.scene.collection.children.link(terrain_collection)
    for obj in (visual, collision):
        terrain_collection.objects.link(obj)
        bpy.context.scene.collection.objects.unlink(obj)

    placement = _place_rover(config, x_axis, y_axis, z)
    repo_root = Path(__file__).resolve().parents[2]
    camera_module = _load_module("setup_mahli_camera", repo_root / "scripts" / "blender" / "setup_mahli_camera.py")
    lighting_module = _load_module("setup_nominal_lighting", repo_root / "scripts" / "blender" / "setup_nominal_lighting.py")
    camera, target_point = camera_module.setup_mahli_camera(config)
    sun, world = lighting_module.setup_nominal_lighting(config)

    scene = bpy.context.scene
    scene.render.engine = config["render"]["engine"]
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = int(config["render"]["samples"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene["gale_product_id"] = config["product"]["id"]
    scene["geospatial_metadata"] = str(geospatial_dir / "metadata.json")
    scene["terrain_texture_product"] = texture_selection["chosen"]
    scene["terrain_color_type"] = texture_selection["color_type"]
    scene["microtexture_enabled"] = False
    scene["bulk_generation_enabled"] = False

    overview_path = output_dir / "terrain_preview.png"
    healthy_path = output_dir / "healthy_rover.png"
    _render_overview(scene, overview_path)
    scene.camera = camera
    scene.render.resolution_x = int(config["camera"]["resolution"][0])
    scene.render.resolution_y = int(config["camera"]["resolution"][1])
    scene.render.filepath = str(healthy_path)
    bpy.ops.render.render(write_still=True)

    blend_path = output_dir / "gale_terrain_scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report = {
        "ok": True,
        "blend": str(blend_path),
        "terrain_preview": str(overview_path),
        "healthy_rover_render": str(healthy_path),
        "visual_mesh": {"object": visual.name, "vertices": len(visual.data.vertices), "faces": len(visual.data.polygons), "dimensions_m": list(visual.dimensions), "uv_layers": [layer.name for layer in visual.data.uv_layers]},
        "collision_mesh": {"object": collision.name, "vertices": len(collision.data.vertices), "faces": len(collision.data.polygons), "dimensions_m": list(collision.dimensions), "hide_render": collision.hide_render},
        "placement": placement,
        "camera": {"object": camera.name, "target": config["camera"]["target_wheel"], "location": list(camera.location), "target_point": list(target_point), "focal_length_mm": camera.data.lens, "horizontal_fov_deg": math.degrees(camera.data.angle_x)},
        "lighting": {"sun": sun.name, "azimuth_deg": sun["azimuth_deg"], "elevation_deg": sun["elevation_deg"], "energy": sun.data.energy, "ambient_strength": world["ambient_strength"]},
        "geospatial_metadata": str(geospatial_dir / "metadata.json"),
        "texture_selection": texture_selection,
        "microtexture_enabled": False,
    }
    (output_dir / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build(*_arguments())
