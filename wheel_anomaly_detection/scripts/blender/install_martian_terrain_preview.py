from __future__ import annotations

import argparse
import bpy
import json
import sys
import tempfile
import zipfile
from pathlib import Path

from mathutils import Vector


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Install the approved visual-only Martian terrain.")
    parser.add_argument("--terrain-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(values)


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[index] for point in points) for index in range(3))),
        Vector(tuple(max(point[index] for point in points) for index in range(3))),
    )


def create_material(terrain_dir: Path) -> bpy.types.Material:
    material = bpy.data.materials.get("MartianTerrain_Visual")
    if material is None:
        material = bpy.data.materials.new("MartianTerrain_Visual")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    diffuse = nodes.new("ShaderNodeTexImage")
    normal_image = nodes.new("ShaderNodeTexImage")
    normal_map = nodes.new("ShaderNodeNormalMap")
    roughness = nodes.new("ShaderNodeTexImage")
    diffuse.name = "Diffuse"
    normal_image.name = "Normal"
    roughness.name = "Roughness"
    diffuse.image = bpy.data.images.load(str(terrain_dir / "textures" / "Diffuse.png"), check_existing=True)
    normal_image.image = bpy.data.images.load(str(terrain_dir / "textures" / "Normal.png"), check_existing=True)
    roughness.image = bpy.data.images.load(str(terrain_dir / "textures" / "roughness.png"), check_existing=True)
    normal_image.image.colorspace_settings.name = "Non-Color"
    roughness.image.colorspace_settings.name = "Non-Color"
    normal_map.inputs["Strength"].default_value = 0.65
    principled.inputs["Roughness"].default_value = 0.85
    links.new(diffuse.outputs["Color"], principled.inputs["Base Color"])
    links.new(normal_image.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    links.new(roughness.outputs["Color"], principled.inputs["Roughness"])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    material["displacement_enabled"] = False
    material["texture_source"] = str(terrain_dir)
    return material


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.matrix_world.translation).to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = parse_args()
    terrain_dir = args.terrain_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archive = terrain_dir / "source" / "Martian Terrain.zip"
    for path in (
        archive,
        terrain_dir / "textures" / "Diffuse.png",
        terrain_dir / "textures" / "Normal.png",
        terrain_dir / "textures" / "roughness.png",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    simulator = bpy.data.scenes.get("Simulator")
    root = bpy.data.objects.get("RoverRoot")
    camera = bpy.data.objects.get("SimulatorCamera")
    if simulator is None or root is None or camera is None:
        raise RuntimeError("Milestone-1 Simulator, RoverRoot, or SimulatorCamera is missing")

    for obj in list(bpy.data.objects):
        if obj.get("role") == "martian_terrain_visual":
            mesh = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
    collection = bpy.data.collections.get("MARTIAN_TERRAIN_VISUAL")
    if collection is None:
        collection = bpy.data.collections.new("MARTIAN_TERRAIN_VISUAL")
        simulator.collection.children.link(collection)

    with tempfile.TemporaryDirectory(prefix="martian_terrain_") as temporary:
        blend_path = Path(temporary) / "Martian-Terrain.blend"
        with zipfile.ZipFile(archive) as source:
            with source.open("Martian-Terrain.blend") as incoming, blend_path.open("wb") as outgoing:
                while block := incoming.read(1024 * 1024):
                    outgoing.write(block)
        with bpy.data.libraries.load(str(blend_path), link=False) as (data_from, data_to):
            if "Plane" not in data_from.objects:
                raise RuntimeError("Terrain source has no Plane object")
            data_to.objects = ["Plane"]
        terrain = data_to.objects[0]
    terrain.name = "MartianTerrain"
    collection.objects.link(terrain)
    terrain["role"] = "martian_terrain_visual"
    terrain["source_archive"] = str(archive)
    terrain["collision_enabled"] = False
    terrain["target_size_xy_meters"] = 50.0
    terrain["target_relief_meters"] = 0.9
    terrain.data.materials.clear()
    terrain.data.materials.append(create_material(terrain_dir))

    initial_min, initial_max = bounds(terrain)
    initial_size = initial_max - initial_min
    terrain.scale.x *= 50.0 / float(initial_size.x)
    terrain.scale.y *= 50.0 / float(initial_size.y)
    terrain.scale.z *= 0.9 / float(initial_size.z)
    bpy.context.view_layer.update()
    scaled_min, scaled_max = bounds(terrain)
    center = (scaled_min + scaled_max) * 0.5
    terrain.location.x -= center.x
    terrain.location.y -= center.y
    terrain.location.z -= scaled_min.z
    bpy.context.view_layer.update()
    final_min, final_max = bounds(terrain)

    simulator["terrain_status"] = "visual_preview_installed"
    simulator["terrain_collision_enabled"] = False
    simulator["terrain_size_xy_meters"] = 50.0
    simulator["terrain_relief_meters"] = 0.9
    simulator["simulator_milestone"] = "1-terrain-preview"
    root.matrix_world.identity()

    renders_dir = output_dir / "renders"
    reports_dir = output_dir / "reports"
    renders_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    saved_camera_matrix = camera.matrix_world.copy()
    saved_camera = simulator.camera
    saved_visibility = {
        obj.name: obj.hide_render
        for obj in simulator.objects
        if obj.type in {"MESH", "CURVE"} and obj != terrain
    }
    for name in saved_visibility:
        bpy.data.objects[name].hide_render = True
    simulator.render.engine = "CYCLES"
    simulator.cycles.device = "CPU"
    simulator.cycles.samples = 24
    simulator.cycles.use_denoising = True
    simulator.render.image_settings.file_format = "PNG"
    simulator.render.film_transparent = False
    camera.data.clip_end = max(float(camera.data.clip_end), 500.0)
    camera.matrix_world.translation = Vector((0.0, 0.0, 73.0))
    look_at(camera, Vector((0.0, 0.0, 0.35)))
    render(simulator, camera, renders_dir / "new_terrain_overview.png")
    camera.matrix_world.translation = Vector((38.0, -42.0, 27.0))
    look_at(camera, Vector((0.0, 0.0, 0.3)))
    render(simulator, camera, renders_dir / "new_terrain_oblique.png")
    for name, hidden in saved_visibility.items():
        bpy.data.objects[name].hide_render = hidden
    camera.matrix_world = saved_camera_matrix
    simulator.camera = saved_camera

    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    report = {
        "schema_version": "1.0",
        "status": "completed",
        "source": str(archive),
        "object": terrain.name,
        "source_bounds_size": list(initial_size),
        "final_bounds_min": list(final_min),
        "final_bounds_max": list(final_max),
        "final_bounds_size": list(final_max - final_min),
        "vertices": len(terrain.data.vertices),
        "polygons": len(terrain.data.polygons),
        "textures": ["Diffuse.png", "Normal.png", "roughness.png"],
        "displacement_enabled": False,
        "collision_enabled": False,
        "rover_transform_unchanged": True,
        "renders": ["renders/new_terrain_overview.png", "renders/new_terrain_oblique.png"],
    }
    (reports_dir / "terrain_preview.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Installed visual-only Martian terrain preview")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
