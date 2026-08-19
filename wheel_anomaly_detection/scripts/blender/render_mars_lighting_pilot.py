"""Render three controlled lighting/material/camera variants of one wheel frame."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return tuple(value.resolve() for value in (args.source_blend, args.pilot_config, args.pose_config, args.sampling_config, args.output_dir))


def _matrix(values) -> Matrix:
    values = list(map(float, values))
    return Matrix([values[index:index + 4] for index in range(0, 16, 4)])


def _matrix_values(matrix: Matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _configure_camera(settings: dict) -> bpy.types.Object:
    old = bpy.data.objects.get("MarsLightingPilotCamera")
    if old:
        bpy.data.objects.remove(old, do_unlink=True)
    data = bpy.data.cameras.new("MarsLightingPilotCamera_data")
    camera = bpy.data.objects.new("MarsLightingPilotCamera", data)
    bpy.context.scene.collection.objects.link(camera)
    data.type = "PERSP"
    data.lens = float(settings["focal_length_mm"])
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = float(settings["sensor_width_mm"])
    data.clip_start = float(settings["clip_start_m"])
    data.dof.use_dof = False
    return camera


def _configure_sun_world(preset: dict) -> None:
    if bool(preset["use_source_sun_world"]):
        return
    sun = bpy.data.objects.get("GaleNominalSun")
    if sun is None or sun.type != "LIGHT" or sun.data.type != "SUN":
        raise RuntimeError("GaleNominalSun is missing")
    settings = preset["sun"]
    azimuth = math.radians(float(settings["azimuth_deg"]))
    elevation = math.radians(float(settings["elevation_deg"]))
    direction_to_sun = Vector((
        math.cos(elevation) * math.sin(azimuth),
        math.cos(elevation) * math.cos(azimuth),
        math.sin(elevation),
    )).normalized()
    sun.rotation_euler = (-direction_to_sun).to_track_quat("-Z", "Y").to_euler()
    sun.data.energy = float(settings["energy"])
    sun.data.angle = math.radians(float(settings["angle_deg"]))
    sun.data.color = tuple(map(float, settings["color"]))
    background = bpy.context.scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = tuple(map(float, preset["world"]["color"]))
    background.inputs["Strength"].default_value = float(preset["world"]["strength"])


def _pilot_fill(camera_settings: dict, basis: dict, camera: bpy.types.Object) -> bpy.types.Object:
    from scripts.blender.microterrain.common import look_at

    settings = camera_settings["pilot_fill_light"]
    data = bpy.data.lights.new("MarsLightingPilotFill_data", type="AREA")
    data.energy = float(settings["energy_w"])
    data.shape = "DISK"
    data.size = float(settings["size_m"])
    data.color = tuple(map(float, settings["color"]))
    light = bpy.data.objects.new("MarsLightingPilotFill", data)
    bpy.context.scene.collection.objects.link(light)
    direction = (camera.location - basis["target"]).normalized()
    light.location = basis["target"] + direction * float(settings["subject_distance_m"]) + basis["up"] * float(settings["height_offset_m"])
    look_at(light, basis["target"])
    return light


def _rover_materials() -> list[bpy.types.Material]:
    root = bpy.data.objects.get("Rover")
    if root is None:
        raise RuntimeError("Rover root is missing")
    materials = set()
    for obj in bpy.data.objects:
        current = obj
        while current is not None and current != root:
            current = current.parent
        if current != root or obj.type != "MESH":
            continue
        materials.update(material for material in obj.data.materials if material and material.use_nodes)
    return sorted(materials, key=lambda material: material.name)


def _float_mix(nodes, links, target, mask, dust_value: float, prefix: str) -> None:
    incoming = next((link for link in links if link.to_socket == target), None)
    inverse = nodes.new("ShaderNodeMath")
    inverse.name = f"{prefix}_InverseMask"
    inverse.operation = "SUBTRACT"
    inverse.inputs[0].default_value = 1.0
    links.new(mask, inverse.inputs[1])
    original = nodes.new("ShaderNodeMath")
    original.name = f"{prefix}_OriginalContribution"
    original.operation = "MULTIPLY"
    if incoming:
        source = incoming.from_socket
        links.remove(incoming)
        links.new(source, original.inputs[0])
    else:
        original.inputs[0].default_value = float(target.default_value)
    links.new(inverse.outputs[0], original.inputs[1])
    dust = nodes.new("ShaderNodeMath")
    dust.name = f"{prefix}_DustContribution"
    dust.operation = "MULTIPLY"
    dust.inputs[0].default_value = float(dust_value)
    links.new(mask, dust.inputs[1])
    combined = nodes.new("ShaderNodeMath")
    combined.name = f"{prefix}_Combined"
    combined.operation = "ADD"
    links.new(original.outputs[0], combined.inputs[0])
    links.new(dust.outputs[0], combined.inputs[1])
    links.new(combined.outputs[0], target)


def _apply_dust_layer(settings: dict) -> list[str]:
    affected = []
    for material in _rover_materials():
        tree = material.node_tree
        nodes, links = tree.nodes, tree.links
        principled_nodes = [node for node in nodes if node.type == "BSDF_PRINCIPLED"]
        if not principled_nodes:
            continue
        geometry = nodes.new("ShaderNodeNewGeometry")
        geometry.name = "MarsDust_Geometry"
        separate = nodes.new("ShaderNodeSeparateXYZ")
        separate.name = "MarsDust_NormalZ"
        links.new(geometry.outputs["Normal"], separate.inputs[0])
        up = nodes.new("ShaderNodeMapRange")
        up.name = "MarsDust_UpwardDeposition"
        up.clamp = True
        up.inputs["From Min"].default_value = -0.15
        up.inputs["From Max"].default_value = 0.85
        up.inputs["To Min"].default_value = 0.08
        up.inputs["To Max"].default_value = 1.0
        links.new(separate.outputs["Z"], up.inputs["Value"])
        noise = nodes.new("ShaderNodeTexNoise")
        noise.name = "MarsDust_Patchiness"
        noise.noise_dimensions = "3D"
        noise.inputs["Scale"].default_value = float(settings["noise_scale"])
        noise.inputs["Detail"].default_value = float(settings["noise_detail"])
        noise.inputs["Roughness"].default_value = 0.68
        links.new(geometry.outputs["Position"], noise.inputs["Vector"])
        patch = nodes.new("ShaderNodeMapRange")
        patch.name = "MarsDust_PatchRange"
        patch.clamp = True
        patch.inputs["From Min"].default_value = 0.22
        patch.inputs["From Max"].default_value = 0.78
        patch.inputs["To Min"].default_value = 0.35
        patch.inputs["To Max"].default_value = 1.0
        links.new(noise.outputs["Fac"], patch.inputs["Value"])
        mask = nodes.new("ShaderNodeMath")
        mask.name = "MarsDust_DepositionMask"
        mask.operation = "MULTIPLY"
        links.new(up.outputs["Result"], mask.inputs[0])
        links.new(patch.outputs["Result"], mask.inputs[1])
        coverage = nodes.new("ShaderNodeMath")
        coverage.name = "MarsDust_Coverage"
        coverage.operation = "MULTIPLY"
        coverage.inputs[0].default_value = float(settings["coverage"])
        links.new(mask.outputs[0], coverage.inputs[1])

        for index, principled in enumerate(principled_nodes):
            base = principled.inputs["Base Color"]
            incoming = next((link for link in links if link.to_socket == base), None)
            mix = nodes.new("ShaderNodeMixRGB")
            mix.name = f"MarsDust_BaseColor_{index}"
            mix.blend_type = "MIX"
            links.new(coverage.outputs[0], mix.inputs[0])
            if incoming:
                source = incoming.from_socket
                links.remove(incoming)
                links.new(source, mix.inputs[1])
            else:
                mix.inputs[1].default_value = tuple(base.default_value)
            mix.inputs[2].default_value = tuple(map(float, settings["color"]))
            links.new(mix.outputs[0], base)
            _float_mix(nodes, links, principled.inputs["Roughness"], coverage.outputs[0], float(settings["roughness"]), f"MarsDust_Roughness_{index}")
            _float_mix(nodes, links, principled.inputs["Metallic"], coverage.outputs[0], 0.0, f"MarsDust_Metallic_{index}")
            normal = principled.inputs["Normal"]
            normal_link = next((link for link in links if link.to_socket == normal), None)
            bump = nodes.new("ShaderNodeBump")
            bump.name = f"MarsDust_MicroBump_{index}"
            bump.inputs["Strength"].default_value = float(settings["micro_bump_strength"])
            bump.inputs["Distance"].default_value = float(settings["micro_bump_distance_m"])
            links.new(noise.outputs["Fac"], bump.inputs["Height"])
            if normal_link:
                source = normal_link.from_socket
                links.remove(normal_link)
                links.new(source, bump.inputs["Normal"])
            links.new(bump.outputs["Normal"], normal)
        affected.append(material.name)
    return affected


def _configure_compositor(scene: bpy.types.Scene, preset: dict) -> None:
    if not bool(preset["camera_post"]):
        scene.compositing_node_group = None
        return
    tree = bpy.data.node_groups.new(f"MarsLightingPilot_{preset['id']}_Compositor", "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    hue = tree.nodes.new("CompositorNodeHueSat")
    hue.inputs["Saturation"].default_value = float(preset["post"]["saturation"])
    hue.inputs["Value"].default_value = float(preset["post"]["value"])
    veil = tree.nodes.new("CompositorNodeAlphaOver")
    veil.inputs["Factor"].default_value = float(preset["post"]["veil_factor"])
    veil.inputs["Foreground"].default_value = tuple(map(float, preset["post"]["veil_color"]))
    output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(render_layers.outputs["Image"], hue.inputs["Image"])
    tree.links.new(hue.outputs["Image"], veil.inputs["Background"])
    tree.links.new(veil.outputs["Image"], output.inputs["Image"])


def render_pilot(source_blend: Path, pilot_config_path: Path, pose_config_path: Path, sampling_config_path: Path, output_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.common import render
    from scripts.blender.render_wheel_camera_pose_pilot import _apply_terrain_color_grade, _projected_bounds
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose

    pilot_config = json.loads(pilot_config_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_config_path.read_text(encoding="utf-8"))
    sampling_config = json.loads(sampling_config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pose = next(entry for entry in pose_config["poses"] if entry["id"] == pilot_config["frame"]["camera_pose"])
    reference_camera_matrix = None
    reference_wheel_matrix = None
    results = []

    for preset in pilot_config["presets"]:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
        scene = bpy.context.scene
        if not bool(scene.get("wheel_pose_sampling_ready", False)):
            raise RuntimeError("Lighting pilot source is not pose-sampling ready")
        wheels = [bpy.data.objects[name] for name in sampling_config["wheels"]]
        base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
        target_wheel = bpy.data.objects[pilot_config["frame"]["target_wheel"]]
        apply_shared_roll(wheels, base_local, float(pilot_config["frame"]["roll_degrees"]))
        camera = _configure_camera(pose_config["camera"])
        basis = camera_pose(camera, target_wheel, pose, pose_config["camera"])
        _configure_sun_world(preset)
        fill = _pilot_fill(pose_config["camera"], basis, camera) if bool(preset["pilot_fill"]) else None
        graded = _apply_terrain_color_grade(pilot_config["terrain_grade"])
        dusty_materials = _apply_dust_layer(preset["dust"]) if bool(preset["dust_layer"]) else []
        scene.view_settings.look = preset["look"]
        scene.view_settings.exposure = float(preset["exposure_ev"])
        _configure_compositor(scene, preset)
        scene.camera = camera
        bpy.context.view_layer.update()
        camera_matrix = _matrix_values(camera.matrix_world)
        wheel_matrix = _matrix_values(target_wheel.matrix_world)
        if reference_camera_matrix is None:
            reference_camera_matrix = camera_matrix
            reference_wheel_matrix = wheel_matrix
        camera_delta = max(abs(a - b) for a, b in zip(camera_matrix, reference_camera_matrix))
        wheel_delta = max(abs(a - b) for a, b in zip(wheel_matrix, reference_wheel_matrix))
        if camera_delta > 1e-7 or wheel_delta > 1e-7:
            raise RuntimeError(f"Frame drift detected in preset {preset['id']}")
        path = output_dir / f"{preset['id']}.png"
        render(scene, camera, path, tuple(map(int, pilot_config["frame"]["resolution"])))
        sun = bpy.data.objects["GaleNominalSun"]
        background = scene.world.node_tree.nodes["Background"]
        results.append({
            "id": preset["id"],
            "label": preset["label"],
            "description": preset["description"],
            "image": str(path),
            "camera_matrix_max_delta": camera_delta,
            "wheel_matrix_max_delta": wheel_delta,
            "projection": _projected_bounds(scene, camera, target_wheel),
            "sun": {"energy": float(sun.data.energy), "angle_deg": math.degrees(float(sun.data.angle)), "color": list(map(float, sun.data.color))},
            "world": {"strength": float(background.inputs["Strength"].default_value), "color": list(map(float, background.inputs["Color"].default_value))},
            "look": scene.view_settings.look,
            "exposure_ev": float(scene.view_settings.exposure),
            "fill_light": fill.name if fill else None,
            "terrain_materials_graded": graded,
            "dusty_materials": dusty_materials,
            "camera_post": bool(preset["camera_post"]),
        })

    report = {
        "schema_version": 1,
        "source_blend": str(source_blend),
        "source_modified": False,
        "frame": pilot_config["frame"],
        "presets": results,
        "validation": {
            "ok": all(entry["camera_matrix_max_delta"] <= 1e-7 and entry["wheel_matrix_max_delta"] <= 1e-7 for entry in results),
            "same_camera_and_geometry": True,
        },
    }
    report_path = output_dir / "lighting_pilot.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    render_pilot(*_arguments())
