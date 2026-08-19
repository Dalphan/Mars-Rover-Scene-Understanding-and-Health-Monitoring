"""Render a deterministic three-level wheel surface-wear comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--wear-config", type=Path, required=True)
    parser.add_argument("--lighting-config", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return tuple(
        value.resolve()
        for value in (
            args.source_blend,
            args.wear_config,
            args.lighting_config,
            args.pose_config,
            args.sampling_config,
            args.output_dir,
        )
    )


def _matrix(values) -> Matrix:
    values = list(map(float, values))
    return Matrix([values[index : index + 4] for index in range(0, 16, 4)])


def _matrix_values(matrix: Matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _target_meshes(root: bpy.types.Object) -> list[bpy.types.Object]:
    meshes = []
    for obj in bpy.data.objects:
        current = obj
        while current is not None and current != root:
            current = current.parent
        if current == root and obj.type == "MESH":
            meshes.append(obj)
    return sorted(meshes, key=lambda obj: obj.name)


def _geometry_signature(objects: list[bpy.types.Object]) -> dict:
    return {
        "object_count": len(objects),
        "vertex_count": sum(len(obj.data.vertices) for obj in objects),
        "polygon_count": sum(len(obj.data.polygons) for obj in objects),
        "objects": [
            {
                "name": obj.name,
                "vertices": len(obj.data.vertices),
                "polygons": len(obj.data.polygons),
                "matrix_world": _matrix_values(obj.matrix_world),
            }
            for obj in objects
        ],
    }


def _wheel_dimensions(root: bpy.types.Object, objects: list[bpy.types.Object]) -> dict:
    root_inverse = root.matrix_world.inverted()
    points = [root_inverse @ obj.matrix_world @ vertex.co for obj in objects for vertex in obj.data.vertices]
    if not points:
        raise RuntimeError(f"Wheel {root.name} has no vertices")
    return {
        "x_min": min(float(point.x) for point in points),
        "x_max": max(float(point.x) for point in points),
        "radial_max": max(float((point.y * point.y + point.z * point.z) ** 0.5) for point in points),
        "outer_side_sign": 1.0 if root.name.endswith("_left") else -1.0,
    }


def _clone_target_materials(objects: list[bpy.types.Object], suffix: str) -> tuple[list[bpy.types.Material], list[str]]:
    clones_by_source = {}
    affected_objects = []
    for obj in objects:
        obj.data = obj.data.copy()
        affected_objects.append(obj.name)
        for index, material in enumerate(list(obj.data.materials)):
            if material is None:
                continue
            key = material.as_pointer()
            clone = clones_by_source.get(key)
            if clone is None:
                clone = material.copy()
                clone.name = f"{material.name}__{suffix}"
                clones_by_source[key] = clone
            obj.data.materials[index] = clone
    return sorted(clones_by_source.values(), key=lambda material: material.name), affected_objects


def _mix_float(nodes, links, target, mask, value: float, prefix: str) -> None:
    incoming = next((link for link in links if link.to_socket == target), None)
    inverse = nodes.new("ShaderNodeMath")
    inverse.name = f"{prefix}_Inverse"
    inverse.operation = "SUBTRACT"
    inverse.inputs[0].default_value = 1.0
    links.new(mask, inverse.inputs[1])
    original = nodes.new("ShaderNodeMath")
    original.name = f"{prefix}_Original"
    original.operation = "MULTIPLY"
    if incoming:
        source = incoming.from_socket
        links.remove(incoming)
        links.new(source, original.inputs[0])
    else:
        original.inputs[0].default_value = float(target.default_value)
    links.new(inverse.outputs[0], original.inputs[1])
    wear = nodes.new("ShaderNodeMath")
    wear.name = f"{prefix}_Wear"
    wear.operation = "MULTIPLY"
    wear.inputs[0].default_value = float(value)
    links.new(mask, wear.inputs[1])
    combined = nodes.new("ShaderNodeMath")
    combined.name = f"{prefix}_Combined"
    combined.operation = "ADD"
    links.new(original.outputs[0], combined.inputs[0])
    links.new(wear.outputs[0], combined.inputs[1])
    links.new(combined.outputs[0], target)


def _map_range(nodes, links, source, minimum: float, maximum: float, name: str):
    node = nodes.new("ShaderNodeMapRange")
    node.name = name
    node.clamp = True
    node.inputs["From Min"].default_value = float(minimum)
    node.inputs["From Max"].default_value = float(maximum)
    node.inputs["To Min"].default_value = 0.0
    node.inputs["To Max"].default_value = 1.0
    links.new(source, node.inputs["Value"])
    return node.outputs["Result"]


def _multiply(nodes, links, left, right, name: str):
    node = nodes.new("ShaderNodeMath")
    node.name = name
    node.operation = "MULTIPLY"
    links.new(left, node.inputs[0])
    links.new(right, node.inputs[1])
    return node.outputs[0]


def _surface_exposure_mask(nodes, links, texcoord, dimensions: dict, settings: dict):
    position = texcoord.outputs["Object"]
    separate_position = nodes.new("ShaderNodeSeparateXYZ")
    separate_position.name = "WheelWear_LocalPosition"
    links.new(position, separate_position.inputs[0])
    y_squared = nodes.new("ShaderNodeMath")
    y_squared.name = "WheelWear_YSquared"
    y_squared.operation = "MULTIPLY"
    links.new(separate_position.outputs["Y"], y_squared.inputs[0])
    links.new(separate_position.outputs["Y"], y_squared.inputs[1])
    z_squared = nodes.new("ShaderNodeMath")
    z_squared.name = "WheelWear_ZSquared"
    z_squared.operation = "MULTIPLY"
    links.new(separate_position.outputs["Z"], z_squared.inputs[0])
    links.new(separate_position.outputs["Z"], z_squared.inputs[1])
    radius_squared = nodes.new("ShaderNodeMath")
    radius_squared.name = "WheelWear_RadiusSquared"
    radius_squared.operation = "ADD"
    links.new(y_squared.outputs[0], radius_squared.inputs[0])
    links.new(z_squared.outputs[0], radius_squared.inputs[1])
    radius = nodes.new("ShaderNodeMath")
    radius.name = "WheelWear_Radius"
    radius.operation = "SQRT"
    links.new(radius_squared.outputs[0], radius.inputs[0])
    radial_mask = _map_range(
        nodes,
        links,
        radius.outputs[0],
        float(dimensions["radial_max"]) * float(settings["radial_start_fraction"]),
        float(dimensions["radial_max"]) * float(settings["radial_full_fraction"]),
        "WheelWear_OuterRadius",
    )

    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.name = "WheelWear_Geometry"
    local_normal = nodes.new("ShaderNodeVectorTransform")
    local_normal.name = "WheelWear_LocalNormal"
    local_normal.vector_type = "NORMAL"
    local_normal.convert_from = "WORLD"
    local_normal.convert_to = "OBJECT"
    links.new(geometry.outputs["Normal"], local_normal.inputs["Vector"])
    radial_vector = nodes.new("ShaderNodeCombineXYZ")
    radial_vector.name = "WheelWear_RadialVector"
    links.new(separate_position.outputs["Y"], radial_vector.inputs["Y"])
    links.new(separate_position.outputs["Z"], radial_vector.inputs["Z"])
    radial_normal = nodes.new("ShaderNodeVectorMath")
    radial_normal.name = "WheelWear_RadialNormal"
    radial_normal.operation = "NORMALIZE"
    links.new(radial_vector.outputs["Vector"], radial_normal.inputs[0])
    outward_dot = nodes.new("ShaderNodeVectorMath")
    outward_dot.name = "WheelWear_OutwardNormalDot"
    outward_dot.operation = "DOT_PRODUCT"
    links.new(local_normal.outputs["Vector"], outward_dot.inputs[0])
    links.new(radial_normal.outputs["Vector"], outward_dot.inputs[1])
    outward_mask = _map_range(
        nodes,
        links,
        outward_dot.outputs["Value"],
        float(settings["tread_normal_start"]),
        float(settings["tread_normal_full"]),
        "WheelWear_TreadNormal",
    )
    tread = _multiply(nodes, links, radial_mask, outward_mask, "WheelWear_TreadExposure")

    outer_sign = float(dimensions["outer_side_sign"])
    signed_position = nodes.new("ShaderNodeMath")
    signed_position.name = "WheelWear_SignedAxialPosition"
    signed_position.operation = "MULTIPLY"
    signed_position.inputs[0].default_value = outer_sign
    links.new(separate_position.outputs["X"], signed_position.inputs[1])
    outer_extent = float(dimensions["x_max"] if outer_sign > 0.0 else -dimensions["x_min"])
    shoulder_position = _map_range(
        nodes,
        links,
        signed_position.outputs[0],
        outer_extent * float(settings["shoulder_position_start_fraction"]),
        outer_extent * float(settings["shoulder_position_full_fraction"]),
        "WheelWear_OutboardShoulderPosition",
    )
    separate_normal = nodes.new("ShaderNodeSeparateXYZ")
    separate_normal.name = "WheelWear_NormalComponents"
    links.new(local_normal.outputs["Vector"], separate_normal.inputs[0])
    signed_normal = nodes.new("ShaderNodeMath")
    signed_normal.name = "WheelWear_SignedAxialNormal"
    signed_normal.operation = "MULTIPLY"
    signed_normal.inputs[0].default_value = outer_sign
    links.new(separate_normal.outputs["X"], signed_normal.inputs[1])
    shoulder_normal = _map_range(
        nodes,
        links,
        signed_normal.outputs[0],
        float(settings["shoulder_normal_start"]),
        float(settings["shoulder_normal_full"]),
        "WheelWear_OutboardShoulderNormal",
    )
    shoulder = _multiply(nodes, links, radial_mask, shoulder_position, "WheelWear_ShoulderRadiusPosition")
    shoulder = _multiply(nodes, links, shoulder, shoulder_normal, "WheelWear_ShoulderExposure")
    combined = nodes.new("ShaderNodeMath")
    combined.name = "WheelWear_TreadOrShoulder"
    combined.operation = "MAXIMUM"
    links.new(tread, combined.inputs[0])
    links.new(shoulder, combined.inputs[1])
    return combined.outputs[0]


def _apply_wear(
    materials: list[bpy.types.Material],
    target_wheel: bpy.types.Object,
    dimensions: dict,
    mask_path: Path,
    settings: dict,
) -> list[str]:
    image = bpy.data.images.load(str(mask_path), check_existing=False)
    image.name = f"WheelWearMask_{mask_path.stem}"
    image.colorspace_settings.name = "Non-Color"
    affected = []
    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        nodes, links = material.node_tree.nodes, material.node_tree.links
        principled_nodes = [node for node in nodes if node.type == "BSDF_PRINCIPLED"]
        if not principled_nodes:
            continue
        texcoord = nodes.new("ShaderNodeTexCoord")
        texcoord.name = "WheelWear_WheelLocalCoordinates"
        texcoord.object = target_wheel
        mapping = nodes.new("ShaderNodeMapping")
        mapping.name = "WheelWear_PhysicalProjection"
        mapping.inputs["Location"].default_value = tuple(map(float, settings["projection"]["location"]))
        mapping.inputs["Scale"].default_value = tuple(map(float, settings["projection"]["scale"]))
        texture = nodes.new("ShaderNodeTexImage")
        texture.name = "WheelWear_Mask"
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "CLIP"
        texture.projection = "BOX"
        texture.projection_blend = float(settings["projection"]["blend"])
        links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        exposure = _surface_exposure_mask(nodes, links, texcoord, dimensions, settings["surface_exposure"])
        exposed_texture = _multiply(nodes, links, texture.outputs["Color"], exposure, "WheelWear_ExposedTexture")
        opacity = nodes.new("ShaderNodeMath")
        opacity.name = "WheelWear_Opacity"
        opacity.operation = "MULTIPLY"
        opacity.inputs[0].default_value = float(settings["opacity"])
        links.new(exposed_texture, opacity.inputs[1])
        mask = opacity.outputs[0]
        for index, principled in enumerate(principled_nodes):
            prefix = f"WheelWear_{index}"
            base = principled.inputs["Base Color"]
            incoming = next((link for link in links if link.to_socket == base), None)
            mix = nodes.new("ShaderNodeMixRGB")
            mix.name = f"{prefix}_BaseColor"
            mix.blend_type = "MIX"
            links.new(mask, mix.inputs[0])
            if incoming:
                source = incoming.from_socket
                links.remove(incoming)
                links.new(source, mix.inputs[1])
            else:
                mix.inputs[1].default_value = tuple(base.default_value)
            mix.inputs[2].default_value = tuple(map(float, settings["scratch_color"]))
            links.new(mix.outputs[0], base)
            _mix_float(nodes, links, principled.inputs["Roughness"], mask, float(settings["roughness"]), f"{prefix}_Roughness")
            _mix_float(nodes, links, principled.inputs["Metallic"], mask, float(settings["metallic"]), f"{prefix}_Metallic")
            normal = principled.inputs["Normal"]
            normal_link = next((link for link in links if link.to_socket == normal), None)
            bump = nodes.new("ShaderNodeBump")
            bump.name = f"{prefix}_Bump"
            bump.inputs["Strength"].default_value = float(settings["bump_strength"])
            bump.inputs["Distance"].default_value = float(settings["bump_distance_m"])
            bump.invert = bool(settings.get("invert_bump", True))
            links.new(mask, bump.inputs["Height"])
            if normal_link:
                source = normal_link.from_socket
                links.remove(normal_link)
                links.new(source, bump.inputs["Normal"])
            links.new(bump.outputs["Normal"], normal)
        affected.append(material.name)
    return affected


def render_pilot(
    source_blend: Path,
    wear_config_path: Path,
    lighting_config_path: Path,
    pose_config_path: Path,
    sampling_config_path: Path,
    output_dir: Path,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender.microterrain.common import render
    from scripts.blender.render_wheel_camera_pose_pilot import _apply_terrain_color_grade, _projected_bounds
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose

    wear_config = json.loads(wear_config_path.read_text(encoding="utf-8"))
    lighting_config = json.loads(lighting_config_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_config_path.read_text(encoding="utf-8"))
    sampling_config = json.loads(sampling_config_path.read_text(encoding="utf-8"))
    role = wear_config["lighting_role"]
    preset_id = lighting_config["dataset_roles"][role]
    if isinstance(preset_id, list):
        raise RuntimeError(f"Lighting role {role} is not a single preset")
    preset = next(entry for entry in lighting_config["presets"] if entry["id"] == preset_id)
    frame = wear_config["frame"]
    pose = next(entry for entry in pose_config["poses"] if entry["id"] == frame["camera_pose"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_camera_matrix = None
    reference_wheel_matrix = None
    reference_geometry = None
    results = []

    for variant in wear_config["variants"]:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
        scene = bpy.context.scene
        if not bool(scene.get("wheel_pose_sampling_ready", False)):
            raise RuntimeError("Surface-wear pilot source is not pose-sampling ready")
        wheels = [bpy.data.objects[name] for name in sampling_config["wheels"]]
        base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
        target_wheel = bpy.data.objects[frame["target_wheel"]]
        apply_shared_roll(wheels, base_local, float(frame["roll_degrees"]))
        meshes = _target_meshes(target_wheel)
        if not meshes:
            raise RuntimeError(f"Target wheel {target_wheel.name} has no mesh objects")
        missing_uv = [obj.name for obj in meshes if not obj.data.uv_layers]
        if bool(variant["wear_enabled"]) and missing_uv:
            raise RuntimeError(f"Wear requires UVs; missing on: {missing_uv}")
        geometry = _geometry_signature(meshes)
        dimensions = _wheel_dimensions(target_wheel, meshes)
        if reference_geometry is None:
            reference_geometry = geometry
        elif geometry != reference_geometry:
            raise RuntimeError(f"Wheel geometry drift detected before variant {variant['id']}")
        cloned_materials = []
        affected_objects = []
        if bool(variant["wear_enabled"]):
            cloned_materials, affected_objects = _clone_target_materials(meshes, variant["id"])
        camera = lighting._configure_camera(pose_config["camera"])
        camera_pose(camera, target_wheel, pose, pose_config["camera"])
        lighting._configure_sun_world(preset)
        graded = _apply_terrain_color_grade(lighting_config["terrain_grade"])
        dusty_materials = lighting._apply_dust_layer(preset["dust"]) if bool(preset["dust_layer"]) else []
        wear_materials = []
        if bool(variant["wear_enabled"]):
            mask_path = output_dir / variant["mask_filename"]
            if not mask_path.is_file():
                raise RuntimeError(f"Wear mask is missing: {mask_path}")
            wear_materials = _apply_wear(cloned_materials, target_wheel, dimensions, mask_path, wear_config["shader"])
        scene.view_settings.look = preset["look"]
        scene.view_settings.exposure = float(preset["exposure_ev"])
        lighting._configure_compositor(scene, preset)
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
            raise RuntimeError(f"Frame drift detected in variant {variant['id']}")
        path = output_dir / f"{variant['id']}.png"
        render(scene, camera, path, tuple(map(int, frame["resolution"])))
        results.append(
            {
                "id": variant["id"],
                "label": variant["label"],
                "image": str(path),
                "wear_enabled": bool(variant["wear_enabled"]),
                "mask": str(output_dir / variant["mask_filename"]) if variant["mask_filename"] else None,
                "target_mask_coverage": float(variant["target_mask_coverage"]),
                "camera_matrix_max_delta": camera_delta,
                "wheel_matrix_max_delta": wheel_delta,
                "projection": _projected_bounds(scene, camera, target_wheel),
                "target_meshes": [obj.name for obj in meshes],
                "missing_uv_meshes": missing_uv,
                "affected_objects": affected_objects,
                "wear_materials": wear_materials,
                "dusty_materials": dusty_materials,
                "terrain_materials_graded": graded,
                "geometry": geometry,
                "wheel_dimensions": dimensions,
            }
        )

    validation_ok = all(
        entry["camera_matrix_max_delta"] <= 1e-7
        and entry["wheel_matrix_max_delta"] <= 1e-7
        and not entry["missing_uv_meshes"]
        for entry in results
    )
    report = {
        "schema_version": 1,
        "source_blend": str(source_blend),
        "source_modified": False,
        "lighting_preset": preset_id,
        "frame": frame,
        "shader": wear_config["shader"],
        "dataset_sampling": wear_config["dataset_sampling"],
        "variants": results,
        "validation": {
            "ok": validation_ok,
            "same_camera_pose_and_geometry": True,
            "materials_cloned_only_for_target_wheel": True,
            "source_geometry_modified": False,
        },
    }
    report_path = output_dir / "surface_wear_pilot.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    render_pilot(*_arguments())
