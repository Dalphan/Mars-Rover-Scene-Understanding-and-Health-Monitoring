"""Blender-side non-destructive hybrid wheel-hole injector.

The source wheel mesh is an open, fragmented shell.  This module therefore
uses a shader cutout plus a reusable physical carrier (rim, open walls and
optional flap) instead of applying Boolean modifiers.
"""

from __future__ import annotations

import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ANOMALY_AOV_NAME = "AnomalyMask"
FRAME_NAME = "WheelHole_AnomalyFrame"
CARRIER_NAME = "WheelHole_AnomalyCarrier"
IMAGE_NAME = "WheelHole_ProfileMask"
CUTOUT_PREFIX = "WheelHole_"
RIM_MATERIAL_NAME = "WheelHole_RimMaterial"
CAVITY_MATERIAL_NAME = "WheelHole_CavityMaterial"


def _matrix_from_frame(center: Vector, long_axis: Vector, short_axis: Vector, normal: Vector) -> Matrix:
    matrix = Matrix((long_axis.normalized(), short_axis.normalized(), normal.normalized())).transposed().to_4x4()
    matrix.translation = center
    return matrix


def positive_roll_effect_sign(wheel: bpy.types.Object, camera_settings: dict) -> float:
    from scripts.blender.wheel_pose_sampling import stable_wheel_basis

    _outward, forward, up, axle = stable_wheel_basis(wheel, camera_settings)
    effect = (axle.cross(up)).dot(forward)
    if abs(effect) < 0.95:
        raise RuntimeError(f"Cannot establish positive roll direction for {wheel.name}")
    return 1.0 if effect > 0.0 else -1.0


def sector_target_offset_degrees(descriptor: dict, anomaly_config: dict, wheel_radius: float) -> dict:
    points = descriptor["points_long_short_m"]
    orientation = descriptor["orientation"]
    tangential_index = 1 if orientation in {"axial", "radial"} else 0
    tangent_half_extent = max(abs(float(point[tangential_index])) for point in points)
    half_extent_degrees = math.degrees(tangent_half_extent / max(float(wheel_radius), 1e-9))
    settings = anomaly_config["placement"]
    limit = max(
        2.0,
        float(settings["maximum_upper_angle_degrees"])
        - half_extent_degrees
        - float(settings["sector_safety_degrees"]),
    )
    sign = {"leading": -1.0, "upper": 0.0, "trailing": 1.0}[descriptor["image_sector"]]
    base = sign * float(settings["sector_scale"]) * limit
    jitter_limit = min(float(settings["sector_jitter_max_degrees"]), 0.15 * limit)
    jitter = float(descriptor["sector_jitter_unit"]) * jitter_limit
    return {
        "target_offset_degrees": float(base + jitter),
        "base_offset_degrees": float(base),
        "jitter_degrees": float(jitter),
        "permitted_center_limit_degrees": float(limit),
        "contour_half_extent_degrees": float(half_extent_degrees),
    }


def _closest_surface(wheel: bpy.types.Object, point_local: Vector, maximum_distance: float = 0.008):
    hit, location, normal, polygon_index = wheel.closest_point_on_mesh(point_local, distance=float(maximum_distance))
    if not hit:
        return None
    return location.copy(), normal.normalized(), int(polygon_index), float((location - point_local).length)


def _tread_frame(wheel: bpy.types.Object, candidate: dict, descriptor: dict, config: dict):
    radius_outer = float(wheel.get("pose_sampling_radius_m", max(math.hypot(v.co.y, v.co.z) for v in wheel.data.vertices)))
    settings = config["placement"]
    radius = radius_outer * float(settings["tread_base_radius_fraction"])
    tolerance = radius_outer * float(settings["tread_radius_tolerance_fraction"])
    half_width = max(abs(float(corner[0])) for corner in wheel.bound_box)
    allowed_half_width = half_width * float(settings["tread_half_width_fraction"])
    points = descriptor["points_long_short_m"]
    axial_half_extent = max(abs(float(point[0 if descriptor["orientation"] == "axial" else 1])) for point in points)
    safe_center_half_width = allowed_half_width - axial_half_extent - float(descriptor["rim_width_m"])
    if safe_center_half_width <= 0.0:
        return None
    theta = 2.0 * math.pi * (
        float(candidate["panel_index"]) + 0.5 + float(candidate["panel_jitter_fraction"])
    ) / float(settings["expected_grouser_count"])
    target_x = float(candidate["axial_fraction"]) * safe_center_half_width
    radial_band = list(map(float, settings["tread_skin_radius_fraction"]))
    period = 2.0 * math.pi / float(settings["expected_grouser_count"])
    ray_hits = []
    # Query actual outer geometry instead of relying on imported face indices.
    # The first hit from outside is either a grouser or the intact skin; the
    # configured radial band retains only the latter.
    for step in range(-12, 13):
        angle = theta + period * 0.42 * float(step) / 12.0
        radial = Vector((0.0, math.cos(angle), math.sin(angle)))
        origin_radius = radius_outer * 1.20
        origin = Vector((target_x, radial.y * origin_radius, radial.z * origin_radius))
        hit, location, normal, _polygon = wheel.ray_cast(origin, -radial, distance=radius_outer * 0.45)
        if not hit:
            continue
        hit_radius = math.hypot(location.y, location.z)
        if not radial_band[0] * radius_outer <= hit_radius <= radial_band[1] * radius_outer:
            continue
        alignment = normal.dot(radial)
        if alignment < 0.0:
            normal = -normal
            alignment = -alignment
        if alignment < 0.35:
            continue
        ray_hits.append((location.copy(), normal.normalized(), abs(angle - theta), hit_radius))
    if not ray_hits:
        return None
    center, normal, _angular_delta, center_radius = min(
        ray_hits,
        key=lambda entry: (entry[2] * radius_outer + 0.35 * abs(entry[3] - radius), entry[2]),
    )
    distance = 0.0
    center_radius = math.hypot(center.y, center.z)
    radial = Vector((0.0, center.y, center.z)).normalized()
    alignment = normal.dot(radial)
    if alignment < 0.0:
        normal = -normal
        alignment = -alignment
    if abs(center_radius - radius) > tolerance or alignment < 0.35 or distance > 0.010:
        return None
    axial = Vector((1.0, 0.0, 0.0))
    tangent = Vector((0.0, -radial.z, radial.y)).normalized()
    long_axis, short_axis = (axial, tangent) if descriptor["orientation"] == "axial" else (tangent, axial)
    return center, long_axis, short_axis, radial, radius_outer


def _shoulder_frame(wheel: bpy.types.Object, candidate: dict, descriptor: dict, config: dict):
    radius_outer = float(wheel.get("pose_sampling_radius_m", max(math.hypot(v.co.y, v.co.z) for v in wheel.data.vertices)))
    half_width = max(abs(float(corner[0])) for corner in wheel.bound_box)
    outboard_sign = 1.0 if wheel.name.endswith("_left") else -1.0
    theta = math.radians(float(candidate["source_angle_degrees"]))
    settings = config["placement"]
    radius = radius_outer * float(settings["shoulder_base_radius_fraction"])
    axial_half_extent = max(abs(float(point[1])) for point in descriptor["points_long_short_m"])
    safe_center = half_width * float(settings["tread_half_width_fraction"]) - axial_half_extent - float(descriptor["rim_width_m"])
    if safe_center <= 0.0:
        return None
    center_x = outboard_sign * safe_center * float(candidate["outboard_center_fraction"])
    center_guess = Vector((center_x, radius * math.cos(theta), radius * math.sin(theta)))
    hit = _closest_surface(wheel, center_guess, maximum_distance=0.012)
    if hit is None:
        return None
    center, normal, _polygon, distance = hit
    radial_length = math.hypot(center.y, center.z)
    radial = Vector((0.0, center.y, center.z)).normalized()
    alignment = normal.dot(radial)
    if alignment < 0.0:
        normal = -normal
        alignment = -alignment
    if alignment < 0.35 or distance > 0.010:
        return None
    tolerance = radius_outer * float(settings["tread_radius_tolerance_fraction"])
    if abs(radial_length - radius) > tolerance or center.x * outboard_sign <= 0.0:
        return None
    tangent = Vector((0.0, -radial.z, radial.y)).normalized()
    axial = Vector((1.0, 0.0, 0.0))
    return center, tangent, axial, radial, radius_outer


def _map_profile_point(
    wheel: bpy.types.Object,
    frame: tuple[Vector, Vector, Vector, Vector, float],
    point: list[float],
    surface: str,
) -> tuple[Vector, Vector] | None:
    center, long_axis, short_axis, normal, _radius_outer = frame
    guess = center + long_axis * float(point[0]) + short_axis * float(point[1])
    hit = _closest_surface(wheel, guess, maximum_distance=0.009 if surface == "tread" else 0.012)
    if hit is None:
        return None
    location, hit_normal, _polygon, distance = hit
    alignment = hit_normal.dot(normal)
    if alignment < 0.0:
        hit_normal = -hit_normal
        alignment = -alignment
    if alignment < 0.25 or distance > (0.007 if surface == "tread" else 0.010):
        return None
    if surface == "tread":
        expected_radius = math.hypot(center.y, center.z)
        if abs(math.hypot(location.y, location.z) - expected_radius) > 0.010:
            return None
    return location, hit_normal


def resolve_placements(wheel: bpy.types.Object, descriptor: dict, anomaly_config: dict) -> list[dict]:
    """Resolve every source-geometry-safe candidate in deterministic order."""

    rejections = []
    placements = []
    for candidate in descriptor["placement_candidates"]:
        frame = (
            _tread_frame(wheel, candidate, descriptor, anomaly_config)
            if descriptor["surface"] == "tread"
            else _shoulder_frame(wheel, candidate, descriptor, anomaly_config)
        )
        if frame is None:
            rejections.append({"attempt": int(candidate["attempt"]), "reason": "center_surface_gate"})
            continue
        mapped = [_map_profile_point(wheel, frame, point, descriptor["surface"]) for point in descriptor["points_long_short_m"]]
        if any(point is None for point in mapped):
            rejections.append({"attempt": int(candidate["attempt"]), "reason": "contour_surface_gate"})
            continue
        center, long_axis, short_axis, normal, radius_outer = frame
        boundary = [point[0] for point in mapped]
        normals = [point[1] for point in mapped]
        center_hit = _closest_surface(wheel, center)
        if center_hit is None:
            rejections.append({"attempt": int(candidate["attempt"]), "reason": "center_probe_missing"})
            continue
        probes = [(center_hit[0], center_hit[1])] + list(zip(boundary, normals, strict=True))
        placements.append({
            "ok": True,
            "candidate": candidate,
            "candidate_attempt": int(candidate["attempt"]),
            "center_local_m": list(map(float, center)),
            "long_axis_local": list(map(float, long_axis)),
            "short_axis_local": list(map(float, short_axis)),
            "normal_local": list(map(float, normal)),
            "boundary_local_m": [list(map(float, point)) for point in boundary],
            "boundary_normals_local": [list(map(float, value)) for value in normals],
            "probes_local": [[list(map(float, point)), list(map(float, probe_normal))] for point, probe_normal in probes],
            "wheel_radius_m": float(radius_outer),
            "rejections": list(rejections),
        })
    return placements


def resolve_placement(wheel: bpy.types.Object, descriptor: dict, anomaly_config: dict) -> dict:
    """Compatibility helper returning the first valid candidate, fail closed."""

    placements = resolve_placements(wheel, descriptor, anomaly_config)
    if placements:
        return placements[0]
    return {
        "ok": False,
        "rejections": [
            {"attempt": int(candidate["attempt"]), "reason": "source_geometry_gate"}
            for candidate in descriptor["placement_candidates"]
        ],
        "fail_closed": True,
    }


def _new_principled_material(name: str, base_color, roughness: float, metallic: float):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    output = nodes.new("ShaderNodeOutputMaterial")
    principled.inputs["Base Color"].default_value = tuple(map(float, base_color))
    principled.inputs["Roughness"].default_value = float(roughness)
    principled.inputs["Metallic"].default_value = float(metallic)
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    aov = nodes.new("ShaderNodeOutputAOV")
    aov.name = f"{CUTOUT_PREFIX}AnomalyAOV"
    aov.aov_name = ANOMALY_AOV_NAME
    aov.inputs["Value"].default_value = 1.0
    return material


def _setup_carrier():
    frame = bpy.data.objects.get(FRAME_NAME)
    if frame is None:
        frame = bpy.data.objects.new(FRAME_NAME, None)
        bpy.context.scene.collection.objects.link(frame)
    mesh = bpy.data.meshes.get(f"{CARRIER_NAME}_Mesh")
    if mesh is None:
        mesh = bpy.data.meshes.new(f"{CARRIER_NAME}_Mesh")
    carrier = bpy.data.objects.get(CARRIER_NAME)
    if carrier is None:
        carrier = bpy.data.objects.new(CARRIER_NAME, mesh)
        bpy.context.scene.collection.objects.link(carrier)
    elif carrier.data != mesh:
        carrier.data = mesh
    rim = _new_principled_material(RIM_MATERIAL_NAME, (0.28, 0.24, 0.20, 1.0), 0.62, 0.72)
    cavity = _new_principled_material(CAVITY_MATERIAL_NAME, (0.035, 0.020, 0.012, 1.0), 0.90, 0.18)
    carrier.data.materials.clear()
    carrier.data.materials.append(rim)
    carrier.data.materials.append(cavity)
    carrier.hide_render = True
    return frame, carrier, rim, cavity


def _install_cutout(material, frame, image) -> bool:
    if not material.use_nodes or material.node_tree is None:
        return False
    nodes, links = material.node_tree.nodes, material.node_tree.links
    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
    if output is None:
        return False
    surface = output.inputs["Surface"]
    incoming = next((link for link in links if link.to_socket == surface), None)
    if incoming is None:
        return False
    original = incoming.from_socket
    links.remove(incoming)
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.name = f"{CUTOUT_PREFIX}Coordinates"
    texcoord.object = frame
    scale = nodes.new("ShaderNodeVectorMath")
    scale.name = f"{CUTOUT_PREFIX}Scale"
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = (1.0, 1.0, 1.0)
    offset = nodes.new("ShaderNodeVectorMath")
    offset.name = f"{CUTOUT_PREFIX}Offset"
    offset.operation = "ADD"
    offset.inputs[1].default_value = (0.5, 0.5, 0.0)
    texture = nodes.new("ShaderNodeTexImage")
    texture.name = f"{CUTOUT_PREFIX}Profile"
    texture.image = image
    texture.interpolation = "Closest"
    texture.extension = "CLIP"
    binary = nodes.new("ShaderNodeMath")
    binary.name = f"{CUTOUT_PREFIX}Binary"
    binary.operation = "GREATER_THAN"
    binary.inputs[1].default_value = 0.5
    separate = nodes.new("ShaderNodeSeparateXYZ")
    separate.name = f"{CUTOUT_PREFIX}Separate"
    absolute = nodes.new("ShaderNodeMath")
    absolute.name = f"{CUTOUT_PREFIX}NormalAbsolute"
    absolute.operation = "ABSOLUTE"
    slab = nodes.new("ShaderNodeMath")
    slab.name = f"{CUTOUT_PREFIX}Slab"
    slab.operation = "LESS_THAN"
    slab.inputs[1].default_value = 0.003
    enable = nodes.new("ShaderNodeMath")
    enable.name = f"{CUTOUT_PREFIX}Enable"
    enable.operation = "MULTIPLY"
    enable.inputs[0].default_value = 0.0
    exposure = nodes.new("ShaderNodeMath")
    exposure.name = f"{CUTOUT_PREFIX}Exposure"
    exposure.operation = "MULTIPLY"
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.name = f"{CUTOUT_PREFIX}Transparent"
    mix = nodes.new("ShaderNodeMixShader")
    mix.name = f"{CUTOUT_PREFIX}Mix"
    links.new(texcoord.outputs["Object"], scale.inputs[0])
    links.new(scale.outputs["Vector"], offset.inputs[0])
    links.new(offset.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], binary.inputs[0])
    links.new(texcoord.outputs["Object"], separate.inputs[0])
    links.new(separate.outputs["Z"], absolute.inputs[0])
    links.new(absolute.outputs[0], slab.inputs[0])
    links.new(binary.outputs[0], enable.inputs[1])
    links.new(enable.outputs[0], exposure.inputs[0])
    links.new(slab.outputs[0], exposure.inputs[1])
    links.new(exposure.outputs[0], mix.inputs[0])
    links.new(original, mix.inputs[1])
    links.new(transparent.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], surface)
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    return True


def setup_runtime(wheel_materials: dict[str, list], placeholder_mask_path: Path) -> dict:
    frame, carrier, rim_material, cavity_material = _setup_carrier()
    image = bpy.data.images.load(str(placeholder_mask_path), check_existing=False)
    image.name = IMAGE_NAME
    image.colorspace_settings.name = "Non-Color"
    installed = {}
    for wheel_name, materials in wheel_materials.items():
        installed[wheel_name] = [material.name for material in materials if _install_cutout(material, frame, image)]
    view_layer = bpy.context.scene.view_layers[0]
    aov = next((entry for entry in view_layer.aovs if entry.name == ANOMALY_AOV_NAME), None)
    if aov is None:
        aov = view_layer.aovs.add()
        aov.name = ANOMALY_AOV_NAME
    aov.type = "VALUE"
    return {
        "frame": frame,
        "carrier": carrier,
        "rim_material": rim_material,
        "cavity_material": cavity_material,
        "image": image,
        "wheel_materials": wheel_materials,
        "installed": installed,
    }


def _set_cutout_nodes(runtime: dict, target_name: str | None, extent: tuple[float, float] | None, slab_half_width: float, enabled: bool) -> None:
    for wheel_name, materials in runtime["wheel_materials"].items():
        value = 1.0 if enabled and wheel_name == target_name else 0.0
        for material in materials:
            if material.node_tree is None:
                continue
            nodes = material.node_tree.nodes
            enable = nodes.get(f"{CUTOUT_PREFIX}Enable")
            if enable is not None:
                enable.inputs[0].default_value = value
            scale = nodes.get(f"{CUTOUT_PREFIX}Scale")
            if scale is not None and extent is not None:
                scale.inputs[1].default_value = (1.0 / float(extent[0]), 1.0 / float(extent[1]), 1.0)
            slab = nodes.get(f"{CUTOUT_PREFIX}Slab")
            if slab is not None:
                slab.inputs[1].default_value = float(slab_half_width)


def disable(runtime: dict) -> None:
    _set_cutout_nodes(runtime, None, None, 0.003, False)
    runtime["carrier"].hide_render = True


def _active_rim_profile(anomaly_config: dict) -> tuple[str, dict]:
    geometry = anomaly_config["geometry"]
    name = str(geometry["active_rim_profile"])
    return name, geometry["rim_profiles"][name]


def _effective_rim_width(anomaly_config: dict, descriptor: dict, profile: dict) -> float:
    source = float(descriptor["rim_width_m"])
    if "effective_width_m" not in profile:
        return source * float(profile["width_scale"])
    source_low, source_high = map(float, anomaly_config["geometry"]["rim_width_m"])
    target_low, target_high = map(float, profile["effective_width_m"])
    unit = min(1.0, max(0.0, (source - source_low) / (source_high - source_low)))
    return target_low + unit * (target_high - target_low)


def _open_wall_depths(anomaly_config: dict, descriptor: dict, count: int) -> list[float]:
    """Return the approved T3 partial-fold depth at every contour vertex.

    The physical skin remains 0.75 mm deep around most of the opening. Two
    deterministic, smoothly tapered arcs fold farther inward, reaching a
    severity-dependent maximum. This avoids both a paper-thin edge everywhere
    and the artificial uniform tunnel produced by the legacy cavity recess.
    """

    geometry = anomaly_config["geometry"]
    if not bool(geometry.get("through_opening", False)):
        return [float(descriptor["cavity_recess_m"])] * count
    skin_depth = float(geometry["skin_thickness_m"])
    maximum_depth = float(geometry["open_wall_depth_max_by_severity_m"][descriptor["severity"]])
    profile_seed = int(descriptor["seeds"]["profile"])
    centres = (
        profile_seed % count,
        (profile_seed // 257 + count // 2) % count,
    )
    half_widths = (
        max(2, round(count * 0.17)),
        max(2, round(count * 0.13)),
    )
    depths = []
    for index in range(count):
        fold_weight = 0.0
        for centre, half_width in zip(centres, half_widths, strict=True):
            distance = abs(index - centre)
            distance = min(distance, count - distance)
            if distance <= half_width:
                taper = 0.5 + 0.5 * math.cos(math.pi * distance / half_width)
                fold_weight = max(fold_weight, taper)
        depths.append(skin_depth + (maximum_depth - skin_depth) * fold_weight)
    return depths


def _carrier_geometry(wheel: bpy.types.Object, resolved: dict, descriptor: dict, anomaly_config: dict):
    center = Vector(resolved["center_local_m"])
    long_axis = Vector(resolved["long_axis_local"])
    short_axis = Vector(resolved["short_axis_local"])
    normal = Vector(resolved["normal_local"])
    points = [Vector((float(value[0]), float(value[1]))) for value in descriptor["points_long_short_m"]]
    _profile_name, profile = _active_rim_profile(anomaly_config)
    rim_width = _effective_rim_width(anomaly_config, descriptor, profile)
    offset = float(profile["surface_offset_m"])
    inner = [Vector(value) for value in resolved["boundary_local_m"]]
    inner_normals = [Vector(value) for value in resolved["boundary_normals_local"]]
    wall_depths = _open_wall_depths(anomaly_config, descriptor, len(inner))
    outer = []
    outer_normals = []
    for point in points:
        direction = point.normalized() if point.length > 1e-9 else Vector((1.0, 0.0))
        expanded = point + direction * rim_width
        guess = center + long_axis * expanded.x + short_axis * expanded.y
        hit = _closest_surface(wheel, guess, maximum_distance=0.012)
        if hit is None:
            location, hit_normal = guess, normal
        else:
            location, hit_normal = hit[0], hit[1]
        outer.append(location)
        outer_normals.append(hit_normal)

    vertices = []
    for location, value_normal in zip(inner, inner_normals, strict=True):
        vertices.append(tuple(location + value_normal * offset))
    for location, value_normal in zip(outer, outer_normals, strict=True):
        vertices.append(tuple(location + value_normal * offset))
    for location, value_normal, wall_depth in zip(inner, inner_normals, wall_depths, strict=True):
        vertices.append(tuple(location - value_normal * wall_depth))
    faces = []
    materials = []
    count = len(inner)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((index, nxt, count + nxt, count + index))
        materials.append(0)
        faces.append((index, 2 * count + index, 2 * count + nxt, nxt))
        materials.append(1)

    flap = descriptor["flap"]
    if bool(flap["enabled"]):
        index = int(flap["edge_index"]) % count
        nxt = (index + 1) % count
        midpoint = (inner[index] + inner[nxt]) * 0.5
        tip = midpoint + normal * float(flap["lift_m"])
        base_a, base_b, tip_index = len(vertices), len(vertices) + 1, len(vertices) + 2
        vertices.extend((tuple(inner[index] + inner_normals[index] * offset), tuple(inner[nxt] + inner_normals[nxt] * offset), tuple(tip)))
        faces.append((base_a, base_b, tip_index))
        materials.append(0)
    return vertices, faces, materials


def enable(runtime: dict, wheel: bpy.types.Object, resolved: dict, descriptor: dict, anomaly_config: dict, mask_path: Path, mapping_extent_m: tuple[float, float]) -> None:
    frame = runtime["frame"]
    carrier = runtime["carrier"]
    center = Vector(resolved["center_local_m"])
    long_axis = Vector(resolved["long_axis_local"])
    short_axis = Vector(resolved["short_axis_local"])
    normal = Vector(resolved["normal_local"])
    frame.parent = wheel
    frame.matrix_parent_inverse = Matrix.Identity(4)
    frame.matrix_local = _matrix_from_frame(center, long_axis, short_axis, normal)
    carrier.parent = wheel
    carrier.matrix_parent_inverse = Matrix.Identity(4)
    carrier.matrix_local = Matrix.Identity(4)
    vertices, faces, material_indices = _carrier_geometry(wheel, resolved, descriptor, anomaly_config)
    mesh = carrier.data
    mesh.clear_geometry()
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    for polygon, material_index in zip(mesh.polygons, material_indices, strict=True):
        polygon.material_index = int(material_index)

    profile_name, profile = _active_rim_profile(anomaly_config)
    wall_depths = _open_wall_depths(anomaly_config, descriptor, len(resolved["boundary_local_m"]))
    clear_depth = float(anomaly_config["geometry"]["minimum_synthetic_clear_depth_m"])
    ray_origin = center + normal * max(0.001, float(profile["surface_offset_m"]) * 2.0)
    ray_hit, _ray_location, _ray_normal, ray_polygon = carrier.ray_cast(
        ray_origin,
        -normal,
        distance=clear_depth,
    )
    through_opening_gate = {
        "ok": not bool(ray_hit),
        "recessed_cap_enabled": False,
        "minimum_synthetic_clear_depth_m": clear_depth,
        "centerline_hit": bool(ray_hit),
        "centerline_hit_polygon": int(ray_polygon) if ray_hit else None,
        "carrier_face_count": len(mesh.polygons),
    }

    material = descriptor["material"]
    rim_principled = next(node for node in runtime["rim_material"].node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    exposed = float(material["rim_exposed_fraction"]) * float(profile["exposed_fraction_scale"])
    dusty = Vector((0.22, 0.09, 0.035))
    metal = Vector((0.52, 0.47, 0.40))
    rim_color = dusty.lerp(metal, exposed)
    rim_principled.inputs["Base Color"].default_value = (*rim_color, 1.0)
    rim_principled.inputs["Roughness"].default_value = 0.82 - 0.28 * exposed
    rim_principled.inputs["Metallic"].default_value = 0.35 + 0.55 * exposed
    cavity_principled = next(node for node in runtime["cavity_material"].node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    cavity_factor = float(material["cavity_luminance_factor"])
    # Keep the recessed cap physically non-black while preserving enough
    # contrast when the healthy tread is already in deep Martian shadow.
    cavity_color = Vector((0.06 * cavity_factor, 0.03 * cavity_factor, 0.015 * cavity_factor))
    cavity_principled.inputs["Base Color"].default_value = (*cavity_color, 1.0)
    cavity_principled.inputs["Roughness"].default_value = float(material["cavity_roughness"])
    cavity_principled.inputs["Metallic"].default_value = 0.18

    image = runtime["image"]
    image.filepath_raw = str(Path(mask_path).resolve())
    image.source = "FILE"
    image.reload()
    sagitta = (float(mapping_extent_m[1]) ** 2) / max(8.0 * float(resolved["wheel_radius_m"]), 1e-9)
    slab_half_width = max(0.003, sagitta + 0.0015)
    _set_cutout_nodes(runtime, wheel.name, mapping_extent_m, slab_half_width, True)
    carrier.hide_render = False
    bpy.context.view_layer.update()
    return {
        "rim_base_color": list(map(float, rim_color)),
        "rim_profile": profile_name,
        "rim_width_source_m": float(descriptor["rim_width_m"]),
        "rim_width_effective_m": _effective_rim_width(anomaly_config, descriptor, profile),
        "rim_surface_offset_m": float(profile["surface_offset_m"]),
        "rim_exposed_fraction_source": float(material["rim_exposed_fraction"]),
        "rim_exposed_fraction_effective": exposed,
        "rim_roughness": float(rim_principled.inputs["Roughness"].default_value),
        "rim_metallic": float(rim_principled.inputs["Metallic"].default_value),
        "cavity_base_color": list(map(float, cavity_color)),
        "cavity_roughness": float(cavity_principled.inputs["Roughness"].default_value),
        "cavity_metallic": float(cavity_principled.inputs["Metallic"].default_value),
        "requested_cavity_recess_m": float(descriptor["cavity_recess_m"]),
        "open_wall_depth_profile": str(anomaly_config["geometry"]["open_wall_depth_profile"]),
        "open_wall_fold_arc_count": int(anomaly_config["geometry"]["open_wall_fold_arc_count"]),
        "effective_open_wall_depth_m": max(wall_depths),
        "effective_open_wall_depth_min_m": min(wall_depths),
        "effective_open_wall_depth_max_m": max(wall_depths),
        "through_opening_gate": through_opening_gate,
        "emission_strength": 0.0,
    }


def set_compositor_mask_aov(compositor, aov_name: str) -> None:
    render_layers = compositor.nodes["CleanBatch_RenderLayers"]
    socket = render_layers.outputs.get(str(aov_name))
    if socket is None:
        raise RuntimeError(f"Render Layers does not expose required AOV {aov_name}")
    target = compositor.nodes["CleanBatch_MaskScale"].inputs[0]
    for link in list(compositor.links):
        if link.to_socket == target:
            compositor.links.remove(link)
    compositor.links.new(socket, target)
