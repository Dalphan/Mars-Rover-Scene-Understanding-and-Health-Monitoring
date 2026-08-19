"""Render one deterministic clean-batch chunk from a single loaded Blend file."""

from __future__ import annotations

import argparse
import binascii
import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bpy


PNG_COMPRESSION_LEVEL = 4
import gpu
from mathutils import Matrix

AOV_NAME = "CleanBatchTargetWheel"
AOV_NODE_NAME = "CleanBatch_TargetWheelAOV"
WEAR_IMAGE_NAME = "CleanBatch_WearMask"


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--domain-config", type=Path, required=True)
    parser.add_argument("--chunk-plan", type=Path, required=True)
    parser.add_argument("--lighting-config", type=Path, required=True)
    parser.add_argument("--wear-config", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--chunk-report", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return tuple(
        value.resolve()
        for value in (
            args.batch_config,
            args.domain_config,
            args.chunk_plan,
            args.lighting_config,
            args.wear_config,
            args.pose_config,
            args.sampling_config,
            args.run_dir,
            args.manifest,
            args.chunk_report,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix(values) -> Matrix:
    values = list(map(float, values))
    return Matrix([values[index : index + 4] for index in range(0, 16, 4)])


def _matrix_values(matrix: Matrix) -> list[float]:
    return [float(matrix[row][column]) for row in range(4) for column in range(4)]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Malformed manifest line {number}") from error
    return rows


def _append_manifest(path: Path, row: dict) -> None:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _jittered_preset(base: dict, sample: dict) -> dict:
    preset = copy.deepcopy(base)
    jitter = sample["lighting_jitter"]
    preset["sun"]["azimuth_deg"] = float(preset["sun"]["azimuth_deg"]) + float(jitter["sun_azimuth_degrees"])
    preset["sun"]["elevation_deg"] = float(preset["sun"]["elevation_deg"]) + float(jitter["sun_elevation_degrees"])
    preset["sun"]["energy"] = float(preset["sun"]["energy"]) * float(jitter["sun_energy_scale"])
    preset["world"]["strength"] = float(preset["world"]["strength"]) * float(jitter["world_strength_scale"])
    preset["exposure_ev"] = float(preset["exposure_ev"]) + float(jitter["exposure_ev"])
    return preset


def _install_wear(materials, target_wheel, dimensions: dict, image, settings: dict) -> list[str]:
    from scripts.blender.render_wheel_surface_wear_pilot import (
        _mix_float,
        _multiply,
        _surface_exposure_mask,
    )

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
        exposed = _multiply(nodes, links, texture.outputs["Color"], exposure, "WheelWear_ExposedTexture")
        opacity = nodes.new("ShaderNodeMath")
        opacity.name = "WheelWear_Opacity"
        opacity.operation = "MULTIPLY"
        opacity.inputs[0].default_value = 0.0
        links.new(exposed, opacity.inputs[1])
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


def _install_aov(materials) -> None:
    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        nodes = material.node_tree.nodes
        old = nodes.get(AOV_NODE_NAME)
        if old is not None:
            nodes.remove(old)
        node = nodes.new("ShaderNodeOutputAOV")
        node.name = AOV_NODE_NAME
        node.label = "Target wheel mask"
        node.aov_name = AOV_NAME
        node.inputs["Value"].default_value = 0.0


def _setup_compositor(scene):
    view_layer = scene.view_layers[0]
    existing = next((entry for entry in view_layer.aovs if entry.name == AOV_NAME), None)
    if existing is None:
        existing = view_layer.aovs.add()
        existing.name = AOV_NAME
    existing.type = "VALUE"

    old = bpy.data.node_groups.get("CleanBatch_Compositor")
    if old is not None:
        bpy.data.node_groups.remove(old, do_unlink=True)
    tree = bpy.data.node_groups.new("CleanBatch_Compositor", "CompositorNodeTree")
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = "CleanBatch_RenderLayers"
    render_layers.layer = view_layer.name
    hue = tree.nodes.new("CompositorNodeHueSat")
    hue.name = "CleanBatch_PostHue"
    veil = tree.nodes.new("CompositorNodeAlphaOver")
    veil.name = "CleanBatch_PostVeil"
    output = tree.nodes.new("NodeGroupOutput")
    output.name = "CleanBatch_Output"
    tree.links.new(render_layers.outputs["Image"], hue.inputs["Image"])
    tree.links.new(hue.outputs["Image"], veil.inputs["Background"])

    aov_socket = render_layers.outputs.get(AOV_NAME)
    if aov_socket is None:
        raise RuntimeError(f"Render Layers does not expose required shader AOV {AOV_NAME}")
    # Replacing opaque scene alpha with a 0/1 mask makes Blender un-premultiply
    # antialiased boundary pixels against a near-zero alpha when writing PNG.
    # That produces a bright fringe in RGB even when alpha is later discarded.
    # Encode the mask in [0.5, 1.0] and premultiply RGB by that encoded alpha;
    # PNG straight-alpha conversion then reconstructs the original RGB bytes.
    mask_scale = tree.nodes.new("ShaderNodeMath")
    mask_scale.name = "CleanBatch_MaskScale"
    mask_scale.operation = "MULTIPLY"
    mask_scale.inputs[1].default_value = 0.5
    mask_encode = tree.nodes.new("ShaderNodeMath")
    mask_encode.name = "CleanBatch_MaskEncode"
    mask_encode.operation = "ADD"
    mask_encode.inputs[1].default_value = 0.5
    premultiply = tree.nodes.new("ShaderNodeVectorMath")
    premultiply.name = "CleanBatch_PremultiplyPackedAlpha"
    premultiply.operation = "SCALE"
    tree.links.new(aov_socket, mask_scale.inputs[0])
    tree.links.new(mask_scale.outputs[0], mask_encode.inputs[0])
    tree.links.new(veil.outputs["Image"], premultiply.inputs[0])
    tree.links.new(mask_encode.outputs[0], premultiply.inputs[3])

    set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
    set_alpha.name = "CleanBatch_PackTargetMask"
    set_alpha.inputs["Type"].default_value = "Replace Alpha"
    tree.links.new(premultiply.outputs["Vector"], set_alpha.inputs["Image"])
    tree.links.new(mask_encode.outputs[0], set_alpha.inputs["Alpha"])
    tree.links.new(set_alpha.outputs["Image"], output.inputs["Image"])
    scene.compositing_node_group = tree
    return tree


def _set_post(tree, preset: dict) -> None:
    hue = tree.nodes["CleanBatch_PostHue"]
    veil = tree.nodes["CleanBatch_PostVeil"]
    if bool(preset["camera_post"]):
        hue.inputs["Saturation"].default_value = float(preset["post"]["saturation"])
        hue.inputs["Value"].default_value = float(preset["post"]["value"])
        veil.inputs["Factor"].default_value = float(preset["post"]["veil_factor"])
        veil.inputs["Foreground"].default_value = tuple(map(float, preset["post"]["veil_color"]))
    else:
        hue.inputs["Saturation"].default_value = 1.0
        hue.inputs["Value"].default_value = 1.0
        veil.inputs["Factor"].default_value = 0.0


def _set_dust(materials, preset: dict, dusty_reference: dict) -> None:
    enabled = bool(preset["dust_layer"])
    coverage = float(preset["dust"]["coverage"]) if enabled else 0.0
    bump_strength = float(preset["dust"]["micro_bump_strength"]) if enabled else 0.0
    for material in materials:
        if material.node_tree is None:
            continue
        coverage_node = material.node_tree.nodes.get("MarsDust_Coverage")
        if coverage_node is not None:
            coverage_node.inputs[0].default_value = coverage
        for node in material.node_tree.nodes:
            if node.name.startswith("MarsDust_MicroBump_"):
                node.inputs["Strength"].default_value = bump_strength
                node.inputs["Distance"].default_value = float(dusty_reference["micro_bump_distance_m"])


def _set_wear(wheel_materials: dict[str, list], target_name: str, enabled: bool, opacity: float) -> None:
    for wheel_name, materials in wheel_materials.items():
        value = float(opacity) if enabled and wheel_name == target_name else 0.0
        for material in materials:
            node = material.node_tree.nodes.get("WheelWear_Opacity") if material.node_tree else None
            if node is not None:
                node.inputs[0].default_value = value


def _set_aov_target(wheel_materials: dict[str, list], target_name: str) -> None:
    for wheel_name, materials in wheel_materials.items():
        value = 1.0 if wheel_name == target_name else 0.0
        for material in materials:
            node = material.node_tree.nodes.get(AOV_NODE_NAME) if material.node_tree else None
            if node is not None:
                node.inputs["Value"].default_value = value


def _datablock_counts() -> dict:
    return {
        "objects": len(bpy.data.objects),
        "meshes": len(bpy.data.meshes),
        "materials": len(bpy.data.materials),
        "images": len(bpy.data.images),
        "cameras": len(bpy.data.cameras),
        "node_groups": len(bpy.data.node_groups),
        "material_nodes": sum(len(material.node_tree.nodes) for material in bpy.data.materials if material.node_tree),
    }


def _inspect_output(path: Path, resolution: tuple[int, int], *, mask: bool) -> dict:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Output is not a PNG: {path}")
    position = 8
    idat = bytearray()
    width = height = bit_depth = color_type = interlace = None
    while position < len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        chunk_type = payload[position + 4 : position + 8]
        data = payload[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            if compression != 0 or filtering != 0:
                raise RuntimeError("Unsupported PNG compression/filter method")
        elif chunk_type == b"IDAT":
            idat.extend(data)
        elif chunk_type == b"IEND":
            break
    expected_color = 0 if mask else 2
    if (
        (width, height) != tuple(map(int, resolution))
        or bit_depth != 8
        or color_type != expected_color
        or interlace != 0
    ):
        raise RuntimeError(
            f"Unexpected PNG contract for {path}: "
            f"{(width, height, bit_depth, color_type, interlace)}"
        )
    channels = 1 if mask else 3
    stride = int(width) * channels
    raw = zlib.decompress(bytes(idat))
    if len(raw) != int(height) * (stride + 1):
        raise RuntimeError(f"Unexpected decompressed PNG size for {path}")
    previous = bytearray(stride)
    values = bytearray()
    offset = 0
    for _row_index in range(int(height)):
        filter_type = raw[offset]
        scan = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        for index in range(stride):
            left = scan[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                scan[index] = (scan[index] + left) & 0xFF
            elif filter_type == 2:
                scan[index] = (scan[index] + up) & 0xFF
            elif filter_type == 3:
                scan[index] = (scan[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                scan[index] = (scan[index] + _paeth(left, up, upper_left)) & 0xFF
            elif filter_type != 0:
                raise RuntimeError(f"Unsupported PNG row filter: {filter_type}")
        if mask:
            values.extend(scan)
        previous = scan
    result = {"width": int(width), "height": int(height)}
    if mask:
        unique = set(values)
        if unique != {0, 255}:
            raise RuntimeError(f"Target mask must be non-empty, non-full and binary: values={sorted(unique)}")
        result.update({"minimum": 0.0, "maximum": 1.0, "binary": True})
    return result


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    distances = (abs(estimate - left), abs(estimate - up), abs(estimate - upper_left))
    return (left, up, upper_left)[distances.index(min(distances))]


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)


def _encode_png(path: Path, width: int, height: int, color_type: int, rows: list[bytes], ancillary: list[tuple[bytes, bytes]]) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    filtered = b"".join(b"\x00" + row for row in rows)
    payload = bytearray(signature)
    payload.extend(_png_chunk(b"IHDR", header))
    for chunk_type, data in ancillary:
        payload.extend(_png_chunk(chunk_type, data))
    payload.extend(_png_chunk(b"IDAT", zlib.compress(filtered, level=int(PNG_COMPRESSION_LEVEL))))
    payload.extend(_png_chunk(b"IEND", b""))
    path.write_bytes(bytes(payload))


def _split_rgba_png(source: Path, rgb_path: Path, mask_path: Path, resolution: tuple[int, int]) -> None:
    payload = source.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError("Combined render is not a PNG")
    position = 8
    idat = bytearray()
    ancillary = []
    width = height = bit_depth = color_type = interlace = None
    while position < len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        chunk_type = payload[position + 4 : position + 8]
        data = payload[position + 8 : position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            if compression != 0 or filtering != 0:
                raise RuntimeError("Unsupported PNG compression/filter method")
        elif chunk_type == b"IDAT":
            idat.extend(data)
        elif chunk_type in {b"sRGB", b"gAMA", b"cHRM", b"iCCP", b"pHYs"}:
            ancillary.append((chunk_type, data))
        elif chunk_type == b"IEND":
            break
    if (width, height) != tuple(map(int, resolution)) or bit_depth != 8 or color_type != 6 or interlace != 0:
        raise RuntimeError(
            f"Combined PNG must be non-interlaced 8-bit RGBA at {resolution}: "
            f"got {(width, height, bit_depth, color_type, interlace)}"
        )
    raw = zlib.decompress(bytes(idat))
    stride = int(width) * 4
    expected = int(height) * (stride + 1)
    if len(raw) != expected:
        raise RuntimeError(f"Unexpected decompressed PNG size: {len(raw)} != {expected}")
    previous = bytearray(stride)
    rgba_rows = []
    offset = 0
    for _row_index in range(int(height)):
        filter_type = raw[offset]
        scan = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        for index in range(stride):
            left = scan[index - 4] if index >= 4 else 0
            up = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 1:
                scan[index] = (scan[index] + left) & 0xFF
            elif filter_type == 2:
                scan[index] = (scan[index] + up) & 0xFF
            elif filter_type == 3:
                scan[index] = (scan[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                scan[index] = (scan[index] + _paeth(left, up, upper_left)) & 0xFF
            elif filter_type != 0:
                raise RuntimeError(f"Unsupported PNG row filter: {filter_type}")
        rgba_rows.append(bytes(scan))
        previous = scan
    rgb_rows = [bytes(channel for index, channel in enumerate(row) if index % 4 != 3) for row in rgba_rows]
    # Alpha stores 0.5 + 0.5 * target coverage. The midpoint (0.75) is the
    # binary silhouette threshold, while the non-zero floor preserves RGB.
    mask_rows = [bytes(255 if row[index] >= 192 else 0 for index in range(3, len(row), 4)) for row in rgba_rows]
    _encode_png(rgb_path, int(width), int(height), 2, rgb_rows, ancillary)
    _encode_png(mask_path, int(width), int(height), 0, mask_rows, [entry for entry in ancillary if entry[0] == b"pHYs"])


def _finalize_sample_artifacts(
    *,
    manifest_path: Path,
    run_dir: Path,
    stage_dir: Path,
    staged_combined: Path,
    staged_rgb: Path,
    staged_mask: Path,
    resolution: tuple[int, int],
    artifact_sample_id: str,
    row: dict,
    sample_started: float,
) -> dict:
    """CPU-only PNG split, validation and commit; safe outside Blender's main thread."""

    write_started = time.perf_counter()
    _split_rgba_png(staged_combined, staged_rgb, staged_mask, resolution)
    staged_combined.unlink()
    rgb_info = _inspect_output(staged_rgb, resolution, mask=False)
    mask_info = _inspect_output(staged_mask, resolution, mask=True)
    rgb_relative = Path("rgb") / f"{artifact_sample_id}.png"
    mask_relative = Path("target_wheel_mask") / f"{artifact_sample_id}.png"
    rgb_final = run_dir / rgb_relative
    mask_final = run_dir / mask_relative
    rgb_final.parent.mkdir(parents=True, exist_ok=True)
    mask_final.parent.mkdir(parents=True, exist_ok=True)
    rgb_sha256 = _sha256(staged_rgb)
    mask_sha256 = _sha256(staged_mask)
    os.replace(staged_rgb, rgb_final)
    os.replace(staged_mask, mask_final)
    shutil.rmtree(stage_dir)
    row["artifacts"] = {
        "rgb": {"path": rgb_relative.as_posix(), "sha256": rgb_sha256, **rgb_info},
        "target_wheel_mask": {"path": mask_relative.as_posix(), "sha256": mask_sha256, **mask_info},
    }
    row["render"]["write_seconds"] = float(time.perf_counter() - write_started)
    row["render"]["total_seconds"] = float(time.perf_counter() - sample_started)
    _append_manifest(manifest_path, row)
    return row


def render_chunk(
    batch_config_path: Path,
    domain_config_path: Path,
    chunk_plan_path: Path,
    lighting_config_path: Path,
    wear_config_path: Path,
    pose_config_path: Path,
    sampling_config_path: Path,
    run_dir: Path,
    manifest_path: Path,
    chunk_report_path: Path,
) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.static_geometry_cache import StaticGeometryCache
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender import render_wheel_surface_wear_pilot as wear
    from scripts.blender.render_domain_randomization_preview import (
        _align_rover_to_patch,
        _framing_gate,
        _jitter_camera,
        _settle_rover_on_patch,
        _terrain_footprint,
    )
    from scripts.blender.render_wheel_camera_pose_pilot import _apply_terrain_color_grade
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose, restore_base_roll
    from src.wheel_preparation.clean_batch import assert_camera_retry_preserves_semantics
    from src.wheel_preparation.domain_randomization import sample_domain_randomization

    batch_config = json.loads(batch_config_path.read_text(encoding="utf-8"))
    domain_config = json.loads(domain_config_path.read_text(encoding="utf-8"))
    chunk_plan = json.loads(chunk_plan_path.read_text(encoding="utf-8"))
    lighting_config = json.loads(lighting_config_path.read_text(encoding="utf-8"))
    wear_config = json.loads(wear_config_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_config_path.read_text(encoding="utf-8"))
    sampling_config = json.loads(sampling_config_path.read_text(encoding="utf-8"))
    lighting_by_id = {entry["id"]: entry for entry in lighting_config["presets"]}
    wear_by_id = {entry["id"]: entry for entry in wear_config["variants"]}
    pose_by_id = {entry["id"]: entry for entry in pose_config["poses"]}

    scene = bpy.context.scene
    if not bool(scene.get("wheel_pose_sampling_ready", False)):
        raise RuntimeError("Clean-batch source is not pose-sampling ready")
    if tuple(bpy.app.version[:2]) != tuple(map(int, batch_config["render"]["blender_major_minor"])):
        raise RuntimeError(f"Blender version mismatch: {bpy.app.version_string}")
    scene.render.engine = str(batch_config["render"]["engine"])
    resolution = tuple(map(int, batch_config["render"]["resolution"]))
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = int(batch_config["render"]["samples"])
    scene.frame_set(1)

    setup_started = time.perf_counter()
    wheels = [bpy.data.objects[name] for name in sampling_config["wheels"]]
    root = bpy.data.objects[domain_config["gates"]["terrain_alignment"]["rover_root_object"]]
    patch = bpy.data.objects.get(domain_config["gates"]["terrain_patch_object"])
    if patch is None or patch.type != "MESH":
        raise RuntimeError("Microterrain patch required by the clean-batch gate is missing")
    base_root_matrix = root.matrix_world.copy()
    base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}

    wheel_materials = {}
    wheel_geometry = {}
    affected_wear_objects = {}
    dimensions_by_wheel = {}
    for wheel_root in wheels:
        target_meshes = wear._target_meshes(wheel_root)
        wheel_geometry[wheel_root.name] = wear._geometry_signature(target_meshes)
        dimensions_by_wheel[wheel_root.name] = wear._wheel_dimensions(wheel_root, target_meshes)
        materials, affected = wear._clone_target_materials(target_meshes, f"clean_batch_{wheel_root.name}")
        wheel_materials[wheel_root.name] = materials
        affected_wear_objects[wheel_root.name] = affected

    graded_materials = _apply_terrain_color_grade(lighting_config["terrain_grade"])
    dusty_preset = lighting_by_id["mars_dusty_refined"]
    dusty_material_names = lighting._apply_dust_layer(dusty_preset["dust"])
    dusty_materials = [bpy.data.materials[name] for name in dusty_material_names]

    placeholder_path = Path(chunk_plan["placeholder_wear_mask"]).resolve()
    wear_image = bpy.data.images.load(str(placeholder_path), check_existing=False)
    wear_image.name = WEAR_IMAGE_NAME
    wear_image.colorspace_settings.name = "Non-Color"
    installed_wear = {}
    for wheel_root in wheels:
        installed_wear[wheel_root.name] = _install_wear(
            wheel_materials[wheel_root.name],
            wheel_root,
            dimensions_by_wheel[wheel_root.name],
            wear_image,
            wear_config["shader"],
        )
        _install_aov(wheel_materials[wheel_root.name])

    camera = lighting._configure_camera(pose_config["camera"])
    scene.camera = camera
    compositor = _setup_compositor(scene)
    benchmark = batch_config.get("benchmark", {})
    global PNG_COMPRESSION_LEVEL
    PNG_COMPRESSION_LEVEL = int(benchmark.get("png_compression_level", 4))
    scene.render.image_settings.compression = int(benchmark.get("scene_png_compression", 15))
    geometry_cache = (
        StaticGeometryCache(wheels, patch, domain_config["gates"]["terrain_alignment"]["vertical_contact"])
        if bool(benchmark.get("geometry_cache", False))
        else None
    )
    async_postprocess = bool(batch_config.get("execution", {}).get("async_postprocess", benchmark.get("async_postprocess", False)))
    async_queue_depth = int(batch_config.get("execution", {}).get("async_queue_depth", 4))
    bpy.context.view_layer.update()
    setup_counts = _datablock_counts()
    setup_seconds = time.perf_counter() - setup_started
    runtime = {"blender_version": bpy.app.version_string, "engine": scene.render.engine, "device": None}

    completed_ids = {row["sample_id"] for row in _read_jsonl(manifest_path)}
    rendered = []
    render_call_count = 0
    futures = []
    pipeline_started = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clean-png") if async_postprocess else None
    for planned_sample in chunk_plan["samples"]:
        artifact_sample_id = str(planned_sample.get("artifact_sample_id", planned_sample["sample_id"]))
        if artifact_sample_id in completed_ids:
            continue
        sample_started = time.perf_counter()
        restore_base_roll(wheels, base_local)
        root.matrix_world = base_root_matrix.copy()
        bpy.context.view_layer.update()
        target = bpy.data.objects[planned_sample["target_wheel"]]
        rover_translation = _align_rover_to_patch(
            scene,
            target,
            domain_config["gates"]["terrain_alignment"],
            base_root_matrix,
        )
        apply_shared_roll(wheels, base_local, float(planned_sample["healthy_roll_degrees"]))
        contact = (
            geometry_cache.settle(target, patch, domain_config["gates"]["terrain_alignment"])
            if geometry_cache
            else _settle_rover_on_patch(scene, target, patch, domain_config["gates"]["terrain_alignment"])
        )
        rover_translation = root.matrix_world.translation - base_root_matrix.translation
        if contact["ok"] is not True:
            raise RuntimeError(f"Wheel/terrain contact gate failed for {planned_sample['sample_id']}: {contact}")
        pose = pose_by_id[planned_sample["camera_pose"]]
        rejections = []
        maximum_attempts = int(domain_config["gates"]["maximum_camera_resample_attempts"])
        for attempt in range(maximum_attempts):
            candidate = sample_domain_randomization(domain_config, int(planned_sample["sample_index"]), attempt=attempt)
            assert_camera_retry_preserves_semantics(planned_sample, candidate)
            camera.data.lens = float(pose_config["camera"]["focal_length_mm"]) * float(candidate["camera_jitter"]["focal_length_scale"])
            basis = camera_pose(camera, target, pose, pose_config["camera"])
            aim_target = _jitter_camera(camera, basis, candidate)
            footprint = (
                geometry_cache.footprint(scene, camera, float(domain_config["gates"]["terrain_minimum_edge_margin_m"]))
                if geometry_cache
                else _terrain_footprint(scene, camera, patch, float(domain_config["gates"]["terrain_minimum_edge_margin_m"]))
            )
            framing = (
                geometry_cache.framing(scene, camera, target, basis, pose, domain_config["gates"])
                if geometry_cache
                else _framing_gate(scene, camera, target, basis, pose, domain_config["gates"])
            )
            if footprint["ok"] and framing["ok"]:
                sample = {"schema_version": 1, "condition": "clean", **candidate}
                break
            rejections.append(
                {
                    "attempt": attempt,
                    "camera_attempt_seed": candidate["camera_attempt_seed"],
                    "terrain_reason": footprint["reason"],
                    "terrain_minimum_edge_margin_m": footprint.get("minimum_edge_margin_m"),
                    "framing_bbox": framing["normalized_bbox"],
                    "framing_bbox_area_fraction": framing["bbox_area_fraction"],
                }
            )
        else:
            raise RuntimeError(f"Camera gates exhausted for {planned_sample['sample_id']}")

        preset = _jittered_preset(lighting_by_id[sample["lighting_preset"]], sample)
        lighting._configure_sun_world(preset)
        scene.view_settings.look = preset["look"]
        scene.view_settings.exposure = float(preset["exposure_ev"])
        _set_post(compositor, preset)
        _set_dust(dusty_materials, preset, dusty_preset["dust"])
        wear_variant = wear_by_id[sample["surface_wear"]]
        wear_enabled = bool(wear_variant["wear_enabled"])
        if wear_enabled:
            mask_path = Path(planned_sample["wear_mask"]).resolve()
            wear_image.filepath_raw = str(mask_path)
            wear_image.source = "FILE"
            wear_image.reload()
        _set_wear(wheel_materials, target.name, wear_enabled, float(wear_config["shader"]["opacity"]))
        _set_aov_target(wheel_materials, target.name)
        bpy.context.view_layer.update()
        sample_setup_seconds = time.perf_counter() - sample_started

        stage_dir = (run_dir / "_staging" / artifact_sample_id).resolve()
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=False)
        staged_combined = stage_dir / "combined.png"
        staged_rgb = stage_dir / "rgb.png"
        staged_mask = stage_dir / "mask.png"
        scene.render.filepath = str(staged_combined)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.compression = int(benchmark.get("scene_png_compression", 15))
        render_started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        render_call_count += 1
        render_seconds = time.perf_counter() - render_started
        if runtime["device"] is None:
            runtime["device"] = {
                "vendor": gpu.platform.vendor_get(),
                "renderer": gpu.platform.renderer_get(),
                "version": gpu.platform.version_get(),
                "backend": gpu.platform.backend_type_get(),
                "type": gpu.platform.device_type_get(),
            }
        sun = bpy.data.objects["GaleNominalSun"]
        background = scene.world.node_tree.nodes["Background"]
        row = {
            "schema_version": 1,
            "sample_id": artifact_sample_id,
            "semantic_sample_id": sample["sample_id"],
            "sample_index": int(sample["sample_index"]),
            "condition": "clean",
            "seeds": {
                "sample": int(sample["sample_seed"]),
                "camera_attempt": int(sample["camera_attempt_seed"]),
                "wear": int(sample["wear_seed"]),
                "attempt": int(sample["attempt"]),
                "pair_lock_id": sample["pair_lock_id"],
            },
            "sampling": {
                "target_wheel": sample["target_wheel"],
                "camera_pose": sample["camera_pose"],
                "healthy_roll_degrees": float(sample["healthy_roll_degrees"]),
                "lighting_preset": sample["lighting_preset"],
                "surface_wear": sample["surface_wear"],
                "lighting_jitter": sample["lighting_jitter"],
                "camera_jitter": sample["camera_jitter"],
            },
            "resolved": {
                "camera_location_m": list(map(float, camera.location)),
                "camera_matrix_world": _matrix_values(camera.matrix_world),
                "aim_target_m": list(map(float, aim_target)),
                "focal_length_mm": float(camera.data.lens),
                "wheel_matrix_world": _matrix_values(target.matrix_world),
                "rover_matrix_world": _matrix_values(root.matrix_world),
                "rover_patch_alignment_translation_m": list(map(float, rover_translation)),
                "lighting": {
                    "sun_energy": float(sun.data.energy),
                    "sun_angle_degrees": math.degrees(float(sun.data.angle)),
                    "sun_color": list(map(float, sun.data.color)),
                    "world_strength": float(background.inputs["Strength"].default_value),
                    "exposure_ev": float(scene.view_settings.exposure),
                },
            },
            "geometry": wheel_geometry[target.name],
            "gates": {
                "framing": True,
                "terrain_coverage": True,
                "terrain_contact": True,
                "framing_details": framing,
                "terrain_details": footprint,
                "terrain_contact_details": contact,
                "camera_rejections": rejections,
            },
            "render": {
                "engine": scene.render.engine,
                "blender_version": bpy.app.version_string,
                "resolution": list(resolution),
                "samples": int(scene.eevee.taa_render_samples),
                "device": runtime["device"],
                "setup_seconds": float(sample_setup_seconds),
                "render_seconds": float(render_seconds),
            },
        }
        finalize_kwargs = {
            "manifest_path": manifest_path,
            "run_dir": run_dir,
            "stage_dir": stage_dir,
            "staged_combined": staged_combined,
            "staged_rgb": staged_rgb,
            "staged_mask": staged_mask,
            "resolution": resolution,
            "artifact_sample_id": artifact_sample_id,
            "row": row,
            "sample_started": sample_started,
        }
        if executor is None:
            rendered.append(_finalize_sample_artifacts(**finalize_kwargs))
        else:
            futures.append(executor.submit(_finalize_sample_artifacts, **finalize_kwargs))
            if len(futures) >= async_queue_depth:
                rendered.append(futures.pop(0).result())
        if executor is None:
            current_counts = _datablock_counts()
            if current_counts != setup_counts:
                raise RuntimeError(f"Blender datablock drift after {sample['sample_id']}: {setup_counts} != {current_counts}")
        print(f"CLEAN_BATCH_SAMPLE {artifact_sample_id} render={render_seconds:.3f}s", flush=True)

    if executor is not None:
        executor.shutdown(wait=True)
        rendered.extend(future.result() for future in futures)
        bpy.context.view_layer.update()
        current_counts = _datablock_counts()
        if current_counts != setup_counts:
            raise RuntimeError(f"Blender datablock drift after asynchronous queue drain: {setup_counts} != {current_counts}")
    pipeline_seconds = time.perf_counter() - pipeline_started
    cpu_postprocess_seconds = sum(float(row["render"]["write_seconds"]) for row in rendered)

    report = {
        "schema_version": 1,
        "ok": True,
        "source_open_count": 1,
        "planned_count": len(chunk_plan["samples"]),
        "rendered_count": len(rendered),
        "render_call_count": render_call_count,
        "runtime": runtime,
        "setup_seconds": float(setup_seconds),
        "chunk_pipeline_seconds": float(pipeline_seconds),
        "cpu_postprocess_seconds": float(cpu_postprocess_seconds),
        "setup_datablocks": setup_counts,
        "graded_materials": graded_materials,
        "dusty_materials": dusty_material_names,
        "wear_materials": installed_wear,
        "benchmark": {
            "geometry_cache": geometry_cache is not None,
            "async_postprocess": async_postprocess,
            "async_queue_depth": async_queue_depth if async_postprocess else 0,
            "png_compression_level": int(PNG_COMPRESSION_LEVEL),
            "scene_png_compression": int(benchmark.get("scene_png_compression", 15)),
        },
    }
    chunk_report_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    render_chunk(*_arguments())
