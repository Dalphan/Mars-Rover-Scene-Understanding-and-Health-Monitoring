"""Render one persistent deterministic paired clean/hole chunk in Blender."""

from __future__ import annotations

import argparse
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
from collections import deque
from pathlib import Path

import bpy
import gpu
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-config", type=Path, required=True)
    parser.add_argument("--domain-config", type=Path, required=True)
    parser.add_argument("--anomaly-config", type=Path, required=True)
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
            args.anomaly_config,
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


def _decode_png(path: Path) -> tuple[int, int, int, list[bytes]]:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise RuntimeError(f"Not a PNG: {path}")
    position = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    while position < len(payload):
        length = struct.unpack(">I", payload[position : position + 4])[0]
        kind = payload[position + 4 : position + 8]
        data = payload[position + 8 : position + 8 + length]
        position += length + 12
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", data)
            if compression or filtering:
                raise RuntimeError("Unsupported PNG compression/filter method")
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    channels = {0: 1, 2: 3}.get(color_type)
    if bit_depth != 8 or channels is None or interlace != 0:
        raise RuntimeError(f"Expected non-interlaced 8-bit L/RGB PNG: {path}")
    stride = int(width) * channels
    raw = zlib.decompress(bytes(compressed))
    if len(raw) != int(height) * (stride + 1):
        raise RuntimeError(f"Unexpected decompressed PNG size: {path}")
    previous = bytearray(stride)
    rows = []
    offset = 0
    from scripts.blender.render_clean_batch import _paeth

    for _ in range(int(height)):
        filter_type = raw[offset]
        row = bytearray(raw[offset + 1 : offset + stride + 1])
        offset += stride + 1
        for index in range(stride):
            left = row[index - channels] if index >= channels else 0
            up = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                row[index] = (row[index] + left) & 0xFF
            elif filter_type == 2:
                row[index] = (row[index] + up) & 0xFF
            elif filter_type == 3:
                row[index] = (row[index] + ((left + up) // 2)) & 0xFF
            elif filter_type == 4:
                row[index] = (row[index] + _paeth(left, up, upper_left)) & 0xFF
            elif filter_type != 0:
                raise RuntimeError(f"Unsupported PNG filter {filter_type}: {path}")
        rows.append(bytes(row))
        previous = row
    return int(width), int(height), channels, rows


def _mask_pixels(path: Path, resolution: tuple[int, int]) -> tuple[set[int], list[bytes]]:
    width, height, channels, rows = _decode_png(path)
    if (width, height) != resolution or channels != 1:
        raise RuntimeError(f"Mask shape/mode mismatch: {path}")
    values = {value for row in rows for value in row}
    if not values <= {0, 255}:
        raise RuntimeError(f"Mask is not binary: {path}")
    return {y * width + x for y, row in enumerate(rows) for x, value in enumerate(row) if value == 255}, rows


def _component_sets(pixels: set[int], width: int, height: int) -> list[set[int]]:
    remaining = set(pixels)
    components = []
    while remaining:
        first = remaining.pop()
        component = {first}
        stack = [first]
        while stack:
            value = stack.pop()
            x, y = value % width, value // width
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not dx and not dy:
                        continue
                    nx, ny = x + dx, y + dy
                    neighbour = ny * width + nx
                    if 0 <= nx < width and 0 <= ny < height and neighbour in remaining:
                        remaining.remove(neighbour)
                        component.add(neighbour)
                        stack.append(neighbour)
        components.append(component)
    return components


def _connected_components(pixels: set[int], width: int, height: int) -> int:
    return len(_component_sets(pixels, width, height))


def _remove_mask_speckles(path: Path, resolution: tuple[int, int], maximum_component_px: int) -> dict:
    """Remove only tiny disconnected AOV quantization islands, never damage geometry."""
    width, height = resolution
    pixels, rows = _mask_pixels(path, resolution)
    components = sorted(_component_sets(pixels, width, height), key=len, reverse=True)
    raw_count = len(components)
    if raw_count <= 1:
        return {"raw_component_count": raw_count, "removed_raster_speckle_px": 0}
    secondary = components[1:]
    if any(len(component) > int(maximum_component_px) for component in secondary):
        return {"raw_component_count": raw_count, "removed_raster_speckle_px": 0}
    removed = set().union(*secondary)
    cleaned_rows = [bytearray(row) for row in rows]
    for value in removed:
        cleaned_rows[value // width][value % width] = 0
    from scripts.blender.render_clean_batch import _encode_png

    _encode_png(path, width, height, 0, [bytes(row) for row in cleaned_rows], [])
    return {"raw_component_count": raw_count, "removed_raster_speckle_px": len(removed)}


def _project_opening_polygon(scene, camera, wheel, placement: dict, resolution: tuple[int, int]) -> dict:
    """Resolve Blender-dependent opening projection on the main thread."""
    width, height = resolution
    polygon = []
    for local in placement["boundary_local_m"]:
        world = wheel.matrix_world @ Matrix.Translation(tuple(map(float, local))).translation
        projected = world_to_camera_view(scene, camera, world)
        if float(projected.z) <= 0.0:
            raise RuntimeError("Through-opening boundary projected behind the camera")
        polygon.append((float(projected.x) * (width - 1), (1.0 - float(projected.y)) * (height - 1)))
    minimum_x = max(0, int(math.floor(min(point[0] for point in polygon))) - 1)
    maximum_x = min(width - 1, int(math.ceil(max(point[0] for point in polygon))) + 1)
    minimum_y = max(0, int(math.floor(min(point[1] for point in polygon))) - 1)
    maximum_y = min(height - 1, int(math.ceil(max(point[1] for point in polygon))) + 1)
    return {
        "polygon": polygon,
        "bbox": [minimum_x, minimum_y, maximum_x, maximum_y],
    }


def _merge_projected_opening_mask_from_projection(
    path: Path, projection: dict, resolution: tuple[int, int]
) -> dict:
    """CPU-only union of the projected cutout footprint with the carrier AOV."""
    width, height = resolution
    _pixels, source_rows = _mask_pixels(path, resolution)
    polygon = [tuple(map(float, point)) for point in projection["polygon"]]
    minimum_x, minimum_y, maximum_x, maximum_y = map(int, projection["bbox"])

    def inside(px: float, py: float) -> bool:
        value = False
        previous = polygon[-1]
        for current in polygon:
            x1, y1 = previous
            x2, y2 = current
            if (y1 > py) != (y2 > py):
                crossing = (x2 - x1) * (py - y1) / (y2 - y1) + x1
                if px < crossing:
                    value = not value
            previous = current
        return value

    opening = {
        y * width + x
        for y in range(minimum_y, maximum_y + 1)
        for x in range(minimum_x, maximum_x + 1)
        if inside(x + 0.5, y + 0.5)
    }
    opening = _dilate(opening, width, height, 1)
    rows = [bytearray(row) for row in source_rows]
    added = 0
    for value in opening:
        y, x = divmod(value, width)
        if rows[y][x] != 255:
            rows[y][x] = 255
            added += 1
    from scripts.blender.render_clean_batch import _encode_png

    _encode_png(path, width, height, 0, [bytes(row) for row in rows], [])
    return {
        "ok": bool(opening),
        "projected_opening_area_px": len(opening),
        "pixels_added_to_carrier_aov": added,
        "edge_overlap_dilation_px": 1,
        "projected_bbox_px": [minimum_x, minimum_y, maximum_x, maximum_y],
    }


def _merge_projected_opening_mask(path: Path, scene, camera, wheel, placement: dict, resolution: tuple[int, int]) -> dict:
    """Compatibility wrapper for synchronous callers."""
    projection = _project_opening_polygon(scene, camera, wheel, placement, resolution)
    return _merge_projected_opening_mask_from_projection(path, projection, resolution)


def _dilate(pixels: set[int], width: int, height: int, radius: int) -> set[int]:
    result = set()
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1) if dx * dx + dy * dy <= radius * radius]
    for value in pixels:
        x, y = value % width, value // width
        for dx, dy in offsets:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                result.add(ny * width + nx)
    return result


def _pair_gates(clean_rgb: Path, anomaly_rgb: Path, target_mask: Path, anomaly_mask: Path, descriptor: dict, config: dict, resolution: tuple[int, int], mask_cleanup: dict | None = None) -> dict:
    import numpy as np

    width, height = resolution
    target, _ = _mask_pixels(target_mask, resolution)
    anomaly, _ = _mask_pixels(anomaly_mask, resolution)
    if not target or len(target) == width * height:
        raise RuntimeError("Target-wheel mask is empty or full")
    if not anomaly or len(anomaly) == width * height:
        raise RuntimeError("Anomaly mask is empty or full")
    xs = [value % width for value in anomaly]
    ys = [value // width for value in anomaly]
    bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
    short_side = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    severity_gate = config["severity"][descriptor["severity"]]
    target_dilated = _dilate(target, width, height, int(config["mask_gates"]["target_roi_dilation_px"]))
    inside_fraction = len(anomaly & target_dilated) / len(anomaly)
    components = _connected_components(anomaly, width, height)

    cw, ch, cc, clean_rows = _decode_png(clean_rgb)
    aw, ah, ac, anomaly_rows = _decode_png(anomaly_rgb)
    if (cw, ch, cc) != (width, height, 3) or (aw, ah, ac) != (width, height, 3):
        raise RuntimeError("RGB shape/mode mismatch")
    threshold = int(config["photometric_gates"]["changed_pixel_threshold_8bit"])
    clean_array = np.frombuffer(b"".join(clean_rows), dtype=np.uint8).reshape(height, width, 3)
    anomaly_array = np.frombuffer(b"".join(anomaly_rows), dtype=np.uint8).reshape(height, width, 3)
    delta_array = np.max(
        np.abs(clean_array.astype(np.int16) - anomaly_array.astype(np.int16)), axis=2
    ).reshape(-1)
    anomaly_indices = np.fromiter(anomaly, dtype=np.int64, count=len(anomaly))
    anomaly_deltas = delta_array[anomaly_indices]
    median_delta = int(np.partition(anomaly_deltas, len(anomaly_deltas) // 2)[len(anomaly_deltas) // 2])
    changed_fraction = float(np.count_nonzero(anomaly_deltas >= threshold) / len(anomaly_deltas))
    total_energy = int(delta_array.sum(dtype=np.int64))
    effect_pixel_count = int(np.count_nonzero(delta_array >= threshold))
    nonzero_effect = total_energy > 0 and effect_pixel_count > 0
    dilation_radius = min(
        int(config["photometric_gates"]["maximum_difference_dilation_px"]),
        max(4, round(0.10 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))),
    )
    local = _dilate(anomaly, width, height, dilation_radius)
    local_indices = np.fromiter(local, dtype=np.int64, count=len(local))
    local_energy = int(delta_array[local_indices].sum(dtype=np.int64))
    energy_fraction = 1.0 if total_energy == 0 else local_energy / total_energy
    effect_fraction = effect_pixel_count / (width * height)
    gates = {
        "binary": True,
        "single_component": components == 1,
        "minimum_area": len(anomaly) >= int(severity_gate["minimum_mask_area_px"]),
        "minimum_short_side": short_side >= int(severity_gate["minimum_mask_short_side_px"]),
        "inside_target_roi": inside_fraction >= float(config["mask_gates"]["minimum_inside_target_roi_fraction"]),
        "changed_fraction": changed_fraction >= float(config["photometric_gates"]["minimum_changed_fraction_inside_mask"]),
        "median_delta": median_delta >= int(config["photometric_gates"]["minimum_median_delta_8bit"]),
        "local_difference_energy": energy_fraction >= float(config["photometric_gates"]["minimum_local_difference_energy_fraction"]),
        "maximum_effect_area": effect_fraction <= float(config["mask_gates"]["maximum_effect_frame_fraction"]),
        "nonzero_effect": nonzero_effect,
    }
    blocking_gate_names = (
        "binary", "single_component", "minimum_area", "minimum_short_side",
        "inside_target_roi", "local_difference_energy", "maximum_effect_area",
        "nonzero_effect",
    )
    photometric_normal = gates["changed_fraction"] and gates["median_delta"]
    if not nonzero_effect:
        photometric_status = "no_effect"
    elif photometric_normal:
        photometric_status = "normal_contrast"
    else:
        photometric_status = "low_contrast"
    diagnostic_policy = config["photometric_gates"].get("failure_policy", "fail_closed") == "diagnostic"
    ok = all(gates[name] for name in blocking_gate_names) and (diagnostic_policy or photometric_normal)
    return {
        "ok": ok,
        "gates": gates,
        "photometric_status": photometric_status,
        "photometric_failure_policy": "diagnostic" if diagnostic_policy else "fail_closed",
        "area_px": len(anomaly),
        "bbox_px": bbox,
        "short_side_px": short_side,
        "component_count": components,
        "raw_component_count": int((mask_cleanup or {}).get("raw_component_count", components)),
        "removed_raster_speckle_px": int((mask_cleanup or {}).get("removed_raster_speckle_px", 0)),
        "inside_target_roi_fraction": inside_fraction,
        "changed_fraction_inside_mask": changed_fraction,
        "median_delta_8bit": median_delta,
        "difference_energy_local_fraction": energy_fraction,
        "difference_dilation_radius_px": dilation_radius,
        "effect_frame_fraction": effect_fraction,
    }


def _finalize_pair_artifacts(
    *,
    clean_module,
    manifest_path: Path,
    run_dir: Path,
    stage: Path,
    clean_combined: Path,
    anomaly_combined: Path,
    clean_rgb: Path,
    anomaly_rgb: Path,
    target_mask: Path,
    anomaly_mask: Path,
    projection: dict,
    descriptor: dict,
    anomaly_config: dict,
    resolution: tuple[int, int],
    relatives: dict,
    row: dict,
    pair_started: float,
    split_combined: bool,
) -> dict:
    """Finalize one pair using only files and pure Python; safe in a worker thread."""
    profile = row["render"]["profiling"]
    if split_combined:
        split_started = time.perf_counter()
        clean_module._split_rgba_png(clean_combined, clean_rgb, target_mask, resolution)
        clean_combined.unlink()
        profile["clean_split_seconds"] = float(time.perf_counter() - split_started)
        split_started = time.perf_counter()
        clean_module._split_rgba_png(anomaly_combined, anomaly_rgb, anomaly_mask, resolution)
        anomaly_combined.unlink()
        profile["anomaly_split_seconds"] = float(time.perf_counter() - split_started)

    mask_started = time.perf_counter()
    opening_mask = _merge_projected_opening_mask_from_projection(anomaly_mask, projection, resolution)
    if opening_mask["ok"] is not True:
        raise RuntimeError(f"Projected through-opening mask failed for {row['pair_id']}")
    mask_cleanup = _remove_mask_speckles(
        anomaly_mask,
        resolution,
        int(anomaly_config["mask_gates"]["maximum_raster_speckle_component_px"]),
    )
    profile["mask_postprocess_seconds"] = float(time.perf_counter() - mask_started)

    inspect_started = time.perf_counter()
    clean_module._inspect_output(clean_rgb, resolution, mask=False)
    clean_module._inspect_output(anomaly_rgb, resolution, mask=False)
    clean_module._inspect_output(target_mask, resolution, mask=True)
    clean_module._inspect_output(anomaly_mask, resolution, mask=True)
    profile["inspect_seconds"] = float(time.perf_counter() - inspect_started)

    gates_started = time.perf_counter()
    pair_gate = _pair_gates(
        clean_rgb,
        anomaly_rgb,
        target_mask,
        anomaly_mask,
        descriptor,
        anomaly_config,
        resolution,
        mask_cleanup,
    )
    if pair_gate["ok"] is not True:
        raise RuntimeError(f"Post-render gates failed for {row['pair_id']}: {pair_gate}")
    profile["pair_gates_seconds"] = float(time.perf_counter() - gates_started)
    row["gates"]["mask_and_photometric"] = pair_gate
    row["gates"]["opening_mask_details"] = opening_mask

    staged = {
        "clean_rgb": clean_rgb,
        "anomaly_rgb": anomaly_rgb,
        "target_wheel_mask": target_mask,
        "anomaly_mask": anomaly_mask,
    }
    artifacts = {}
    hash_started = time.perf_counter()
    for key, relative in relatives.items():
        final = run_dir / relative
        final.parent.mkdir(parents=True, exist_ok=True)
        artifacts[key] = {"path": relative.as_posix(), "sha256": clean_module._sha256(staged[key])}
    profile["hash_seconds"] = float(time.perf_counter() - hash_started)
    row["artifacts"] = artifacts

    write_started = time.perf_counter()
    for key, relative in relatives.items():
        os.replace(staged[key], run_dir / relative)
    shutil.rmtree(stage)
    row["render"]["write_seconds"] = float(time.perf_counter() - write_started)
    profile["commit_seconds"] = row["render"]["write_seconds"]
    row["render"]["total_seconds"] = float(time.perf_counter() - pair_started)
    clean_module._append_manifest(manifest_path, row)
    return row


def render_chunk(batch_path, domain_path, anomaly_path, chunk_path, lighting_path, wear_path, pose_path, sampling_path, run_dir, manifest_path, report_path):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.static_geometry_cache import StaticGeometryCache
    from scripts.blender import render_clean_batch as clean
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender import render_wheel_surface_wear_pilot as wear
    from scripts.blender import wheel_hole_anomaly as hole
    from scripts.blender.render_domain_randomization_preview import _align_rover_to_patch, _framing_gate, _jitter_camera, _settle_rover_on_patch, _terrain_footprint
    from scripts.blender.render_wheel_camera_pose_pilot import _apply_terrain_color_grade
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose, restore_base_roll, select_visible_anomaly_roll
    from src.wheel_preparation.anomaly_batch import assert_pair_retry_preserves_semantics, resolved_pair_lock_sha256
    from src.wheel_preparation.domain_randomization import sample_domain_randomization

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    anomaly_config = json.loads(anomaly_path.read_text(encoding="utf-8"))
    chunk = json.loads(chunk_path.read_text(encoding="utf-8"))
    lighting_config = json.loads(lighting_path.read_text(encoding="utf-8"))
    wear_config = json.loads(wear_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_path.read_text(encoding="utf-8"))
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    lighting_by_id = {entry["id"]: entry for entry in lighting_config["presets"]}
    wear_by_id = {entry["id"]: entry for entry in wear_config["variants"]}
    pose_by_id = {entry["id"]: entry for entry in pose_config["poses"]}

    scene = bpy.context.scene
    if not bool(scene.get("wheel_pose_sampling_ready", False)):
        raise RuntimeError("Anomaly source is not pose-sampling ready")
    if tuple(bpy.app.version[:2]) != tuple(map(int, batch["render"]["blender_major_minor"])):
        raise RuntimeError(f"Blender version mismatch: {bpy.app.version_string}")
    scene.render.engine = str(batch["render"]["engine"])
    resolution = tuple(map(int, batch["render"]["resolution"]))
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.eevee.taa_render_samples = int(batch["render"]["samples"])
    scene.frame_set(1)

    setup_started = time.perf_counter()
    wheels = [bpy.data.objects[name] for name in sampling["wheels"]]
    root = bpy.data.objects[domain["gates"]["terrain_alignment"]["rover_root_object"]]
    patch = bpy.data.objects[domain["gates"]["terrain_patch_object"]]
    base_root = root.matrix_world.copy()
    base_local = {wheel.name: clean._matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
    wheel_materials, wheel_geometry, dimensions_by_wheel = {}, {}, {}
    for wheel in wheels:
        meshes = wear._target_meshes(wheel)
        wheel_geometry[wheel.name] = wear._geometry_signature(meshes)
        dimensions_by_wheel[wheel.name] = wear._wheel_dimensions(wheel, meshes)
        materials, _affected = wear._clone_target_materials(meshes, f"anomaly_batch_{wheel.name}")
        wheel_materials[wheel.name] = materials
    graded_materials = _apply_terrain_color_grade(lighting_config["terrain_grade"])
    dusty_preset = lighting_by_id["mars_dusty_refined"]
    dusty_names = lighting._apply_dust_layer(dusty_preset["dust"])
    dusty_materials = [bpy.data.materials[name] for name in dusty_names]
    wear_image = bpy.data.images.load(str(Path(chunk["placeholder_wear_mask"]).resolve()), check_existing=False)
    wear_image.name = clean.WEAR_IMAGE_NAME
    wear_image.colorspace_settings.name = "Non-Color"
    for wheel in wheels:
        clean._install_wear(wheel_materials[wheel.name], wheel, dimensions_by_wheel[wheel.name], wear_image, wear_config["shader"])
        clean._install_aov(wheel_materials[wheel.name])
    hole_runtime = hole.setup_runtime(wheel_materials, Path(chunk["placeholder_hole_mask"]).resolve())
    camera = lighting._configure_camera(pose_config["camera"])
    scene.camera = camera
    compositor = clean._setup_compositor(scene)
    benchmark = batch.get("benchmark", {})
    clean.PNG_COMPRESSION_LEVEL = int(benchmark.get("png_compression_level", 4))
    execution = batch["execution"]
    geometry_cache_enabled = bool(benchmark.get("geometry_cache", execution.get("geometry_cache", False)))
    geometry_cache = StaticGeometryCache(wheels, patch, domain["gates"]["terrain_alignment"]["vertical_contact"]) if geometry_cache_enabled else None
    hole.disable(hole_runtime)
    bpy.context.view_layer.update()
    setup_counts = clean._datablock_counts()
    setup_seconds = time.perf_counter() - setup_started
    runtime = {"blender_version": bpy.app.version_string, "engine": scene.render.engine, "device": None}
    completed = {row["pair_id"] for row in clean._read_jsonl(manifest_path)}
    rendered, render_calls = [], 0
    async_enabled = bool(benchmark.get("async_postprocess", execution.get("async_postprocess", False)))
    async_depth = int(benchmark.get("async_queue_depth", execution.get("async_queue_depth", 4)))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anomaly-png") if async_enabled else None
    pending = deque()

    def collect_one():
        future = pending.popleft()
        rendered.append(future.result())

    for planned in chunk["pairs"]:
        pair_id = str(planned["pair_id"])
        if pair_id in completed:
            continue
        pair_started = time.perf_counter()
        profile = {}
        hole.disable(hole_runtime)
        restore_base_roll(wheels, base_local)
        root.matrix_world = base_root.copy()
        bpy.context.view_layer.update()
        planned_domain = planned["domain_sample"]
        descriptor = planned["anomaly"]
        target = bpy.data.objects[planned_domain["target_wheel"]]
        translation = _align_rover_to_patch(scene, target, domain["gates"]["terrain_alignment"], base_root)
        placements = hole.resolve_placements(target, descriptor, anomaly_config)
        if not placements:
            raise RuntimeError(f"Placement exhausted for {pair_id}")
        pose = pose_by_id[planned_domain["camera_pose"]]
        rejections = []
        accepted = None
        maximum_attempts = int(domain["gates"]["maximum_camera_resample_attempts"])
        for attempt in range(maximum_attempts):
            candidate = sample_domain_randomization(domain, int(planned["sample_index"]), attempt=attempt)
            assert_pair_retry_preserves_semantics(planned_domain, candidate)
            camera.data.lens = float(pose_config["camera"]["focal_length_mm"]) * float(candidate["camera_jitter"]["focal_length_scale"])
            basis = camera_pose(camera, target, pose, pose_config["camera"])
            aim_target = _jitter_camera(camera, basis, candidate)
            for placement in placements:
                sector = hole.sector_target_offset_degrees(descriptor, anomaly_config, float(placement["wheel_radius_m"]))
                roll = select_visible_anomaly_roll(
                    scene, camera, target, wheels, base_local, pose, pose_config["camera"],
                    sampling["anomaly_sampling"], placement["probes_local"],
                    hole.positive_roll_effect_sign(target, pose_config["camera"]),
                    target_jitter_degrees=float(sector["target_offset_degrees"]),
                    carrier_objects=[target], camera_basis=basis,
                )
                if roll.get("ok") is not True:
                    rejections.append({"attempt": attempt, "camera_attempt_seed": candidate["camera_attempt_seed"], "placement_candidate_attempt": placement["candidate_attempt"], "reason": "anomaly_visibility"})
                    continue
                apply_shared_roll(wheels, base_local, float(roll["selected"]["roll_degrees"]))
                contact = geometry_cache.settle(target, patch, domain["gates"]["terrain_alignment"]) if geometry_cache else _settle_rover_on_patch(scene, target, patch, domain["gates"]["terrain_alignment"])
                translation = root.matrix_world.translation - base_root.translation
                basis = camera_pose(camera, target, pose, pose_config["camera"])
                aim_target = _jitter_camera(camera, basis, candidate)
                footprint = geometry_cache.footprint(scene, camera, float(domain["gates"]["terrain_minimum_edge_margin_m"])) if geometry_cache else _terrain_footprint(scene, camera, patch, float(domain["gates"]["terrain_minimum_edge_margin_m"]))
                framing = geometry_cache.framing(scene, camera, target, basis, pose, domain["gates"]) if geometry_cache else _framing_gate(scene, camera, target, basis, pose, domain["gates"])
                if contact["ok"] and footprint["ok"] and framing["ok"]:
                    accepted = (candidate, basis, aim_target, roll, footprint, framing, contact, placement, sector, translation)
                    break
                rejections.append({"attempt": attempt, "camera_attempt_seed": candidate["camera_attempt_seed"], "placement_candidate_attempt": placement["candidate_attempt"], "reason": "camera_or_terrain"})
                restore_base_roll(wheels, base_local)
            if accepted is not None:
                break
        if accepted is None:
            raise RuntimeError(f"Camera/visibility retries exhausted for {pair_id}")
        sample, basis, aim_target, roll, footprint, framing, contact, placement, sector, translation = accepted

        preset = clean._jittered_preset(lighting_by_id[sample["lighting_preset"]], sample)
        lighting._configure_sun_world(preset)
        scene.view_settings.look = preset["look"]
        scene.view_settings.exposure = float(preset["exposure_ev"])
        clean._set_post(compositor, preset)
        clean._set_dust(dusty_materials, preset, dusty_preset["dust"])
        wear_variant = wear_by_id[sample["surface_wear"]]
        wear_enabled = bool(wear_variant["wear_enabled"])
        if wear_enabled:
            wear_image.filepath_raw = str(Path(planned["wear_mask"]).resolve())
            wear_image.source = "FILE"
            wear_image.reload()
        clean._set_wear(wheel_materials, target.name, wear_enabled, float(wear_config["shader"]["opacity"]))
        clean._set_aov_target(wheel_materials, target.name)
        setup_pair_seconds = time.perf_counter() - pair_started
        profile["resolve_and_setup_seconds"] = float(setup_pair_seconds)

        stage = run_dir / "_staging" / pair_id
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.compression = int(benchmark.get("scene_png_compression", 15))

        hole.disable(hole_runtime)
        hole.set_compositor_mask_aov(compositor, clean.AOV_NAME)
        clean_combined = stage / "clean_combined.png"
        clean_rgb, target_mask = stage / "clean_rgb.png", stage / "target_mask.png"
        scene.render.filepath = str(clean_combined)
        clean_started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        render_calls += 1
        clean_seconds = time.perf_counter() - clean_started
        profile["clean_render_seconds"] = float(clean_seconds)
        hole_started = time.perf_counter()
        resolved_hole_material = hole.enable(
            hole_runtime,
            target,
            placement,
            descriptor,
            anomaly_config,
            Path(planned["hole_mask"]).resolve(),
            tuple(map(float, planned["hole_mapping_extent_m"])),
        )
        through_opening = resolved_hole_material["through_opening_gate"]
        if through_opening["ok"] is not True:
            raise RuntimeError(f"Synthetic carrier blocks through opening for {pair_id}: {through_opening}")
        profile["hole_enable_seconds"] = float(time.perf_counter() - hole_started)
        hole.set_compositor_mask_aov(compositor, hole.ANOMALY_AOV_NAME)
        anomaly_combined = stage / "anomaly_combined.png"
        anomaly_rgb, anomaly_mask = stage / "anomaly_rgb.png", stage / "anomaly_mask.png"
        scene.render.filepath = str(anomaly_combined)
        anomaly_started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        render_calls += 1
        anomaly_seconds = time.perf_counter() - anomaly_started
        profile["anomaly_render_seconds"] = float(anomaly_seconds)
        projection = _project_opening_polygon(scene, camera, target, placement, resolution)
        if runtime["device"] is None:
            runtime["device"] = {
                "vendor": gpu.platform.vendor_get(),
                "renderer": gpu.platform.renderer_get(),
                "version": gpu.platform.version_get(),
                "backend": gpu.platform.backend_type_get(),
                "type": gpu.platform.device_type_get(),
            }

        relatives = {
            "clean_rgb": Path("rgb/clean") / f"{pair_id}.png",
            "anomaly_rgb": Path("rgb/anomaly") / f"{pair_id}.png",
            "target_wheel_mask": Path("target_wheel_mask") / f"{pair_id}.png",
            "anomaly_mask": Path("anomaly_mask") / f"{pair_id}.png",
        }
        sun = bpy.data.objects["GaleNominalSun"]
        background = scene.world.node_tree.nodes["Background"]
        shared = {
            "pair_id": pair_id,
            "pair_lock_id": planned["pair_lock_id"],
            "sample_index": int(planned["sample_index"]),
            "camera_attempt": int(sample["attempt"]),
            "camera_attempt_seed": int(sample["camera_attempt_seed"]),
            "target_wheel": target.name,
            "camera_pose": sample["camera_pose"],
            "lighting_preset": sample["lighting_preset"],
            "surface_wear": sample["surface_wear"],
            "wear_mask_sha256": planned.get("wear_mask_sha256"),
            "camera_matrix_world": clean._matrix_values(camera.matrix_world),
            "wheel_matrix_world": clean._matrix_values(target.matrix_world),
            "rover_matrix_world": clean._matrix_values(root.matrix_world),
            "roll_degrees": float(roll["selected"]["roll_degrees"]),
        }
        row = {
            "schema_version": 1,
            "pair_id": pair_id,
            "pair_index": int(planned["pair_index"]),
            "sample_index": int(planned["sample_index"]),
            "condition": "paired_clean_hole",
            "members": planned["members"],
            "pair_lock_id": planned["pair_lock_id"],
            "resolved_pair_lock_sha256": resolved_pair_lock_sha256(shared),
            "seeds": {"sample": int(sample["sample_seed"]), "camera_attempt": int(sample["camera_attempt_seed"]), "wear": int(sample["wear_seed"]), **descriptor["seeds"]},
            "sampling": {
                "target_wheel": sample["target_wheel"], "camera_pose": sample["camera_pose"],
                "healthy_roll_degrees": float(sample["healthy_roll_degrees"]), "lighting_preset": sample["lighting_preset"],
                "surface_wear": sample["surface_wear"], "lighting_jitter": sample["lighting_jitter"], "camera_jitter": sample["camera_jitter"],
            },
            "anomaly": descriptor,
            "resolved": {
                **shared,
                "camera_location_m": list(map(float, camera.location)), "aim_target_m": list(map(float, aim_target)),
                "focal_length_mm": float(camera.data.lens), "rover_patch_alignment_translation_m": list(map(float, translation)),
                "placement": placement, "sector_target": sector, "visibility_roll": roll,
                "hole_material": resolved_hole_material,
                "lighting": {"sun_energy": float(sun.data.energy), "sun_angle_degrees": math.degrees(float(sun.data.angle)), "sun_color": list(map(float, sun.data.color)), "world_strength": float(background.inputs["Strength"].default_value), "exposure_ev": float(scene.view_settings.exposure)},
            },
            "geometry": wheel_geometry[target.name],
            "gates": {"framing": True, "terrain_coverage": True, "terrain_contact": True, "anomaly_visibility": True, "through_opening": True, "mask_and_photometric": None, "framing_details": framing, "terrain_details": footprint, "terrain_contact_details": contact, "through_opening_details": through_opening, "opening_mask_details": None, "camera_rejections": rejections},
            "artifacts": {},
            "render": {"engine": scene.render.engine, "blender_version": bpy.app.version_string, "resolution": list(resolution), "samples": int(scene.eevee.taa_render_samples), "device": runtime["device"], "setup_seconds": float(setup_pair_seconds), "clean_seconds": float(clean_seconds), "anomaly_seconds": float(anomaly_seconds), "profiling": profile, "total_seconds": float(time.perf_counter() - pair_started)},
        }
        finalize_kwargs = {
            "clean_module": clean,
            "manifest_path": manifest_path,
            "run_dir": run_dir,
            "stage": stage,
            "clean_combined": clean_combined,
            "anomaly_combined": anomaly_combined,
            "clean_rgb": clean_rgb,
            "anomaly_rgb": anomaly_rgb,
            "target_mask": target_mask,
            "anomaly_mask": anomaly_mask,
            "projection": projection,
            "descriptor": descriptor,
            "anomaly_config": anomaly_config,
            "resolution": resolution,
            "relatives": relatives,
            "row": row,
            "pair_started": pair_started,
            "split_combined": True,
        }
        if executor is None:
            rendered.append(_finalize_pair_artifacts(**finalize_kwargs))
        else:
            pending.append(executor.submit(_finalize_pair_artifacts, **finalize_kwargs))
            if len(pending) >= async_depth:
                collect_one()
        if clean._datablock_counts() != setup_counts:
            raise RuntimeError(f"Blender datablock drift after {pair_id}")
        print(f"ANOMALY_PAIR {pair_id} clean={clean_seconds:.3f}s anomaly={anomaly_seconds:.3f}s", flush=True)

    while pending:
        collect_one()
    if executor is not None:
        executor.shutdown(wait=True)

    report = {
        "schema_version": 1, "ok": True, "source_open_count": 1,
        "planned_pair_count": len(chunk["pairs"]), "rendered_pair_count": len(rendered),
        "render_call_count": render_calls, "runtime": runtime, "setup_seconds": float(setup_seconds),
        "setup_datablocks": setup_counts, "graded_materials": graded_materials, "dusty_materials": dusty_names,
        "anomaly_runtime": {"installed": hole_runtime["installed"], "carrier": hole_runtime["carrier"].name, "frame": hole_runtime["frame"].name},
        "benchmark": {"geometry_cache": geometry_cache is not None, "async_postprocess": async_enabled, "async_queue_depth": async_depth, "png_compression_level": int(clean.PNG_COMPRESSION_LEVEL), "scene_png_compression": int(benchmark.get("scene_png_compression", 15))},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    render_chunk(*_arguments())
