"""Fail-closed geometric audit for the paired wheel-hole anomaly contract."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-config", type=Path, required=True)
    parser.add_argument("--anomaly-config", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return (
        args.domain_config.resolve(),
        args.anomaly_config.resolve(),
        args.pose_config.resolve(),
        args.sampling_config.resolve(),
        tuple(args.resolution),
        args.output.resolve(),
    )


def _matrix(values) -> Matrix:
    flat = list(map(float, values))
    return Matrix([flat[index : index + 4] for index in range(0, 16, 4)])


def audit(domain_path, anomaly_path, pose_path, sampling_path, resolution, output_path):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender.render_domain_randomization_preview import (
        _align_rover_to_patch,
        _framing_gate,
        _settle_rover_on_patch,
        _terrain_footprint,
    )
    from scripts.blender.wheel_hole_anomaly import (
        positive_roll_effect_sign,
        resolve_placements,
        sector_target_offset_degrees,
    )
    from scripts.blender.wheel_pose_sampling import (
        apply_shared_roll,
        camera_pose,
        restore_base_roll,
        select_visible_anomaly_roll,
    )
    from src.wheel_preparation.hole_anomaly import sample_hole_descriptor

    started = time.perf_counter()
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_path.read_text(encoding="utf-8"))
    sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = map(int, resolution)
    scene.render.resolution_percentage = 100
    scene.frame_set(1)
    wheels = [bpy.data.objects[name] for name in sampling["wheels"]]
    base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
    root = bpy.data.objects[domain["gates"]["terrain_alignment"]["rover_root_object"]]
    patch = bpy.data.objects[domain["gates"]["terrain_patch_object"]]
    base_root = root.matrix_world.copy()
    camera = lighting._configure_camera(pose_config["camera"])
    scene.camera = camera
    pose_by_id = {entry["id"]: entry for entry in pose_config["poses"]}
    strata = (
        ("tread", "small"),
        ("tread", "medium"),
        ("tread", "large"),
        ("shoulder", "small"),
        ("shoulder", "medium"),
    )
    sectors = ("leading", "upper", "trailing")
    records = []
    failures = []
    synthetic_index = 0
    for wheel in wheels:
        for pose_id, pose in pose_by_id.items():
            for surface, severity in strata:
                for sector in sectors:
                    restore_base_roll(wheels, base_local)
                    root.matrix_world = base_root.copy()
                    bpy.context.view_layer.update()
                    translation = _align_rover_to_patch(
                        scene, wheel, domain["gates"]["terrain_alignment"], base_root
                    )
                    descriptor = sample_hole_descriptor(
                        anomaly,
                        master_seed=int(domain["master_seed"]),
                        sample_index=1_000_000 + synthetic_index,
                        surface_wear="surface_current",
                        severity=severity,
                        surface=surface,
                        image_sector=sector,
                    )
                    synthetic_index += 1
                    placements = resolve_placements(wheel, descriptor, anomaly)
                    record = {
                        "wheel": wheel.name,
                        "pose": pose_id,
                        "surface": surface,
                        "severity": severity,
                        "image_sector": sector,
                        "valid_source_placement_count": len(placements),
                    }
                    if not placements:
                        record["ok"] = False
                        record["reason"] = "placement_exhausted"
                        failures.append(record)
                        records.append(record)
                        continue
                    selected = None
                    placement_attempts = []
                    for placement in placements:
                        camera.data.lens = float(pose_config["camera"]["focal_length_mm"])
                        basis = camera_pose(camera, wheel, pose, pose_config["camera"])
                        sector_result = sector_target_offset_degrees(
                            descriptor, anomaly, float(placement["wheel_radius_m"])
                        )
                        roll = select_visible_anomaly_roll(
                            scene,
                            camera,
                            wheel,
                            wheels,
                            base_local,
                            pose,
                            pose_config["camera"],
                            sampling["anomaly_sampling"],
                            placement["probes_local"],
                            positive_roll_effect_sign(wheel, pose_config["camera"]),
                            target_jitter_degrees=float(sector_result["target_offset_degrees"]),
                            carrier_objects=[wheel],
                            camera_basis=basis,
                        )
                        placement_attempts.append({
                            "candidate_attempt": placement["candidate_attempt"],
                            "roll_ok": bool(roll.get("ok")),
                        })
                        if roll.get("ok") is True:
                            selected = (placement, basis, sector_result, roll)
                            break
                    record["placement_attempts"] = placement_attempts
                    if selected is None:
                        record["ok"] = False
                        record["reason"] = "visibility_roll_exhausted"
                        failures.append(record)
                        records.append(record)
                        continue
                    placement, basis, sector_result, roll = selected
                    record["placement"] = placement
                    record["sector_target"] = sector_result
                    record["roll"] = roll
                    apply_shared_roll(wheels, base_local, float(roll["selected"]["roll_degrees"]))
                    contact = _settle_rover_on_patch(
                        scene, wheel, patch, domain["gates"]["terrain_alignment"]
                    )
                    translation = root.matrix_world.translation - base_root.translation
                    basis = camera_pose(camera, wheel, pose, pose_config["camera"])
                    footprint = _terrain_footprint(
                        scene, camera, patch, float(domain["gates"]["terrain_minimum_edge_margin_m"])
                    )
                    framing = _framing_gate(scene, camera, wheel, basis, pose, domain["gates"])
                    record.update({
                        "ok": bool(contact["ok"] and footprint["ok"] and framing["ok"]),
                        "reason": None if contact["ok"] and footprint["ok"] and framing["ok"] else "camera_or_terrain_gate",
                        "terrain": footprint,
                        "terrain_contact": contact,
                        "framing": framing,
                        "rover_patch_alignment_translation_m": list(map(float, translation)),
                    })
                    if not record["ok"]:
                        failures.append(record)
                    records.append(record)
    restore_base_roll(wheels, base_local)
    root.matrix_world = base_root
    report = {
        "schema_version": 1,
        "ok": not failures and len(records) == 360,
        "source_open_count": 1,
        "expected_count": 360,
        "tested_count": len(records),
        "passed_count": sum(bool(record.get("ok")) for record in records),
        "failed_count": len(failures),
        "seconds": float(time.perf_counter() - started),
        "failures": failures,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"records", "failures"}}, indent=2))
    if not report["ok"]:
        raise RuntimeError(f"Anomaly matrix audit failed: {len(failures)}/360")
    return report


if __name__ == "__main__":
    audit(*_arguments())
