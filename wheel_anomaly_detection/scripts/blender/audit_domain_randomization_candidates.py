"""Preflight camera/terrain gates for deterministic domain-randomization candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return tuple(value.resolve() for value in (args.config, args.plan, args.pose_config, args.sampling_config, args.output))


def audit(config_path: Path, plan_path: Path, pose_config_path: Path, sampling_config_path: Path, output_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender.render_domain_randomization_preview import (
        _align_rover_to_patch,
        _framing_gate,
        _jitter_camera,
        _matrix,
        _terrain_footprint,
    )
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose

    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    pose_config = json.loads(pose_config_path.read_text(encoding="utf-8"))
    sampling_config = json.loads(sampling_config_path.read_text(encoding="utf-8"))
    pose_by_id = {entry["id"]: entry for entry in pose_config["poses"]}
    scene = bpy.context.scene
    resolution = tuple(map(int, config["preview"]["resolution"]))
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    wheels = [bpy.data.objects[name] for name in sampling_config["wheels"]]
    base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
    root = bpy.data.objects[config["gates"]["terrain_alignment"]["rover_root_object"]]
    base_root_matrix = root.matrix_world.copy()
    patch = bpy.data.objects.get(config["gates"]["terrain_patch_object"])
    if patch is None or patch.type != "MESH":
        raise RuntimeError("Microterrain patch required by the jitter gate is missing")
    results = []
    for sample in plan["samples"]:
        target = bpy.data.objects[sample["target_wheel"]]
        rover_translation = _align_rover_to_patch(
            scene, target, config["gates"]["terrain_alignment"], base_root_matrix
        )
        apply_shared_roll(wheels, base_local, float(sample["healthy_roll_degrees"]))
        pose = pose_by_id[sample["camera_pose"]]
        camera = lighting._configure_camera(pose_config["camera"])
        camera.data.lens = float(pose_config["camera"]["focal_length_mm"]) * float(sample["camera_jitter"]["focal_length_scale"])
        basis = camera_pose(camera, target, pose, pose_config["camera"])
        _jitter_camera(camera, basis, sample)
        scene.camera = camera
        footprint = _terrain_footprint(scene, camera, patch, float(config["gates"]["terrain_minimum_edge_margin_m"]))
        framing = _framing_gate(scene, camera, target, basis, pose, config["gates"])
        results.append(
            {
                "sample_index": int(sample["sample_index"]),
                "ok": bool(footprint["ok"] and framing["ok"]),
                "terrain_ok": bool(footprint["ok"]),
                "framing_ok": bool(framing["ok"]),
                "terrain_minimum_edge_margin_m": footprint.get("minimum_edge_margin_m"),
                "framing_bbox": framing["normalized_bbox"],
                "framing_bbox_area_fraction": framing["bbox_area_fraction"],
                "rover_patch_alignment_translation_m": list(map(float, rover_translation)),
            }
        )
    report = {
        "schema_version": 1,
        "candidate_count": len(results),
        "valid_count": sum(int(entry["ok"]) for entry in results),
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("candidate_count", "valid_count")}, indent=2))
    return report


if __name__ == "__main__":
    audit(*_arguments())
