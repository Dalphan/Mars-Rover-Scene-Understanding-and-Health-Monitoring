"""Audit every wheel/camera family against the fixed 4x4 m patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-config", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, required=True)
    parser.add_argument("--sampling-config", type=Path, required=True)
    parser.add_argument("--resolution", nargs=2, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args


def _matrix(values) -> Matrix:
    values = list(map(float, values))
    return Matrix([values[index : index + 4] for index in range(0, 16, 4)])


def audit(args) -> dict:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender import render_mars_lighting_pilot as lighting
    from scripts.blender.render_domain_randomization_preview import (
        _align_rover_to_patch,
        _framing_gate,
        _jitter_camera,
        _terrain_footprint,
    )
    from scripts.blender.wheel_pose_sampling import apply_shared_roll, camera_pose, restore_base_roll
    from src.wheel_preparation.domain_randomization import sample_domain_randomization

    domain_config = json.loads(args.domain_config.resolve().read_text(encoding="utf-8"))
    pose_config = json.loads(args.pose_config.resolve().read_text(encoding="utf-8"))
    sampling_config = json.loads(args.sampling_config.resolve().read_text(encoding="utf-8"))
    pose_by_id = {entry["id"]: entry for entry in pose_config["poses"]}
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = map(int, args.resolution)
    scene.render.resolution_percentage = 100
    wheels = [bpy.data.objects[name] for name in sampling_config["wheels"]]
    base_local = {wheel.name: _matrix(wheel["pose_sampling_base_matrix_local"]) for wheel in wheels}
    root = bpy.data.objects[domain_config["gates"]["terrain_alignment"]["rover_root_object"]]
    base_root_matrix = root.matrix_world.copy()
    patch = bpy.data.objects[domain_config["gates"]["terrain_patch_object"]]
    camera = lighting._configure_camera(pose_config["camera"])
    results = []
    combo_index = 0
    for wheel in wheels:
        for pose_id, pose in pose_by_id.items():
            restore_base_roll(wheels, base_local)
            root.matrix_world = base_root_matrix.copy()
            bpy.context.view_layer.update()
            translation = _align_rover_to_patch(
                scene, wheel, domain_config["gates"]["terrain_alignment"], base_root_matrix
            )
            apply_shared_roll(wheels, base_local, 0.0)
            rejections = []
            accepted = None
            for attempt in range(int(domain_config["gates"]["maximum_camera_resample_attempts"])):
                candidate = sample_domain_randomization(domain_config, 1000000 + combo_index, attempt=attempt)
                camera.data.lens = float(pose_config["camera"]["focal_length_mm"]) * float(candidate["camera_jitter"]["focal_length_scale"])
                basis = camera_pose(camera, wheel, pose, pose_config["camera"])
                _jitter_camera(camera, basis, candidate)
                scene.camera = camera
                footprint = _terrain_footprint(scene, camera, patch, float(domain_config["gates"]["terrain_minimum_edge_margin_m"]))
                framing = _framing_gate(scene, camera, wheel, basis, pose, domain_config["gates"])
                if footprint["ok"] and framing["ok"]:
                    accepted = {
                        "attempt": attempt,
                        "camera_attempt_seed": candidate["camera_attempt_seed"],
                        "terrain_minimum_edge_margin_m": footprint["minimum_edge_margin_m"],
                        "framing_bbox_area_fraction": framing["bbox_area_fraction"],
                    }
                    break
                rejections.append({"attempt": attempt, "terrain": footprint.get("reason"), "framing_ok": framing["ok"]})
            results.append(
                {
                    "target_wheel": wheel.name,
                    "camera_pose": pose_id,
                    "ok": accepted is not None,
                    "accepted": accepted,
                    "rejection_count": len(rejections),
                    "rover_patch_alignment_translation_m": list(map(float, translation)),
                }
            )
            combo_index += 1
    report = {
        "schema_version": 1,
        "ok": len(results) == 24 and all(entry["ok"] for entry in results),
        "combination_count": len(results),
        "accepted_count": sum(int(entry["ok"]) for entry in results),
        "alignment": domain_config["gates"]["terrain_alignment"],
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("ok", "combination_count", "accepted_count")}, indent=2))
    if not report["ok"]:
        raise RuntimeError("Clean-batch wheel/camera matrix audit failed")
    return report


if __name__ == "__main__":
    audit(_arguments())
