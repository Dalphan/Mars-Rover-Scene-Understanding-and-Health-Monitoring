from __future__ import annotations

import argparse
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Validate a reopened milestone-1 simulator blend.")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(values)


def mesh_digest(obj: bpy.types.Object) -> str:
    digest = hashlib.sha256()
    mesh = obj.data
    digest.update(f"{len(mesh.vertices)}:{len(mesh.edges)}:{len(mesh.polygons)}".encode())
    for vertex in mesh.vertices:
        digest.update(("%.8f,%.8f,%.8f;" % tuple(vertex.co)).encode())
    for polygon in mesh.polygons:
        digest.update((",".join(str(int(v)) for v in polygon.vertices) + ";").encode())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    checks: dict[str, Any] = {
        "blender_version": bpy.app.version_string,
        "scene_names": sorted(scene.name for scene in bpy.data.scenes),
    }
    simulator = bpy.data.scenes.get("Simulator")
    normal = bpy.data.scenes.get("Normal")
    perforation = bpy.data.scenes.get("Perforation")
    root = bpy.data.objects.get("RoverRoot")
    if simulator is None or normal is None or perforation is None:
        errors.append("Missing Simulator, Normal, or Perforation scene")
    if root is None or root.type != "EMPTY":
        errors.append("Missing RoverRoot empty")

    required = {
        "Rover_Without_Selected_Wheel", "Wheel_Attachment", "Wheel_Details_Normal",
        "Wheel_Skin_Normal", "Wheel_Details_Perforation", "Wheel_Skin_Anomaly",
        "WheelCamera",
    }
    descendants: set[bpy.types.Object] = set()
    if root is not None:
        if simulator is not None and bpy.context.window is not None:
            bpy.context.window.scene = simulator
        pending = list(root.children)
        while pending:
            current = pending.pop()
            if current in descendants:
                continue
            descendants.add(current)
            pending.extend(current.children)
        names = {obj.name for obj in descendants}
        checks["root_descendants"] = sorted(names)
        missing = required - names
        if missing:
            errors.append("RoverRoot children missing: " + ", ".join(sorted(missing)))
        if bpy.data.objects.get("Wheel_Attachment") not in descendants:
            errors.append("Wheel_Attachment is not parented to RoverRoot")
        probes = [obj for obj in descendants if obj.get("role") == "ground_probe"]
        checks["probe_count"] = len(probes)
        if len(probes) != 4:
            errors.append(f"Expected four static probes, found {len(probes)}")
        footprint = bpy.data.objects.get("GroundProbe_Footprint")
        if footprint is None or footprint not in descendants:
            errors.append("Ground probe footprint is missing from RoverRoot")
        identity = all(
            abs(float(root.matrix_world[row][column]) - (1.0 if row == column else 0.0)) <= 1e-6
            for row in range(4) for column in range(4)
        )
        checks["root_identity"] = identity
        preview_pose = json.loads(str(root.get("visual_terrain_pose", "{}")))
        preview_pose_valid = (
            simulator is not None
            and simulator.get("simulator_milestone") == "1-terrain-preview"
            and preview_pose
            and abs(float(root.location.x) - float(preview_pose["x"])) <= 1e-6
            and abs(float(root.location.y) - float(preview_pose["y"])) <= 1e-6
            and abs(float(root.location.z) - float(preview_pose["z"])) <= 1e-6
            and abs(float(root.rotation_euler.z) - math.radians(float(preview_pose["yaw_degrees"]))) <= 1e-6
        )
        checks["approved_visual_pose"] = preview_pose
        checks["approved_visual_pose_valid"] = bool(preview_pose_valid)
        if not identity and not preview_pose_valid:
            errors.append("RoverRoot is neither identity nor at the approved visual terrain pose")

        movable = [
            bpy.data.objects[name]
            for name in required
            if name in bpy.data.objects and simulator is not None and name in simulator.objects
        ]
        before = {obj.name: obj.matrix_world.translation.copy() for obj in movable}
        root_matrix = root.matrix_world.copy()
        delta = (0.125, -0.25, 0.05)
        root.location.x += delta[0]
        root.location.y += delta[1]
        root.location.z += delta[2]
        bpy.context.view_layer.update()
        moves_all = all(
            all(abs(float(obj.matrix_world.translation[i]) - float(before[obj.name][i]) - delta[i]) <= 1e-6
                for i in range(3)) for obj in movable
        )
        root.matrix_world = root_matrix
        bpy.context.view_layer.update()
        checks["root_translation_moves_all_rover_children"] = moves_all
        if not moves_all:
            errors.append("RoverRoot translation does not move every rover child")

    forbidden = [
        obj.name for obj in bpy.data.objects
        if obj.name in {"MarsMosaicSurface", "TerrainCollider"}
        or str(obj.get("role", "")).startswith("mars_terrain_mosaic")
        or obj.get("role") in {"terrain_collision_mesh", "terrain_contact_normal"}
    ]
    checks["milestone_2_objects"] = forbidden
    if forbidden:
        errors.append("Milestone 2/2.5 objects remain: " + ", ".join(forbidden))
    if simulator is not None:
        checks["milestone"] = simulator.get("simulator_milestone")
        checks["collision_enabled"] = simulator.get("collision_enabled")
        checks["terrain_status"] = simulator.get("terrain_status")
        checks["simulator_camera"] = simulator.camera.name if simulator.camera else None
        if checks["milestone"] not in {"1", "1-terrain-preview"}:
            errors.append("Simulator metadata is outside milestone 1")
        if checks["collision_enabled"] is not False:
            errors.append("Milestone 1 must not enable collision")
        if simulator.camera is None or simulator.camera.name != "SimulatorCamera":
            errors.append("Simulator scene does not use SimulatorCamera")
        if simulator.get("terrain_shared_with_pair_scenes") is True:
            terrain = bpy.data.objects.get("MartianTerrain")
            lighting = bpy.data.collections.get("MARS_CLEAR_DAY_LIGHTING")
            pair_scenes = [scene for scene in (simulator, normal, perforation) if scene is not None]
            checks["shared_terrain_present"] = (
                terrain is not None
                and len(pair_scenes) == 3
                and all(terrain.name in scene.objects for scene in pair_scenes)
            )
            checks["shared_world_identical"] = (
                len(pair_scenes) == 3
                and len({scene.world.name if scene.world else None for scene in pair_scenes}) == 1
            )
            checks["shared_exposure_identical"] = (
                len(pair_scenes) == 3
                and len({float(scene.view_settings.exposure) for scene in pair_scenes}) == 1
            )
            checks["shared_sun_present"] = (
                lighting is not None
                and len(pair_scenes) == 3
                and all(lighting.name in scene.collection.children for scene in pair_scenes)
            )
            for key in (
                "shared_terrain_present",
                "shared_world_identical",
                "shared_exposure_identical",
                "shared_sun_present",
            ):
                if not checks[key]:
                    errors.append(f"Pair context gate failed: {key}")
    if root is not None and required.issubset(set(bpy.data.objects.keys())):
        current = {
            "normal": mesh_digest(bpy.data.objects["Wheel_Skin_Normal"]),
            "anomaly": mesh_digest(bpy.data.objects["Wheel_Skin_Anomaly"]),
        }
        expected = {"normal": root.get("pair_digest_normal"),
                    "anomaly": root.get("pair_digest_anomaly")}
        checks["pair_geometry_unchanged"] = current == expected
        if current != expected:
            errors.append("Normal/perforation wheel geometry changed")

    report = {"schema_version": "1.0", "status": "completed" if not errors else "failed",
              "valid": not errors, "checks": checks, "errors": errors}
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Validated reopened milestone-1 rover simulator: {bpy.data.filepath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
