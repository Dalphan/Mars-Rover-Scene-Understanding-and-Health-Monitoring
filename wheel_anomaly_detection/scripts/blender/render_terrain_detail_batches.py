from __future__ import annotations

import argparse
import bpy
import json
import math
import sys
from pathlib import Path

from mathutils import Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Render the approved terrain detail patches.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--render-prefix", default="patch")
    parser.add_argument("--patch", choices=("A", "B", "C"), required=True)
    parser.add_argument("--render", choices=("overview", "zenith", "normal", "perforation"), required=True)
    return parser.parse_args(values)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)


def terrain_surface(terrain: bpy.types.Object, bvh: BVHTree, x: float, y: float) -> float:
    inverse = terrain.matrix_world.inverted()
    origin = inverse @ Vector((x, y, 10.0))
    direction = (inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
    hit, _, _, _ = bvh.ray_cast(origin, direction, 20.0)
    if hit is None:
        raise RuntimeError(f"Terrain raycast missed at ({x}, {y})")
    return (terrain.matrix_world @ hit).z


def descendant(obj: bpy.types.Object, root: bpy.types.Object) -> bool:
    parent = obj.parent
    while parent is not None:
        if parent == root:
            return True
        parent = parent.parent
    return False


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    output_dir = args.output_dir.expanduser().resolve()
    renders_dir = output_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)
    simulator = bpy.data.scenes["Simulator"]
    normal = bpy.data.scenes["Normal"]
    perforation = bpy.data.scenes["Perforation"]
    terrain = bpy.data.objects[config["terrain_object"]]
    root = bpy.data.objects[config["rover_root"]]
    chase = bpy.data.objects["SimulatorCamera"]
    wheel_camera = bpy.data.objects["WheelCamera"]
    patch = next(item for item in config["patches"] if item["id"] == f"TerrainPatch_{args.patch}")
    patch_obj = bpy.data.objects[patch["id"]]
    if patch_obj is None:
        raise RuntimeError(f"Missing patch object {patch['id']}")
    report_path = args.report or (output_dir / "reports" / "terrain_detail_pilot.json")
    report = json.loads(report_path.expanduser().resolve().read_text(encoding="utf-8"))
    report_by_id = {entry["patch_id"]: entry for entry in report["patches"]}
    base_root = root.location.copy()
    base_chase = chase.matrix_world.copy()
    base_wheel_local = wheel_camera.matrix_local.copy()
    wheel_bbox = [root.matrix_world @ Vector(corner) for corner in bpy.data.objects[config["observed_wheel"]].bound_box]
    anchor_x = sum(point.x for point in wheel_bbox) / len(wheel_bbox)
    anchor_y = sum(point.y for point in wheel_bbox) / len(wheel_bbox)
    anchor_surface = report_by_id["TerrainPatch_A"]["surface_z"]
    if "center_world" in patch:
        center_x, center_y = patch["center_world"]
    else:
        center_x, center_y = anchor_x, anchor_y
    surface = report_by_id[patch["id"]]["surface_z"]
    root.location = Vector((center_x, center_y, base_root.z + surface - anchor_surface))
    delta = root.location - base_root
    if not descendant(chase, root):
        chase.matrix_world.translation = base_chase.translation + delta
    if not descendant(wheel_camera, root):
        wheel_camera.matrix_world.translation = (wheel_camera.matrix_world.translation + delta)
    look_at(chase, root.location + Vector((0.0, 0.0, 0.35)))

    patch_camera = bpy.data.objects.get("TerrainPatchCamera")
    if patch_camera is None:
        camera_data = bpy.data.cameras.new("TerrainPatchCamera_Data")
        patch_camera = bpy.data.objects.new("TerrainPatchCamera", camera_data)
        simulator.collection.objects.link(patch_camera)
    patch_camera.location = (center_x, center_y, 12.0)
    look_at(patch_camera, Vector((center_x, center_y, surface)))
    patch_camera.data.type = "ORTHO"
    patch_camera.data.ortho_scale = patch["size_m"] * 1.25
    filename = f"{args.render_prefix}_{args.patch}_{args.render}.png"
    if args.render == "overview":
        render(simulator, chase, renders_dir / filename)
    elif args.render == "zenith":
        render(simulator, patch_camera, renders_dir / filename)
    elif args.render == "normal":
        render(normal, wheel_camera, renders_dir / filename)
    else:
        render(perforation, wheel_camera, renders_dir / filename)
    root.location = base_root
    chase.matrix_world = base_chase
    wheel_camera.matrix_local = base_wheel_local
    simulator.camera = chase
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    print(f"Rendered patch {args.patch} {args.render}: {renders_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
