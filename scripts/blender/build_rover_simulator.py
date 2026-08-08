from __future__ import annotations

import argparse
import bpy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mathutils import Vector


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rover_simulator.core import derive_four_probe_specs


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build milestone 1 of the rover simulator.")
    parser.add_argument("--source-blend", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--preparation-report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(values)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mesh_digest(obj: bpy.types.Object) -> str:
    digest = hashlib.sha256()
    mesh = obj.data
    digest.update(f"{len(mesh.vertices)}:{len(mesh.edges)}:{len(mesh.polygons)}".encode())
    for vertex in mesh.vertices:
        digest.update(("%.8f,%.8f,%.8f;" % tuple(vertex.co)).encode())
    for polygon in mesh.polygons:
        digest.update((",".join(str(int(v)) for v in polygon.vertices) + ";").encode())
    return digest.hexdigest()


def link_object(scene: bpy.types.Scene, obj: bpy.types.Object) -> None:
    if obj.name not in scene.objects:
        scene.collection.objects.link(obj)


def parent_keep_world(obj: bpy.types.Object, parent: bpy.types.Object) -> None:
    matrix = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = matrix


def descendants_of(root: bpy.types.Object) -> set[bpy.types.Object]:
    result: set[bpy.types.Object] = set()
    pending = list(root.children)
    while pending:
        current = pending.pop()
        if current in result:
            continue
        result.add(current)
        pending.extend(current.children)
    return result


def bounds(objects: Iterable[bpy.types.Object]) -> tuple[Vector, Vector]:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    return (
        Vector(tuple(min(point[i] for point in points) for i in range(3))),
        Vector(tuple(max(point[i] for point in points) for i in range(3))),
    )


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.matrix_world.translation).to_track_quat("-Z", "Y").to_euler()


def material(name: str, color: Sequence[float]) -> bpy.types.Material:
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1.0)
    result.use_nodes = True
    principled = result.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (*color, 1.0)
        principled.inputs["Emission Color"].default_value = (*color, 1.0)
        principled.inputs["Emission Strength"].default_value = 2.5
    return result


def create_probe(name: str, radius: float, height: float, probe_material: bpy.types.Material,
                 collection: bpy.types.Collection) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=height)
    obj = bpy.context.object
    obj.name = name
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)
    obj.data.materials.append(probe_material)
    obj["role"] = "ground_probe"
    return obj


def create_footprint(probes: Sequence[Any], height: float, footprint_material: bpy.types.Material,
                     collection: bpy.types.Collection) -> bpy.types.Object:
    ordered = sorted(probes, key=lambda p: (p.longitudinal, p.side))
    by_key = {(p.longitudinal, p.side): p for p in ordered}
    sequence = [by_key[("front", "left")], by_key[("front", "right")],
                by_key[("rear", "right")], by_key[("rear", "left")]]
    curve = bpy.data.curves.new("GroundProbe_Footprint_Curve", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.012
    spline = curve.splines.new("POLY")
    spline.points.add(4)
    for point, probe in zip(spline.points, [*sequence, sequence[0]]):
        x, y, z = probe.local_position
        point.co = (x, y, z + height * 0.5, 1.0)
    obj = bpy.data.objects.new("GroundProbe_Footprint", curve)
    collection.objects.link(obj)
    curve.materials.append(footprint_material)
    obj["role"] = "ground_probe_layout"
    return obj


def render(scene: bpy.types.Scene, path: Path) -> None:
    scene.render.filepath = str(path)
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = args.output_dir.expanduser().resolve()
    reports_dir = output_dir / "reports"
    renders_dir = output_dir / "renders"
    diagnostics_dir = output_dir / "diagnostics"
    for path in (reports_dir, renders_dir, diagnostics_dir):
        path.mkdir(parents=True, exist_ok=True)
    source = args.source_blend.expanduser().resolve()
    report_path = reports_dir / "build.json"
    source_hash = None
    try:
        if tuple(bpy.app.version[:2]) != (5, 2):
            raise RuntimeError(f"Expected Blender 5.2, got {bpy.app.version_string}")
        audit = load_json(args.audit_report.expanduser().resolve())
        preparation = load_json(args.preparation_report.expanduser().resolve())
        config = load_json(args.config.expanduser().resolve())
        source_hash = sha256_file(source)
        candidate_id = str(config["candidate_id"])
        if candidate_id != str(preparation["candidate"]["id"]):
            raise RuntimeError("Simulator candidate differs from preparation report")
        normal = bpy.data.scenes.get("Normal")
        perforation = bpy.data.scenes.get("Perforation")
        if normal is None or perforation is None:
            raise RuntimeError("Source blend is missing Normal or Perforation scene")
        required = {
            "Rover_Without_Selected_Wheel", "Wheel_Attachment", "Wheel_Details_Normal",
            "Wheel_Skin_Normal", "Wheel_Details_Perforation", "Wheel_Skin_Anomaly",
            "WheelCamera",
        }
        missing = required - set(bpy.data.objects.keys())
        if missing:
            raise RuntimeError("Source blend objects missing: " + ", ".join(sorted(missing)))
        bpy.data.objects["Wheel_Skin_Normal"].hide_render = False
        bpy.data.objects["Wheel_Skin_Normal"].hide_viewport = False
        pair_before = {
            "normal_skin": mesh_digest(bpy.data.objects["Wheel_Skin_Normal"]),
            "anomaly_skin": mesh_digest(bpy.data.objects["Wheel_Skin_Anomaly"]),
        }
        root = bpy.data.objects.new("RoverRoot", None)
        root.empty_display_type = "PLAIN_AXES"
        root.empty_display_size = 0.35
        root["role"] = "simulator_rover_root"
        root["candidate_id"] = candidate_id
        link_object(normal, root)
        link_object(perforation, root)
        for name in sorted(required):
            parent_keep_world(bpy.data.objects[name], root)

        simulator = bpy.data.scenes.new("Simulator")
        simulator.world = normal.world
        for collection in normal.collection.children:
            simulator.collection.children.link(collection)
        for obj in normal.collection.objects:
            link_object(simulator, obj)
        link_object(simulator, root)
        controls = bpy.data.collections.new("SIMULATOR_CONTROLS")
        simulator.collection.children.link(controls)

        radius = float(preparation["repair"]["outer_radius"])
        probes = derive_four_probe_specs(
            audit["wheel_detection"]["candidates"], candidate_id,
            preparation["canonical_transform"]["axis"], radius,
        )
        colors = {
            "GroundProbe_Front_Left": (1.0, 0.08, 0.04),
            "GroundProbe_Front_Right": (0.05, 1.0, 0.18),
            "GroundProbe_Rear_Left": (0.05, 0.25, 1.0),
            "GroundProbe_Rear_Right": (1.0, 0.72, 0.03),
        }
        probe_objects = []
        for spec in probes:
            obj = create_probe(spec.name, float(config["probes"]["marker_radius"]),
                               float(config["probes"]["marker_height"]),
                               material(f"{spec.name}_Material", colors[spec.name]), controls)
            obj.parent = root
            obj.location = Vector(spec.local_position)
            obj["source_candidate_id"] = spec.candidate_id
            obj["side"] = spec.side
            obj["longitudinal"] = spec.longitudinal
            probe_objects.append(obj)
        footprint = create_footprint(
            probes, float(config["probes"]["marker_height"]),
            material("GroundProbe_Footprint_Material", (0.0, 0.92, 1.0)), controls,
        )
        footprint.parent = root

        normal_rover = [bpy.data.objects[name] for name in (
            "Rover_Without_Selected_Wheel", "Wheel_Attachment",
            "Wheel_Details_Normal", "Wheel_Skin_Normal")]
        rover_min, rover_max = bounds(normal_rover)
        dimensions = rover_max - rover_min
        camera_cfg = config["simulator_camera"]
        target = Vector(((rover_min.x + rover_max.x) * 0.5,
                         (rover_min.y + rover_max.y) * 0.5,
                         rover_min.z + dimensions.z * float(camera_cfg["target_height_ratio"])))
        camera_data = bpy.data.cameras.new("SimulatorCamera")
        camera_data.lens = float(camera_cfg["lens_mm"])
        camera_data.clip_end = 1000.0
        chase = bpy.data.objects.new("SimulatorCamera", camera_data)
        controls.objects.link(chase)
        chase.parent = root
        chase.location = Vector((
            target.x + dimensions.x * float(camera_cfg["lateral_offset_width_ratio"]),
            rover_min.y - dimensions.y * float(camera_cfg["rear_offset_length_ratio"]),
            rover_max.z + dimensions.z * float(camera_cfg["height_offset_rover_height_ratio"]),
        ))
        look_at(chase, target)
        chase["role"] = "simulator_chase_camera"
        wheel_camera = bpy.data.objects["WheelCamera"]
        wheel_camera["role"] = "dataset_wheel_camera"
        simulator.camera = chase
        simulator["simulator_milestone"] = "1"
        simulator["controller_enabled"] = False
        simulator["collision_enabled"] = False
        simulator["terrain_status"] = "awaiting_replacement"
        simulator["candidate_id"] = candidate_id
        root["probe_names"] = json.dumps([probe.name for probe in probes])
        root["probe_specs"] = json.dumps([probe.as_dict() for probe in probes])
        root["pair_digest_normal"] = pair_before["normal_skin"]
        root["pair_digest_anomaly"] = pair_before["anomaly_skin"]

        render_cfg = config["render"]
        simulator.render.resolution_x = int(render_cfg["resolution_x"])
        simulator.render.resolution_y = int(render_cfg["resolution_y"])
        simulator.render.resolution_percentage = int(render_cfg["resolution_percentage"])
        simulator.render.image_settings.file_format = str(render_cfg["image_format"])
        simulator.render.engine = normal.render.engine
        render(simulator, renders_dir / "simulator_overview.png")
        for obj in [*probe_objects, footprint]:
            obj.hide_render = True
        simulator.camera = wheel_camera
        render(simulator, renders_dir / "wheel_closeup.png")
        for obj in [*probe_objects, footprint]:
            obj.hide_render = False
        simulator.camera = chase

        pair_after = {
            "normal_skin": mesh_digest(bpy.data.objects["Wheel_Skin_Normal"]),
            "anomaly_skin": mesh_digest(bpy.data.objects["Wheel_Skin_Anomaly"]),
        }
        if pair_after != pair_before or sha256_file(source) != source_hash:
            raise RuntimeError("Source or counterfactual geometry changed during build")
        output_blend = diagnostics_dir / "rover_simulator.blend"
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
        write_report(report_path, {
            "schema_version": "1.0", "status": "completed",
            "duration_seconds": round(time.perf_counter() - started, 3),
            "candidate_id": candidate_id,
            "scene": {"name": simulator.name, "rover_root": root.name,
                      "rover_children": sorted(obj.name for obj in descendants_of(root))},
            "probes": [probe.as_dict() for probe in probes],
            "cameras": {"simulator": chase.name, "wheel": wheel_camera.name},
            "wheel_attachment": {"object": "Wheel_Attachment", "parent": root.name},
            "counterfactual_geometry": {"before": pair_before, "after": pair_after,
                                         "unchanged": pair_before == pair_after},
            "terrain": {"status": "awaiting_replacement"},
            "milestone": {"number": 1, "controller_enabled": False,
                          "collision_enabled": False},
            "artifacts": {"blend": "diagnostics/rover_simulator.blend",
                          "overview": "renders/simulator_overview.png",
                          "closeup": "renders/wheel_closeup.png"},
            "errors": [],
        })
        print(f"Built rover simulator milestone 1: {output_blend}")
        return 0
    except Exception as exc:
        write_report(report_path, {"schema_version": "1.0", "status": "failed",
                                   "duration_seconds": round(time.perf_counter() - started, 3),
                                   "errors": [str(exc)], "traceback": traceback.format_exc()})
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))
