from __future__ import annotations

import argparse
import bmesh
import bpy
import sys
from pathlib import Path
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Render selected disconnected Wheel_Details components."
    )
    parser.add_argument("--indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--source-object", default="Wheel_Details")
    parser.add_argument("--skin-object", default="Wheel_Skin")
    parser.add_argument("--isolate-only", action="store_true")
    parser.add_argument("--frame-component", action="store_true")
    return parser.parse_args(values)


def components(mesh: bpy.types.Mesh) -> list[set[int]]:
    adjacency = [set() for _ in mesh.vertices]
    for edge in mesh.edges:
        left, right = (int(value) for value in edge.vertices)
        adjacency[left].add(right)
        adjacency[right].add(left)
    result: list[set[int]] = []
    unseen = set(range(len(mesh.vertices)))
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        current_component: set[int] = set()
        while stack:
            current = stack.pop()
            current_component.add(current)
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        result.append(current_component)
    return result


def isolate_component(
    source: bpy.types.Object, vertex_indices: set[int], name: str
) -> bpy.types.Object:
    obj = source.copy()
    obj.data = source.data.copy()
    obj.name = name
    bpy.context.scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    delete_vertices = [
        vertex for vertex in bm.verts if int(vertex.index) not in vertex_indices
    ]
    bmesh.ops.delete(bm, geom=delete_vertices, context="VERTS")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update(calc_edges=True)
    return obj


def read_pixels(path: Path) -> list[float]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return list(image.pixels[:])
    finally:
        bpy.data.images.remove(image)


def compare_renders(reference: Path, output_dir: Path) -> None:
    baseline = read_pixels(reference)
    rows = []
    for path in sorted(output_dir.glob("component_*.png")):
        pixels = read_pixels(path)
        changed = 0
        total = 0.0
        for offset in range(0, len(baseline), 4):
            difference = max(
                abs(float(baseline[offset + channel]) - float(pixels[offset + channel]))
                for channel in range(3)
            )
            changed += int(difference > (2.0 / 255.0))
            total += difference
        rows.append((changed, total, path.name))
    for changed, total, name in sorted(rows, reverse=True)[:30]:
        print(f"COMPONENT_DIFF {name} changed={changed} total={total:.8f}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.reference is not None:
        compare_renders(args.reference.expanduser().resolve(), output_dir)
        return 0
    if not args.indices:
        raise ValueError("--indices is required when --reference is absent")
    requested = [int(value) for value in args.indices.split(",")]
    scene = bpy.data.scenes.get("PreparationWorkspace")
    details = bpy.data.objects.get(args.source_object)
    skin = bpy.data.objects.get(args.skin_object)
    if scene is None or details is None or skin is None:
        raise RuntimeError(
            "PreparationWorkspace or requested source/skin object missing: "
            f"source={args.source_object}, skin={args.skin_object}"
        )
    groups = components(details.data)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera = scene.camera
    if camera is None:
        raise RuntimeError("PreparationWorkspace has no active camera")
    camera_matrix = camera.matrix_world.copy()
    view_direction = camera.location.normalized()
    original_visibility = {
        obj.name: bool(obj.hide_render) for obj in bpy.data.objects if obj.type == "MESH"
    }
    for component_index in requested:
        if not 0 <= component_index < len(groups):
            raise ValueError(f"Component index out of range: {component_index}")
        isolated = isolate_component(
            details, groups[component_index], f"Detail_Component_{component_index:03d}"
        )
        if args.frame_component:
            points = [isolated.matrix_world @ Vector(corner) for corner in isolated.bound_box]
            minimum = Vector(
                tuple(min(point[axis] for point in points) for axis in range(3))
            )
            maximum = Vector(
                tuple(max(point[axis] for point in points) for axis in range(3))
            )
            target = (minimum + maximum) * 0.5
            extent = max(maximum - minimum)
            camera.location = target + view_direction * max(extent * 3.2, 0.08)
            camera.rotation_euler = (target - camera.location).to_track_quat(
                "-Z", "Y"
            ).to_euler()
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                visible = {isolated} if args.isolate_only else {skin, isolated}
                obj.hide_render = obj not in visible
        scene.render.filepath = str(output_dir / f"component_{component_index:03d}.png")
        bpy.ops.render.render(scene=scene.name, write_still=True)
        bpy.data.objects.remove(isolated, do_unlink=True)
    for name, hidden in original_visibility.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = hidden
    camera.matrix_world = camera_matrix
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
