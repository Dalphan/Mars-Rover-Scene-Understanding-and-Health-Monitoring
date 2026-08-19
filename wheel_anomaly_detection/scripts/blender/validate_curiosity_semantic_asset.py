"""Validate and print the six semantic wheel objects in the open Blender file."""

from __future__ import annotations

import json

import bpy


EXPECTED = (
    "wheel_front_left",
    "wheel_front_right",
    "wheel_middle_left",
    "wheel_middle_right",
    "wheel_rear_left",
    "wheel_rear_right",
)


def _rounded(values):
    return [round(float(value), 9) for value in values]


def _bbox_world(obj):
    corners = [obj.matrix_world @ obj.data.vertices[index].co for index in range(len(obj.data.vertices))]
    return {
        "min": _rounded(min(point[axis] for point in corners) for axis in range(3)),
        "max": _rounded(max(point[axis] for point in corners) for axis in range(3)),
    }


def validate() -> dict:
    wheel_like = sorted(obj.name for obj in bpy.data.objects if obj.name.startswith("wheel_"))
    missing = sorted(set(EXPECTED) - set(wheel_like))
    unexpected = sorted(set(wheel_like) - set(EXPECTED))
    duplicates = sorted(name for name in wheel_like if any(name.startswith(f"{expected}.") for expected in EXPECTED))
    wheels = []
    errors = []
    for name in EXPECTED:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        if obj.type != "MESH":
            errors.append(f"{name} is {obj.type}, expected MESH")
            continue
        materials = [slot.material.name if slot.material else None for slot in obj.material_slots]
        uv_layers = [layer.name for layer in obj.data.uv_layers]
        if "wheels" not in materials:
            errors.append(f"{name} does not retain the wheels material")
        if not uv_layers:
            errors.append(f"{name} has no UV layer")
        if obj.parent is None or obj.parent.name != "Wheels":
            errors.append(f"{name} is not parented to Wheels")
        wheels.append(
            {
                "name": name,
                "location": _rounded(obj.location),
                "rotation_euler": _rounded(obj.rotation_euler),
                "scale": _rounded(obj.scale),
                "bounding_box_world": _bbox_world(obj),
                "vertices": len(obj.data.vertices),
                "edges": len(obj.data.edges),
                "faces": len(obj.data.polygons),
                "materials": materials,
                "uv_layers": uv_layers,
            }
        )

    if missing:
        errors.append(f"Missing wheel names: {missing}")
    if unexpected:
        errors.append(f"Unexpected wheel-like names: {unexpected}")
    if duplicates:
        errors.append(f"Duplicate wheel variants: {duplicates}")
    if len(wheels) != 6:
        errors.append(f"Expected exactly six wheel meshes, found {len(wheels)}")

    report = {
        "ok": not errors,
        "scene": bpy.context.scene.name,
        "blend_file": bpy.data.filepath,
        "expected_wheel_count": 6,
        "wheel_like_objects": wheel_like,
        "missing": missing,
        "unexpected": unexpected,
        "duplicates": duplicates,
        "wheels": wheels,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("Semantic wheel validation failed: " + "; ".join(errors))
    return report


if __name__ == "__main__":
    validate()
