"""Shared Blender helpers for microterrain generation and validation."""

from __future__ import annotations

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def bbox_center_xy(obj: bpy.types.Object) -> tuple[float, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / len(corners)
    return float(center.x), float(center.y)


def bbox_min_z(obj: bpy.types.Object) -> float:
    return min(float((obj.matrix_world @ Vector(corner)).z) for corner in obj.bound_box)


def sample_height(x_axis, y_axis, z, x: float, y: float) -> float:
    column = float(np.interp(x, x_axis, np.arange(len(x_axis))))
    row = float(np.interp(y, y_axis, np.arange(len(y_axis))))
    c0, r0 = int(math.floor(column)), int(math.floor(row))
    c1, r1 = min(c0 + 1, len(x_axis) - 1), min(r0 + 1, len(y_axis) - 1)
    wc, wr = column - c0, row - r0
    return float(z[r0, c0] * (1 - wc) * (1 - wr) + z[r0, c1] * wc * (1 - wr) + z[r1, c0] * (1 - wc) * wr + z[r1, c1] * wc * wr)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def rover_objects(root: bpy.types.Object) -> list[bpy.types.Object]:
    result = []
    for obj in bpy.data.objects:
        current = obj
        while current is not None:
            if current == root:
                result.append(obj)
                break
            current = current.parent
    return result


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path, resolution: tuple[int, int]) -> None:
    scene.camera = camera
    scene.render.resolution_x, scene.render.resolution_y = map(int, resolution)
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
