"""Create deterministic nominal Gale illumination from solar metadata."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector


def setup_nominal_lighting(config: dict) -> tuple[bpy.types.Object, bpy.types.World]:
    lighting = config["lighting"]
    sun = bpy.data.objects.get(lighting["sun_name"])
    if sun is None:
        data = bpy.data.lights.new(f"{lighting['sun_name']}_data", "SUN")
        sun = bpy.data.objects.new(lighting["sun_name"], data)
        bpy.context.scene.collection.objects.link(sun)
    elif sun.type != "LIGHT" or sun.data.type != "SUN":
        raise RuntimeError(f"{sun.name} exists but is not a Sun light")

    azimuth = math.radians(float(lighting["azimuth_deg"]))
    elevation = math.radians(float(lighting["elevation_deg"]))
    direction_to_sun = Vector(
        (
            math.cos(elevation) * math.sin(azimuth),
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        )
    ).normalized()
    ray_direction = -direction_to_sun
    sun.rotation_euler = ray_direction.to_track_quat("-Z", "Y").to_euler()
    sun.data.energy = float(lighting["energy"])
    sun.data.angle = math.radians(float(lighting["angle_deg"]))
    sun["azimuth_deg"] = float(lighting["azimuth_deg"])
    sun["elevation_deg"] = float(lighting["elevation_deg"])
    sun["direction_to_sun"] = list(direction_to_sun)

    world = bpy.data.worlds.get("GaleNominalWorld") or bpy.data.worlds.new("GaleNominalWorld")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = tuple(lighting["ambient_color"])
    background.inputs["Strength"].default_value = float(lighting["ambient_strength"])
    world["ambient_strength"] = float(lighting["ambient_strength"])
    bpy.context.scene.world = world
    return sun, world


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not argv:
        raise RuntimeError("Pass the Gale config JSON path after --")
    setup_nominal_lighting(json.loads(Path(argv[0]).read_text(encoding="utf-8")))
