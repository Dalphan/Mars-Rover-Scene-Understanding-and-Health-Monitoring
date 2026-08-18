"""Immutable geometry cache shared by clean and paired Blender renderers."""

from __future__ import annotations

import math

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


class StaticGeometryCache:
    """Cache local mesh coordinates, never resolved poses or gate results."""

    def __init__(self, wheels, patch, contact_settings):
        from scripts.blender.render_domain_randomization_preview import _patch_bounds

        self.patch_bounds = _patch_bounds(patch)
        self.patch_top = max(float((patch.matrix_world @ Vector(corner)).z) for corner in patch.bound_box)
        self.patch_inverse = patch.matrix_world.inverted()
        self.patch_down_local = (self.patch_inverse.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized()
        self.outer_local = {}
        self.projection_local = {}
        outer_fraction = float(contact_settings["outer_radius_fraction"])
        for wheel in wheels:
            center = sum((Vector(corner) for corner in wheel.bound_box), Vector()) / 8.0
            vertices = [vertex.co.copy() for vertex in wheel.data.vertices]
            radial = [math.hypot(float(point.y - center.y), float(point.z - center.z)) for point in vertices]
            threshold = max(radial) * outer_fraction
            self.outer_local[wheel.name] = [point for point, radius in zip(vertices, radial) if radius >= threshold]
            self.projection_local[wheel.name] = vertices

    def settle(self, target, patch, settings):
        contact = settings["vertical_contact"]
        root = bpy.data.objects[settings["rover_root_object"]]
        outer_points = [target.matrix_world @ point for point in self.outer_local[target.name]]
        maximum_samples = int(contact["maximum_contact_samples"])
        sample_points = sorted(outer_points, key=lambda point: (float(point.z), float(point.x), float(point.y)))[:maximum_samples]
        clearances = []
        for point in sample_points:
            origin_world = Vector((float(point.x), float(point.y), self.patch_top + 1.0))
            hit, location, _normal, _face = patch.ray_cast(
                self.patch_inverse @ origin_world,
                self.patch_down_local,
                distance=5.0,
            )
            if hit:
                clearances.append(float(point.z - (patch.matrix_world @ location).z))
        if len(clearances) < int(contact["minimum_contact_samples"]):
            raise RuntimeError(f"Terrain contact rays missed the patch for {target.name}")
        before = min(clearances)
        desired = -float(contact["target_penetration_m"])
        vertical_translation = desired - before
        root.matrix_world = Matrix.Translation((0.0, 0.0, vertical_translation)) @ root.matrix_world
        bpy.context.view_layer.update()
        after = before + vertical_translation
        return {
            "ok": bool(-float(contact["maximum_penetration_m"]) <= after <= float(contact["maximum_gap_m"])),
            "enabled": True,
            "target_wheel": target.name,
            "outer_radius_fraction": float(contact["outer_radius_fraction"]),
            "outer_candidate_count": len(outer_points),
            "sampled_contact_count": len(clearances),
            "minimum_clearance_before_m": before,
            "vertical_translation_m": vertical_translation,
            "minimum_clearance_after_m": after,
            "target_penetration_m": float(contact["target_penetration_m"]),
            "maximum_penetration_m": float(contact["maximum_penetration_m"]),
            "maximum_gap_m": float(contact["maximum_gap_m"]),
        }

    def footprint(self, scene, camera, minimum_margin):
        bounds = self.patch_bounds
        origin = camera.matrix_world.translation
        rotation = camera.matrix_world.to_3x3()
        intersections = []
        for corner in camera.data.view_frame(scene=scene):
            direction = (rotation @ corner).normalized()
            if direction.z >= -1e-8:
                return {"ok": False, "reason": "frustum_corner_above_horizon", "bounds": bounds, "corners_world_xy": []}
            distance = (float(bounds["plane_z"]) - float(origin.z)) / float(direction.z)
            if distance <= 0.0:
                return {"ok": False, "reason": "terrain_plane_behind_camera", "bounds": bounds, "corners_world_xy": []}
            intersections.append(origin + direction * distance)
        margins = [
            min(
                float(point.x) - bounds["x_min"],
                bounds["x_max"] - float(point.x),
                float(point.y) - bounds["y_min"],
                bounds["y_max"] - float(point.y),
            )
            for point in intersections
        ]
        minimum = min(margins)
        return {
            "ok": minimum >= float(minimum_margin),
            "reason": None if minimum >= float(minimum_margin) else "insufficient_patch_edge_margin",
            "minimum_edge_margin_m": float(minimum),
            "required_edge_margin_m": float(minimum_margin),
            "bounds": bounds,
            "corners_world_xy": [[float(point.x), float(point.y)] for point in intersections],
        }

    def framing(self, scene, camera, wheel, basis, pose, settings):
        projected = [world_to_camera_view(scene, camera, wheel.matrix_world @ point) for point in self.projection_local[wheel.name]]
        xs, ys = [float(point.x) for point in projected], [float(point.y) for point in projected]
        box = [min(xs), min(ys), max(xs), max(ys)]
        projection = {
            "normalized_bbox": box,
            "projection_source": "cached_mesh_vertices",
            "bbox_area_fraction": max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]),
            "fully_inside_frame": box[0] >= 0.0 and box[1] >= 0.0 and box[2] <= 1.0 and box[3] <= 1.0,
        }
        if pose["crop_policy"] == "full_wheel":
            margin = float(settings["full_wheel_frame_margin_fraction"])
            low, high = map(float, settings["full_wheel_bbox_area_fraction"])
            ok = (
                box[0] >= margin
                and box[1] >= margin
                and box[2] <= 1.0 - margin
                and box[3] <= 1.0 - margin
                and low <= projection["bbox_area_fraction"] <= high
            )
            return {**projection, "ok": bool(ok), "policy": "full_wheel", "required_margin_fraction": margin}
        visible_area = max(0.0, min(1.0, box[2]) - max(0.0, box[0])) * max(0.0, min(1.0, box[3]) - max(0.0, box[1]))
        target = world_to_camera_view(scene, camera, basis["target"])
        margin = float(settings["detail_target_frame_margin_fraction"])
        target_ok = margin <= target.x <= 1.0 - margin and margin <= target.y <= 1.0 - margin and target.z > 0.0
        return {
            **projection,
            "ok": bool(visible_area >= float(settings["detail_minimum_visible_bbox_area_fraction"]) and target_ok),
            "policy": "intentional_detail_crop",
            "visible_bbox_area_fraction": float(visible_area),
            "target_normalized": [float(target.x), float(target.y), float(target.z)],
        }


__all__ = ["StaticGeometryCache"]
