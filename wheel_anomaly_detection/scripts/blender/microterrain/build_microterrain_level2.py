"""Build, render and persist the deterministic Level-2 clast checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--level1-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.source_blend.resolve(), args.level1_report.resolve(), args.output_dir.resolve()


def _wheel_exclusion_zones(bounds: list[float], padding: float) -> list[list[float]]:
    zones = []
    for wheel in sorted((obj for obj in bpy.data.objects if obj.name.startswith("wheel_")), key=lambda obj: obj.name):
        corners = [wheel.matrix_world @ Vector(corner) for corner in wheel.bound_box]
        minimum_x, maximum_x = min(v.x for v in corners), max(v.x for v in corners)
        minimum_y, maximum_y = min(v.y for v in corners), max(v.y for v in corners)
        if maximum_x < bounds[0] or minimum_x > bounds[2] or maximum_y < bounds[1] or minimum_y > bounds[3]:
            continue
        center_x = (minimum_x + maximum_x) / 2.0
        center_y = (minimum_y + maximum_y) / 2.0
        # The wheel axle is approximately Blender X. Preserve its full tread
        # width but only the local ground-contact portion of its Y diameter.
        radius_x = 0.55 * (maximum_x - minimum_x) + padding
        radius_y = 0.25 * (maximum_y - minimum_y) + padding
        zones.append([float(center_x), float(center_y), float(radius_x), float(radius_y)])
    return zones


def build(config_path: Path, source_blend: Path, level1_report_path: Path, output_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.build_clast_library import build_clast_library
    from scripts.blender.microterrain.build_clast_scatter import build_clast_scatter
    from scripts.blender.microterrain.render_level2_comparison import render_level2_comparisons
    from src.microterrain.clasts import generate_clast_scatter, validate_level2_config
    from src.microterrain.core import generate_meso_relief
    from src.microterrain.level2_validation import validate_level2_report

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = validate_level2_config(config)
    level1_report = json.loads(level1_report_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(bpy.data.filepath).resolve() != source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    patch = bpy.data.objects.get("MicroterrainPatch_L1")
    if patch is None or patch.type != "MESH":
        raise RuntimeError("The verified Level-1 patch is missing")
    if not patch.matrix_world.is_identity:
        raise RuntimeError("Level-1 patch must retain identity world transform")
    if patch.get("displacement_signature_sha256") != level1_report["metrics"]["signature_sha256"]:
        raise RuntimeError("Persisted Level-1 patch does not match its report")

    center_x, center_y = map(float, level1_report["patch"]["center_xy_m"])
    half_size = float(level1_report["patch"]["size_m"]) / 2.0
    samples = contract["samples"]
    patch_x = np.linspace(center_x - half_size, center_x + half_size, samples, dtype=np.float32)
    patch_y = np.linspace(center_y - half_size, center_y + half_size, samples, dtype=np.float32)
    coordinates = np.empty(len(patch.data.vertices) * 3, dtype=np.float32)
    patch.data.vertices.foreach_get("co", coordinates)
    surface_z = coordinates.reshape(-1, 3)[:, 2].reshape(samples, samples)
    displacement, regenerated_l1 = generate_meso_relief(patch_x, patch_y, config, np)
    if regenerated_l1["signature_sha256"] != level1_report["metrics"]["signature_sha256"]:
        raise RuntimeError("Level-1 relief cannot be regenerated from the Level-2 configuration")
    bounds = [float(patch_x[0]), float(patch_y[0]), float(patch_x[-1]), float(patch_y[-1])]
    zones = _wheel_exclusion_zones(bounds, float(config["microterrain"]["clasts"]["wheel_exclusion_padding_m"]))
    families, clast_metrics = generate_clast_scatter(patch_x, patch_y, surface_z, displacement, config, np, zones)
    source_collections, library_metrics = build_clast_library(config)
    scatter_objects = build_clast_scatter(families, source_collections, clast_metrics)

    # Persist and reopen before rendering. Blender 5.2/Eevee initializes the
    # instanced shadow state reliably from the saved native L2 visibility,
    # while rendering immediately after node construction can produce a dark
    # first frame for unrelated rover meshes.
    blend_path = output_dir / "microterrain_L2.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    patch = bpy.data.objects["MicroterrainPatch_L1"]
    scatter_objects = {
        "fine_grains": bpy.data.objects["ClastScatter_Fine_L2"],
        "fragments": bpy.data.objects["ClastScatter_Fragments_L2"],
        "coarse_clasts": bpy.data.objects["ClastScatter_Coarse_L2"],
    }

    coarse = families["coarse_clasts"]
    coarse_positions = coarse["positions"]
    medium_indices = np.flatnonzero(coarse["size_class_index"] == 1)
    candidates = medium_indices if len(medium_indices) else np.arange(len(coarse_positions))
    distances = (coarse_positions[candidates, 0] - center_x) ** 2 + (coarse_positions[candidates, 1] - center_y) ** 2
    target_index = int(candidates[int(np.argmin(distances))])
    close_target = Vector(tuple(map(float, coarse_positions[target_index])))
    center_index = samples // 2
    patch_center = Vector((center_x, center_y, float(surface_z[center_index, center_index])))
    renders = render_level2_comparisons(config, output_dir / "renders", close_target, patch_center)
    actual_burial = {
        family: [float(families[family]["burial"].min()), float(families[family]["burial"].max())]
        for family in families
    }
    report = {
        "schema_version": 1,
        "level": 2,
        "config": str(config_path),
        "source_blend": str(source_blend),
        "level1": {
            "report": str(level1_report_path),
            "displacement_signature_sha256": regenerated_l1["signature_sha256"],
            "patch_object": patch.name,
            "patch_topology": [len(patch.data.vertices), len(patch.data.polygons)],
            "source_modified": False,
        },
        "clasts": {**clast_metrics, "actual_burial_ranges": actual_burial},
        "library": library_metrics,
        "instancing": {
            "geometry_nodes": True,
            "instances_realized": False,
            "scatter_object_count": len(scatter_objects),
            "scatter_objects": {family: obj.name for family, obj in scatter_objects.items()},
            "point_vertex_count": sum(len(obj.data.vertices) for obj in scatter_objects.values()),
        },
        "closeup_target": {
            "position_m": list(map(float, close_target)),
            "selection_method": "representative medium coarse clast closest to patch center",
            "coarse_instance_index": target_index,
            "characteristic_size_m": float(coarse["characteristic_size_m"][target_index]),
            "size_class_index": int(coarse["size_class_index"][target_index]),
        },
        "renders": renders,
        "scope": {"level3_shading_enabled": False},
    }
    errors = validate_level2_report(config, report)
    report["validation"] = {"ok": not errors, "errors": errors}
    report_path = output_dir / "build_L2.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Level-2 build report failed: " + "; ".join(errors))
    bpy.context.scene["microterrain_level"] = 2
    bpy.context.scene["microterrain_seed"] = int(config["microterrain"]["seed"])
    bpy.context.scene["microterrain_level2_build_report"] = str(report_path)
    bpy.context.scene["level2_scatter_signature_sha256"] = clast_metrics["signature_sha256"]
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report["blend"] = str(blend_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build(*_arguments())
