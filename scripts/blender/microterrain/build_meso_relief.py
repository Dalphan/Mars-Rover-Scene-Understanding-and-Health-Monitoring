"""Build, render and persist the deterministic microterrain Level 1 checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--geospatial-dir", type=Path)
    parser.add_argument("--source-blend", type=Path)
    parser.add_argument("--output-dir", type=Path)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    values = (args.config, args.geospatial_dir, args.source_blend, args.output_dir)
    if any(value is None for value in values):
        raise RuntimeError("config, geospatial-dir, source-blend and output-dir are required")
    return tuple(Path(value).resolve() for value in values)


def build(config_path: Path, geospatial_dir: Path, source_blend: Path, output_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.build_microterrain_patch import build_patch
    from scripts.blender.microterrain.common import bbox_center_xy, bbox_min_z, sample_height
    from scripts.blender.microterrain.render_microterrain_comparison import render_comparisons
    from src.microterrain.core import generate_meso_relief, validate_level1_config
    from src.microterrain.validation import validate_level1_report

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = validate_level1_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(bpy.data.filepath).resolve() != source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    heightfield = np.load(geospatial_dir / "terrain_heightfield.npz")
    macro_x, macro_y, macro_z = heightfield["x_local"], heightfield["y_local"], heightfield["z_local"]
    target_wheel = bpy.data.objects[config["microterrain"]["target_wheel"]]
    center_x, center_y = bbox_center_xy(target_wheel)
    half_size = float(config["microterrain"]["patch_size_m"]) / 2.0
    samples = contract["samples"]
    patch_x = np.linspace(center_x - half_size, center_x + half_size, samples, dtype=np.float32)
    patch_y = np.linspace(center_y - half_size, center_y + half_size, samples, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(patch_x, patch_y)
    columns = np.interp(grid_x, macro_x, np.arange(len(macro_x)))
    rows = np.interp(grid_y, macro_y, np.arange(len(macro_y)))
    c0, r0 = np.floor(columns).astype(np.int32), np.floor(rows).astype(np.int32)
    c1, r1 = np.minimum(c0 + 1, len(macro_x) - 1), np.minimum(r0 + 1, len(macro_y) - 1)
    wc, wr = columns - c0, rows - r0
    base_z = (macro_z[r0, c0] * (1 - wc) * (1 - wr) + macro_z[r0, c1] * wc * (1 - wr) + macro_z[r1, c0] * (1 - wc) * wr + macro_z[r1, c1] * wc * wr).astype(np.float32)
    displacement, metrics = generate_meso_relief(patch_x, patch_y, config, np)
    macro_local_bounds = [float(macro_x[0]), float(macro_y[0]), float(macro_x[-1]), float(macro_y[-1])]
    patch, macro_proxy = build_patch(patch_x, patch_y, base_z, displacement, macro_local_bounds, metrics)

    root = bpy.data.objects[config["rover"]["root_object"]]
    wheel_names = sorted(obj.name for obj in bpy.data.objects if obj.name.startswith("wheel_"))
    clearance = float(config["rover"]["clearance_m"])
    before_z = float(root.location.z)
    requirements = []
    for name in wheel_names:
        wheel = bpy.data.objects[name]
        x, y = bbox_center_xy(wheel)
        if patch_x[0] <= x <= patch_x[-1] and patch_y[0] <= y <= patch_y[-1]:
            terrain_z = sample_height(patch_x, patch_y, base_z + displacement, x, y)
        else:
            terrain_z = sample_height(macro_x, macro_y, macro_z, x, y)
        requirements.append(terrain_z + clearance - bbox_min_z(wheel))
    root.location.z += max(requirements)
    bpy.context.view_layer.update()
    clearances = {}
    for name in wheel_names:
        wheel = bpy.data.objects[name]
        x, y = bbox_center_xy(wheel)
        terrain_z = sample_height(patch_x, patch_y, base_z + displacement, x, y) if patch_x[0] <= x <= patch_x[-1] and patch_y[0] <= y <= patch_y[-1] else sample_height(macro_x, macro_y, macro_z, x, y)
        clearances[name] = bbox_min_z(wheel) - terrain_z

    # Select a deterministic, non-edge validation crop with high local relief
    # variance. The patch itself remains centered on the target wheel.
    spacing = float(config["microterrain"]["grid_spacing_m"])
    window_radius = max(2, int(round(0.05 / spacing)))
    edge_margin = max(window_radius + 1, int(round(0.30 / spacing)))
    stride = max(1, int(round(0.04 / spacing)))
    best_score, best_row, best_column = -1.0, samples // 2, samples // 2
    for row in range(edge_margin, samples - edge_margin, stride):
        for column in range(edge_margin, samples - edge_margin, stride):
            local_window = displacement[row - window_radius : row + window_radius + 1, column - window_radius : column + window_radius + 1]
            score = float(local_window.std())
            if score > best_score:
                best_score, best_row, best_column = score, row, column
    validation_x, validation_y = float(patch_x[best_column]), float(patch_y[best_row])
    validation_z = float((base_z + displacement)[best_row, best_column])
    renders = render_comparisons(config, output_dir / "renders", Vector((validation_x, validation_y, validation_z)))
    report = {
        "schema_version": 1, "level": 1, "config": str(config_path), "source_blend": str(source_blend),
        "patch": {"object": patch.name, "center_xy_m": [center_x, center_y], "bounds_m": [float(patch_x[0]), float(patch_y[0]), float(patch_x[-1]), float(patch_y[-1])], "size_m": float(config["microterrain"]["patch_size_m"]), "grid_spacing_m": float(config["microterrain"]["grid_spacing_m"]), "vertex_count": len(patch.data.vertices), "face_count": len(patch.data.polygons), "uv_layers": [layer.name for layer in patch.data.uv_layers], "expected_uv_bounds": [(float(patch_x[0]) - macro_local_bounds[0]) / (macro_local_bounds[2] - macro_local_bounds[0]), (float(patch_y[0]) - macro_local_bounds[1]) / (macro_local_bounds[3] - macro_local_bounds[1]), (float(patch_x[-1]) - macro_local_bounds[0]) / (macro_local_bounds[2] - macro_local_bounds[0]), (float(patch_y[-1]) - macro_local_bounds[1]) / (macro_local_bounds[3] - macro_local_bounds[1])]},
        "metrics": metrics,
        "macroterrain": {"source_object": "GaleTerrainVisual", "source_modified": False, "source_vertices": len(bpy.data.objects["GaleTerrainVisual"].data.vertices), "source_faces": len(bpy.data.objects["GaleTerrainVisual"].data.polygons), "runtime_proxy": macro_proxy.name, "opening_method": "material-space rectangular alpha mask; source mesh unchanged"},
        "rover_contact": {"root_z_before_m": before_z, "root_z_after_m": float(root.location.z), "vertical_adjustment_m": float(root.location.z) - before_z, "wheel_clearances_m": clearances, "method": "combined macroterrain/microterrain wheel-center clearance"},
        "renders": renders,
        "validation_camera_target": {"position_m": [validation_x, validation_y, validation_z], "selection_method": "maximum local displacement standard deviation in a 10 cm window, sampled on a 4 cm stride and excluding a 30 cm edge margin", "local_displacement_std_m": best_score},
        "scope": {"level2_clasts_enabled": False, "level3_shading_enabled": False},
    }
    errors = validate_level1_report(config, report)
    report["validation"] = {"ok": not errors, "errors": errors}
    (output_dir / "build_L1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Level 1 build report failed: " + "; ".join(errors))
    blend_path = output_dir / "microterrain_L1.blend"
    bpy.context.scene["microterrain_level"] = 1
    bpy.context.scene["microterrain_build_report"] = str(output_dir / "build_L1.json")
    bpy.context.scene["microterrain_seed"] = int(config["microterrain"]["seed"])
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report["blend"] = str(blend_path)
    (output_dir / "build_L1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build(*_arguments())
