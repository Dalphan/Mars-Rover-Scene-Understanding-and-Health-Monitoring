"""Validate the persisted Level-1 microterrain scene and artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.build_report.resolve(), args.output_report.resolve()


def validate(config_path: Path, build_report_path: Path, output_report_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.build_microterrain_patch import MACRO_PROXY_OBJECT, PATCH_OBJECT
    from src.microterrain.core import generate_meso_relief, validate_level1_config
    from src.microterrain.validation import validate_level1_report

    config = json.loads(config_path.read_text(encoding="utf-8"))
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    contract = validate_level1_config(config)
    errors = validate_level1_report(config, build_report)
    patch = bpy.data.objects.get(PATCH_OBJECT)
    original = bpy.data.objects.get("GaleTerrainVisual")
    proxy = bpy.data.objects.get(MACRO_PROXY_OBJECT)
    if patch is None or patch.type != "MESH":
        errors.append("Missing Level 1 patch mesh")
    else:
        if len(patch.data.vertices) != contract["vertex_count"] or len(patch.data.polygons) != contract["face_count"]:
            errors.append("Persisted patch topology mismatch")
        if [layer.name for layer in patch.data.uv_layers] != ["GeoreferencedUV"]:
            errors.append("Persisted patch georeferenced UV layer mismatch")
        if patch.get("displacement_signature_sha256") != build_report["metrics"]["signature_sha256"]:
            errors.append("Persisted patch signature mismatch")
        uv_layer = patch.data.uv_layers.get("GeoreferencedUV")
        if uv_layer:
            uv_values = np.empty(len(uv_layer.uv) * 2, dtype=np.float32)
            uv_layer.uv.foreach_get("vector", uv_values)
            uv_values = uv_values.reshape(-1, 2)
            actual_uv_bounds = [float(uv_values[:, 0].min()), float(uv_values[:, 1].min()), float(uv_values[:, 0].max()), float(uv_values[:, 1].max())]
            if not np.allclose(actual_uv_bounds, build_report["patch"]["expected_uv_bounds"], atol=1e-6):
                errors.append(f"Patch UV bounds are not georeferenced: {actual_uv_bounds}")
    if original is None or len(original.data.vertices) != build_report["macroterrain"]["source_vertices"] or len(original.data.polygons) != build_report["macroterrain"]["source_faces"]:
        errors.append("Original macroterrain topology changed")
    if proxy is None or not proxy.data.materials or not proxy.data.materials[0].get("microterrain_hole_bounds_m"):
        errors.append("Runtime macroterrain opening is missing")
    forbidden = [obj.name for obj in bpy.data.objects if "clast" in obj.name.lower()]
    if forbidden:
        errors.append(f"Level 2 objects found prematurely: {forbidden}")

    samples = contract["samples"]
    # Reconstruct the axes from the same authoritative center/size inputs used
    # by the builder. Re-linspacing between already float32-rounded endpoints
    # can move interior samples by one ULP and therefore change the byte-level
    # deterministic signature even though the configured terrain is identical.
    center_x, center_y = map(float, build_report["patch"]["center_xy_m"])
    half_size = float(build_report["patch"]["size_m"]) / 2.0
    x_axis = np.linspace(center_x - half_size, center_x + half_size, samples, dtype=np.float32)
    y_axis = np.linspace(center_y - half_size, center_y + half_size, samples, dtype=np.float32)
    _, regenerated_metrics = generate_meso_relief(x_axis, y_axis, config, np)
    if regenerated_metrics["signature_sha256"] != build_report["metrics"]["signature_sha256"]:
        errors.append("Regenerated relief is not deterministic")
    for path in build_report["renders"].values():
        if not Path(path).is_file():
            errors.append(f"Missing render file: {path}")
    minimum_clearance = min(map(float, build_report["rover_contact"]["wheel_clearances_m"].values()))
    if minimum_clearance < float(config["rover"]["clearance_m"]) - 1e-5:
        errors.append(f"Rover penetrates combined terrain: {minimum_clearance}")
    report = {"ok": not errors, "level": 1, "blend": bpy.data.filepath, "errors": errors, "deterministic_signature_sha256": regenerated_metrics["signature_sha256"], "patch_vertices": 0 if patch is None else len(patch.data.vertices), "patch_faces": 0 if patch is None else len(patch.data.polygons), "minimum_wheel_clearance_m": minimum_clearance, "renders": build_report["renders"], "scope": {"level2_clasts_enabled": False, "level3_shading_enabled": False}}
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("Microterrain Level 1 validation failed: " + "; ".join(errors))
    return report


if __name__ == "__main__":
    validate(*_arguments())
