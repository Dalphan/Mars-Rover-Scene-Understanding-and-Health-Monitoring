"""Build, render and persist the deterministic Level-3 material checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-blend", type=Path, required=True)
    parser.add_argument("--level2-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.source_blend.resolve(), args.level2_report.resolve(), args.output_dir.resolve()


def build(config_path: Path, source_blend: Path, level2_report_path: Path, output_dir: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.build_microterrain_material import build_level3_materials
    from scripts.blender.microterrain.render_level3_comparison import render_level3_comparisons
    from src.microterrain.level3_validation import validate_level3_report
    from src.microterrain.material import validate_level3_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract = validate_level3_config(config)
    level2_report = json.loads(level2_report_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(bpy.data.filepath).resolve() != source_blend:
        bpy.ops.wm.open_mainfile(filepath=str(source_blend))
    if int(bpy.context.scene.get("microterrain_level", 0)) != 2:
        raise RuntimeError("Level 3 must start from the persisted Level-2 checkpoint")
    expected_scatter = level2_report["clasts"]["signature_sha256"]
    if bpy.context.scene.get("level2_scatter_signature_sha256") != expected_scatter:
        raise RuntimeError("Level-2 scene and report scatter signatures differ")
    patch = bpy.data.objects["MicroterrainPatch_L1"]
    patch_topology_before = [len(patch.data.vertices), len(patch.data.polygons)]
    patch_signature_before = patch.get("displacement_signature_sha256")
    prototype_names = [name for names in level2_report["library"]["families"].values() for name in names]
    terrain_material, clast_mapping, material_metrics = build_level3_materials(config, contract["material_signature_sha256"], prototype_names)
    terrain_material_name = terrain_material.name

    blend_path = output_dir / "microterrain_L3.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    close_target = Vector(tuple(map(float, level2_report["closeup_target"]["position_m"])))
    renders = render_level3_comparisons(config, output_dir / "renders", close_target, prototype_names)
    patch = bpy.data.objects["MicroterrainPatch_L1"]
    scatter_objects = [bpy.data.objects[name] for name in level2_report["instancing"]["scatter_objects"].values()]
    scatter_signatures = {obj.get("clast_family"): obj.get("scatter_signature_sha256") for obj in scatter_objects}
    geometry_unchanged = (
        [len(patch.data.vertices), len(patch.data.polygons)] == patch_topology_before
        and patch.get("displacement_signature_sha256") == patch_signature_before
        and scatter_signatures == level2_report["clasts"]["family_signatures_sha256"]
    )
    report = {
        "schema_version": 1,
        "level": 3,
        "config": str(config_path),
        "source_blend": str(source_blend),
        "level2": {
            "report": str(level2_report_path),
            "scatter_signature_sha256": expected_scatter,
            "instance_count": level2_report["clasts"]["total_instances"],
        },
        "material": material_metrics,
        "geometry": {
            "level2_modified": not geometry_unchanged,
            "patch_topology": patch_topology_before,
            "level1_displacement_signature_sha256": patch_signature_before,
            "scatter_signature_sha256": expected_scatter,
            "family_signatures_sha256": scatter_signatures,
        },
        "ablation": {
            "level2_materials_preserved": all(bpy.data.materials.get(name) for name in material_metrics["level2_clast_materials"]) and bpy.data.materials.get("GaleTerrainMacroAlbedo") is not None,
            "level2_terrain_material": "GaleTerrainMacroAlbedo",
            "level3_terrain_material": terrain_material_name,
            "prototype_count": len(prototype_names),
        },
        "closeup_target": level2_report["closeup_target"],
        "renders": renders,
        "scope": {"multi_distance_matrix_generated": False},
    }
    errors = validate_level3_report(config, report)
    report["validation"] = {"ok": not errors, "errors": errors}
    report_path = output_dir / "build_L3.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise RuntimeError("Level-3 build report failed: " + "; ".join(errors))
    bpy.context.scene["microterrain_level"] = 3
    bpy.context.scene["microterrain_level3_build_report"] = str(report_path)
    bpy.context.scene["level3_material_signature_sha256"] = contract["material_signature_sha256"]
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    report["blend"] = str(blend_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build(*_arguments())
