"""Validate the persisted Level-3 material scene and ablation contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parser.parse_args(argv)
    return args.config.resolve(), args.build_report.resolve(), args.output_report.resolve()


def _upstream_contains_type(material: bpy.types.Material, start_socket, node_type: str) -> bool:
    links = material.node_tree.links
    stack = [link.from_node for link in links if link.to_socket == start_socket]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.type == node_type:
            return True
        for input_socket in node.inputs:
            stack.extend(link.from_node for link in links if link.to_socket == input_socket)
    return False


def validate(config_path: Path, build_report_path: Path, output_report_path: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.blender.microterrain.build_microterrain_material import TERRAIN_L2_MATERIAL, TERRAIN_L3_MATERIAL
    from src.microterrain.level3_validation import validate_level3_report
    from src.microterrain.material import validate_level3_config

    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(build_report_path.read_text(encoding="utf-8"))
    contract = validate_level3_config(config)
    settings = config["microterrain"]["material"]
    errors = validate_level3_report(config, report)
    if int(bpy.context.scene.get("microterrain_level", 0)) != 3:
        errors.append("Persisted scene is not marked Level 3")
    if bpy.context.scene.get("level3_material_signature_sha256") != contract["material_signature_sha256"]:
        errors.append("Persisted scene material signature mismatch")
    patch = bpy.data.objects.get("MicroterrainPatch_L1")
    if patch is None or patch.type != "MESH":
        errors.append("Level-1 patch is missing")
    else:
        if [len(patch.data.vertices), len(patch.data.polygons)] != report["geometry"]["patch_topology"]:
            errors.append("Patch topology changed at Level 3")
        if patch.get("displacement_signature_sha256") != report["geometry"]["level1_displacement_signature_sha256"]:
            errors.append("Level-1 displacement signature changed at Level 3")
        if not patch.data.materials or patch.data.materials[0].name != TERRAIN_L3_MATERIAL:
            errors.append("Patch is not assigned the canonical Level-3 material")

    terrain = bpy.data.materials.get(TERRAIN_L3_MATERIAL)
    if terrain is None or not terrain.use_nodes:
        errors.append("Canonical Level-3 terrain material is missing")
    else:
        if terrain.get("material_signature_sha256") != contract["material_signature_sha256"]:
            errors.append("Terrain material custom signature mismatch")
        nodes = terrain.node_tree.nodes
        noise_nodes = [node for node in nodes if node.type == "TEX_NOISE"]
        if len(noise_nodes) != 6:
            errors.append("Terrain must contain exactly six Level-3 noise bands")
        expected_scales = {
            "L3_CoarseAlbedo": 1.0 / float(settings["coarse_albedo_wavelength_m"]),
            "L3_MediumAlbedo": 1.0 / float(settings["medium_albedo_wavelength_m"]),
            "L3_FineRoughness": 1.0 / float(settings["fine_roughness_wavelength_m"]),
            "L3_VeryFineRoughness": 1.0 / float(settings["very_fine_roughness_wavelength_m"]),
            "L3_MicroBumpPrimary": 1.0 / float(settings["micro_bump_wavelength_m"]),
            "L3_MicroBumpSecondary": 1.0 / float(settings["micro_bump_secondary_wavelength_m"]),
        }
        for name, expected in expected_scales.items():
            node = nodes.get(name)
            if node is None or abs(float(node.inputs["Scale"].default_value) - expected) > max(1e-5, expected * 1e-6):
                errors.append(f"Noise scale mismatch for {name}")
        bump = nodes.get("L3_SubgranularBump")
        if bump is None or abs(float(bump.inputs["Strength"].default_value) - float(settings["micro_bump_strength"])) > 1e-7 or abs(float(bump.inputs["Distance"].default_value) - float(settings["micro_bump_distance_m"])) > 1e-9:
            errors.append("Terrain micro-bump parameters mismatch")
        principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled is None or not principled.inputs["Base Color"].is_linked or not principled.inputs["Roughness"].is_linked or not principled.inputs["Normal"].is_linked:
            errors.append("Terrain Level-3 channels are not fully connected")
        elif not _upstream_contains_type(terrain, principled.inputs["Base Color"], "TEX_IMAGE"):
            errors.append("HiRISE image is not upstream of the Level-3 base color")

    level2_terrain = bpy.data.materials.get(TERRAIN_L2_MATERIAL)
    if level2_terrain is None or not any(node.type == "TEX_IMAGE" for node in level2_terrain.node_tree.nodes):
        errors.append("Level-2 HiRISE material was not preserved")
    clast_material_names = report["material"]["clast_materials"]
    for name in clast_material_names:
        material = bpy.data.materials.get(name)
        if material is None:
            errors.append(f"Missing Level-3 clast material {name}")
            continue
        if material.get("material_signature_sha256") != contract["material_signature_sha256"]:
            errors.append(f"Clast material signature mismatch for {name}")
        noise_count = sum(node.type == "TEX_NOISE" for node in material.node_tree.nodes)
        bump = material.node_tree.nodes.get("L3_ClastSubgranularBump")
        if noise_count != 3 or bump is None:
            errors.append(f"Incomplete multiscale clast shader for {name}")
        elif abs(float(bump.inputs["Distance"].default_value) - float(settings["clast_bump_distance_m"])) > 1e-9:
            errors.append(f"Clast bump distance mismatch for {name}")

    level2_report = json.loads(Path(report["level2"]["report"]).read_text(encoding="utf-8"))
    family_signatures = {}
    for family, object_name in level2_report["instancing"]["scatter_objects"].items():
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            errors.append(f"Missing Level-2 scatter object {object_name}")
            continue
        family_signatures[family] = obj.get("scatter_signature_sha256")
        if len(obj.data.vertices) != level2_report["clasts"]["counts"][family]:
            errors.append(f"Scatter point count changed for {family}")
        for modifier in (modifier for modifier in obj.modifiers if modifier.type == "NODES" and modifier.node_group):
            if any(node.bl_idname == "GeometryNodeRealizeInstances" for node in modifier.node_group.nodes):
                errors.append(f"Instances were realized for {family}")
    if family_signatures != level2_report["clasts"]["family_signatures_sha256"]:
        errors.append("Persisted Level-2 family signatures changed")
    prototype_names = [name for names in level2_report["library"]["families"].values() for name in names]
    for name in prototype_names:
        obj = bpy.data.objects.get(name)
        if obj is None or not obj.get("level2_material_name") or not obj.get("level3_material_name"):
            errors.append(f"Prototype material ablation metadata missing for {name}")
            continue
        if obj.data.materials[0].name != obj["level3_material_name"]:
            errors.append(f"Prototype {name} is not in native Level-3 state")
        if bpy.data.materials.get(obj["level2_material_name"]) is None:
            errors.append(f"Prototype {name} lost its Level-2 material")
    for path in report["renders"].values():
        if not Path(path).is_file():
            errors.append(f"Missing render file: {path}")
    result = {
        "ok": not errors,
        "level": 3,
        "blend": bpy.data.filepath,
        "errors": errors,
        "material_signature_sha256": contract["material_signature_sha256"],
        "terrain_noise_layers": 0 if terrain is None else sum(node.type == "TEX_NOISE" for node in terrain.node_tree.nodes),
        "clast_material_count": len(clast_material_names),
        "level2_scatter_signature_sha256": report["level2"]["scatter_signature_sha256"],
        "geometry_unchanged": not any("changed" in error.lower() or "realized" in error.lower() for error in errors),
        "scope": {"multi_distance_matrix_generated": False},
        "renders": report["renders"],
    }
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError("Microterrain Level 3 validation failed: " + "; ".join(errors))
    return result


if __name__ == "__main__":
    validate(*_arguments())
