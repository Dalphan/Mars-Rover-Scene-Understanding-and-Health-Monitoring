from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build Milestone 2 multi-scale regolith materials.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-label", default="rover_simulator_m1_camera_contact.blend")
    parser.add_argument("--patches", default="A,B,C", help="Comma-separated patch suffixes to process")
    return parser.parse_args(values)


def node(nodes: bpy.types.Nodes, name: str, node_type: str) -> bpy.types.Node:
    existing = nodes.get(name)
    if existing is not None:
        nodes.remove(existing)
    created = nodes.new(node_type)
    created.name = name
    created.label = name
    return created


def remove_m2_nodes(material: bpy.types.Material, prefix: str) -> None:
    for item in list(material.node_tree.nodes):
        if item.name.startswith(prefix):
            material.node_tree.nodes.remove(item)


def linked_source(socket: bpy.types.NodeSocket) -> bpy.types.NodeSocket | None:
    return socket.links[0].from_socket if socket.links else None


def clear_socket(socket: bpy.types.NodeSocket) -> None:
    for link in list(socket.links):
        socket.id_data.links.remove(link)


def build_noise(nodes: bpy.types.Nodes, links: bpy.types.NodeTreeLinks, name: str, scale: float, detail: float,
                roughness: float, vector_socket: bpy.types.NodeSocket) -> bpy.types.Node:
    noise = node(nodes, name, "ShaderNodeTexNoise")
    noise.noise_dimensions = "3D"
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = roughness
    links.new(vector_socket, noise.inputs["Vector"])
    return noise


def update_material(material: bpy.types.Material, patch_id: str, settings: dict) -> dict:
    if not material.use_nodes:
        material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    principled = nodes.get("Principled BSDF") or next(
        item for item in nodes if item.type == "BSDF_PRINCIPLED"
    )
    prefix = f"{patch_id}_M2_"

    base_source = linked_source(principled.inputs["Base Color"])
    rough_source = linked_source(principled.inputs["Roughness"])
    normal_source = linked_source(principled.inputs["Normal"])
    if base_source is None or rough_source is None or normal_source is None:
        raise RuntimeError(f"{material.name} has no complete pre-M2 terrain shader")

    clear_socket(principled.inputs["Base Color"])
    clear_socket(principled.inputs["Roughness"])
    clear_socket(principled.inputs["Normal"])
    remove_m2_nodes(material, prefix)

    texcoord = node(nodes, prefix + "TextureCoordinate", "ShaderNodeTexCoord")
    mapping = node(nodes, prefix + "Mapping", "ShaderNodeMapping")
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

    macro = build_noise(nodes, links, prefix + "MacroNoise", settings["macro_scale"], settings["macro_detail"], 0.72, mapping.outputs["Vector"])
    meso = build_noise(nodes, links, prefix + "MesoNoise", settings["meso_scale"], settings["meso_detail"], 0.76, mapping.outputs["Vector"])
    micro = build_noise(nodes, links, prefix + "MicroNoise", settings["micro_scale"], settings["micro_detail"], 0.64, mapping.outputs["Vector"])
    granule = build_noise(nodes, links, prefix + "GranuleNoise", settings["granule_scale"], settings["granule_detail"], 0.58, mapping.outputs["Vector"])

    macro_ramp = node(nodes, prefix + "MacroAlbedo", "ShaderNodeValToRGB")
    macro_ramp.color_ramp.elements[0].color = (0.80, 0.48, 0.27, 1.0)
    macro_ramp.color_ramp.elements[1].color = (1.05, 0.82, 0.58, 1.0)
    links.new(macro.outputs["Fac"], macro_ramp.inputs["Fac"])

    meso_ramp = node(nodes, prefix + "MesoAlbedo", "ShaderNodeValToRGB")
    meso_ramp.color_ramp.elements[0].color = (0.84, 0.68, 0.49, 1.0)
    meso_ramp.color_ramp.elements[1].color = (1.03, 0.86, 0.64, 1.0)
    links.new(meso.outputs["Fac"], meso_ramp.inputs["Fac"])

    macro_mix = node(nodes, prefix + "MacroAlbedoMix", "ShaderNodeMixRGB")
    macro_mix.blend_type = "MULTIPLY"
    macro_mix.inputs["Fac"].default_value = float(settings["albedo_mix"])
    links.new(base_source, macro_mix.inputs[1])
    links.new(macro_ramp.outputs["Color"], macro_mix.inputs[2])

    meso_mix = node(nodes, prefix + "MesoAlbedoMix", "ShaderNodeMixRGB")
    meso_mix.blend_type = "MULTIPLY"
    meso_mix.inputs["Fac"].default_value = 0.10
    links.new(macro_mix.outputs["Color"], meso_mix.inputs[1])
    links.new(meso_ramp.outputs["Color"], meso_mix.inputs[2])
    links.new(meso_mix.outputs["Color"], principled.inputs["Base Color"])

    rough_map = node(nodes, prefix + "GranuleRoughness", "ShaderNodeMapRange")
    rough_map.inputs["From Min"].default_value = 0.20
    rough_map.inputs["From Max"].default_value = 0.82
    rough_map.inputs["To Min"].default_value = float(settings["roughness_min"])
    rough_map.inputs["To Max"].default_value = float(settings["roughness_max"])
    rough_map.clamp = True
    links.new(granule.outputs["Fac"], rough_map.inputs["Value"])

    rough_mix = node(nodes, prefix + "GranuleRoughnessMix", "ShaderNodeMixRGB")
    rough_mix.blend_type = "MIX"
    rough_mix.inputs["Fac"].default_value = float(settings["roughness_mix"])
    links.new(rough_source, rough_mix.inputs[1])
    links.new(rough_map.outputs["Result"], rough_mix.inputs[2])
    links.new(rough_mix.outputs["Color"], principled.inputs["Roughness"])

    height_mix = node(nodes, prefix + "GranularHeightMix", "ShaderNodeMixRGB")
    height_mix.blend_type = "MULTIPLY"
    height_mix.inputs["Fac"].default_value = 0.72
    links.new(micro.outputs["Fac"], height_mix.inputs[1])
    links.new(granule.outputs["Fac"], height_mix.inputs[2])

    bump = node(nodes, prefix + "GranularBump", "ShaderNodeBump")
    bump.inputs["Distance"].default_value = float(settings["bump_distance_m"])
    bump.inputs["Strength"].default_value = float(settings["bump_strength"])
    links.new(height_mix.outputs["Color"], bump.inputs["Height"])
    links.new(normal_source, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    material["milestone_2_regolith"] = True
    material["m2_layers"] = "macro_albedo_meso_albedo_granular_roughness_micro_voronoi_bump"
    material["m2_settings"] = json.dumps(settings, sort_keys=True)
    return {
        "material": material.name,
        "nodes": [item.name for item in (macro, meso, micro, granule, rough_map, bump)],
        "settings": settings,
    }


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["terrain_detail"]
    settings = dict(config["regolith_m2"])
    requested = {f"TerrainPatch_{suffix.strip().upper()}" for suffix in args.patches.split(",") if suffix.strip()}
    patches = [item for item in config["patches"] if item["id"] in requested]
    if not patches:
        raise RuntimeError(f"No requested patches found: {sorted(requested)}")

    reports = []
    for patch in patches:
        patch_id = patch["id"]
        material = bpy.data.materials.get(f"{patch_id}_Material")
        patch_obj = bpy.data.objects.get(patch_id)
        if material is None or patch_obj is None:
            raise RuntimeError(f"Missing {patch_id} material or mesh")
        material_report = update_material(material, patch_id, settings)
        reports.append({"patch_id": patch_id, **material_report})
        patch_obj["milestone_2_regolith"] = True
        patch_obj["m2_material_name"] = material.name

    for scene_name in config["scenes"]:
        scene = bpy.data.scenes.get(scene_name)
        if scene is not None:
            scene["terrain_detail_milestone"] = "2-regolith-material"
            scene["milestone_2_regolith"] = True
            scene["collision_enabled"] = False

    root = bpy.data.objects.get(config["rover_root"])
    if root is not None:
        root["milestone_2_regolith"] = True
        root["milestone_2_source_milestone"] = "1-camera-contact"

    bpy.context.view_layer.update()
    output_blend = args.output_blend.expanduser().resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))

    report = {
        "schema_version": "1.0",
        "status": "completed",
        "milestone": "2-regolith-material",
        "source_scene": args.source_label,
        "output_blend": str(output_blend),
        "scope": [item["patch_id"] for item in reports],
        "materials": reports,
        "pair_context_unchanged": True,
        "camera_changed": False,
        "lighting_changed": False,
        "collision_enabled": False,
        "external_resources": [],
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Built Milestone 2 multi-scale regolith variant: {output_blend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
