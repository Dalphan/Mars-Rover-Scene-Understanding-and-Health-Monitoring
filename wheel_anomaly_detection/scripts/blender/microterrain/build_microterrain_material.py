"""Build non-destructive Level-3 terrain and clast materials."""

from __future__ import annotations

import bpy


TERRAIN_L2_MATERIAL = "GaleTerrainMacroAlbedo"
TERRAIN_L3_MATERIAL = "GaleTerrainMicroterrain_L3"
CLAST_L2_PREFIX = "Clast_L2_"
CLAST_L3_PREFIX = "Clast_L3_"


def _remove_material(name: str) -> None:
    material = bpy.data.materials.get(name)
    if material:
        bpy.data.materials.remove(material)


def _noise(nodes, geometry, name: str, wavelength: float, detail: float, roughness: float):
    node = nodes.new("ShaderNodeTexNoise")
    node.name = name
    node.noise_dimensions = "3D"
    node.inputs["Scale"].default_value = 1.0 / float(wavelength)
    node.inputs["Detail"].default_value = float(detail)
    node.inputs["Roughness"].default_value = float(roughness)
    node.inputs["Distortion"].default_value = 0.0
    node.id_data.links.new(geometry.outputs["Position"], node.inputs["Vector"])
    return node


def _grayscale_ramp(nodes, name: str, minimum: float, maximum: float):
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = name
    ramp.color_ramp.elements[0].color = (minimum, minimum, minimum, 1.0)
    ramp.color_ramp.elements[1].color = (maximum, maximum, maximum, 1.0)
    return ramp


def _terrain_material(config: dict, signature: str) -> bpy.types.Material:
    settings = config["microterrain"]["material"]
    _remove_material(TERRAIN_L3_MATERIAL)
    source = bpy.data.materials[TERRAIN_L2_MATERIAL]
    material = source.copy()
    material.name = TERRAIN_L3_MATERIAL
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    base_link = next((link for link in links if link.to_node == principled and link.to_socket.name == "Base Color"), None)
    if base_link is None:
        raise RuntimeError("Level-2 terrain material has no linked HiRISE base color")
    base_color_socket = base_link.from_socket
    links.remove(base_link)
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.name = "L3MetricPosition"
    coarse = _noise(nodes, geometry, "L3_CoarseAlbedo", settings["coarse_albedo_wavelength_m"], 4.0, 0.62)
    medium = _noise(nodes, geometry, "L3_MediumAlbedo", settings["medium_albedo_wavelength_m"], 3.0, 0.58)
    fine_roughness = _noise(nodes, geometry, "L3_FineRoughness", settings["fine_roughness_wavelength_m"], 3.0, 0.68)
    very_fine_roughness = _noise(nodes, geometry, "L3_VeryFineRoughness", settings["very_fine_roughness_wavelength_m"], 2.0, 0.72)
    bump_primary = _noise(nodes, geometry, "L3_MicroBumpPrimary", settings["micro_bump_wavelength_m"], 3.0, 0.70)
    bump_secondary = _noise(nodes, geometry, "L3_MicroBumpSecondary", settings["micro_bump_secondary_wavelength_m"], 2.0, 0.74)

    coarse_strength = float(settings["coarse_albedo_strength"])
    coarse_ramp = _grayscale_ramp(nodes, "L3_CoarseAlbedoRamp", 1.0 - coarse_strength, 1.0 + 0.55 * coarse_strength)
    links.new(coarse.outputs["Fac"], coarse_ramp.inputs["Fac"])
    coarse_multiply = nodes.new("ShaderNodeMixRGB")
    coarse_multiply.name = "L3_PreserveHiRISE_Coarse"
    coarse_multiply.blend_type = "MULTIPLY"
    coarse_multiply.inputs[0].default_value = 1.0
    links.new(base_color_socket, coarse_multiply.inputs[1])
    links.new(coarse_ramp.outputs["Color"], coarse_multiply.inputs[2])
    medium_strength = float(settings["medium_albedo_strength"])
    medium_ramp = _grayscale_ramp(nodes, "L3_MediumAlbedoRamp", 1.0 - medium_strength, 1.0 + medium_strength)
    links.new(medium.outputs["Fac"], medium_ramp.inputs["Fac"])
    medium_multiply = nodes.new("ShaderNodeMixRGB")
    medium_multiply.name = "L3_PreserveHiRISE_Medium"
    medium_multiply.blend_type = "MULTIPLY"
    medium_multiply.inputs[0].default_value = 1.0
    links.new(coarse_multiply.outputs["Color"], medium_multiply.inputs[1])
    links.new(medium_ramp.outputs["Color"], medium_multiply.inputs[2])
    dust_factor = nodes.new("ShaderNodeMath")
    dust_factor.name = "L3_DustMask"
    dust_factor.operation = "MULTIPLY"
    dust_factor.inputs[1].default_value = float(settings["dust_amount"])
    links.new(coarse.outputs["Fac"], dust_factor.inputs[0])
    dust_mix = nodes.new("ShaderNodeMixRGB")
    dust_mix.name = "L3_DustAlbedo"
    dust_mix.blend_type = "MIX"
    dust_mix.inputs[2].default_value = tuple(map(float, settings["dust_color"]))
    links.new(dust_factor.outputs[0], dust_mix.inputs[0])
    links.new(medium_multiply.outputs["Color"], dust_mix.inputs[1])
    links.new(dust_mix.outputs["Color"], principled.inputs["Base Color"])

    roughness_add = nodes.new("ShaderNodeMath")
    roughness_add.name = "L3_RoughnessMultiscaleAdd"
    roughness_add.operation = "ADD"
    links.new(fine_roughness.outputs["Fac"], roughness_add.inputs[0])
    links.new(very_fine_roughness.outputs["Fac"], roughness_add.inputs[1])
    roughness_average = nodes.new("ShaderNodeMath")
    roughness_average.name = "L3_RoughnessAverage"
    roughness_average.operation = "MULTIPLY"
    roughness_average.inputs[1].default_value = 0.5
    links.new(roughness_add.outputs[0], roughness_average.inputs[0])
    roughness_map = nodes.new("ShaderNodeMapRange")
    roughness_map.name = "L3_DustRoughness"
    variation = float(settings["roughness_variation"])
    roughness_map.inputs["From Min"].default_value = 0.0
    roughness_map.inputs["From Max"].default_value = 1.0
    roughness_map.inputs["To Min"].default_value = max(0.0, float(settings["dust_roughness"]) - variation)
    roughness_map.inputs["To Max"].default_value = min(1.0, float(settings["dust_roughness"]) + 0.5 * variation)
    links.new(roughness_average.outputs[0], roughness_map.inputs["Value"])
    links.new(roughness_map.outputs["Result"], principled.inputs["Roughness"])

    primary_weight = nodes.new("ShaderNodeMath")
    primary_weight.name = "L3_BumpPrimaryWeight"
    primary_weight.operation = "MULTIPLY"
    primary_weight.inputs[1].default_value = 0.72
    links.new(bump_primary.outputs["Fac"], primary_weight.inputs[0])
    secondary_weight = nodes.new("ShaderNodeMath")
    secondary_weight.name = "L3_BumpSecondaryWeight"
    secondary_weight.operation = "MULTIPLY"
    secondary_weight.inputs[1].default_value = 0.28
    links.new(bump_secondary.outputs["Fac"], secondary_weight.inputs[0])
    bump_add = nodes.new("ShaderNodeMath")
    bump_add.name = "L3_BumpMultiscaleAdd"
    bump_add.operation = "ADD"
    links.new(primary_weight.outputs[0], bump_add.inputs[0])
    links.new(secondary_weight.outputs[0], bump_add.inputs[1])
    bump = nodes.new("ShaderNodeBump")
    bump.name = "L3_SubgranularBump"
    bump.inputs["Strength"].default_value = float(settings["micro_bump_strength"])
    bump.inputs["Distance"].default_value = float(settings["micro_bump_distance_m"])
    links.new(bump_add.outputs[0], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    material["microterrain_level"] = 3
    material["material_signature_sha256"] = signature
    material["source_level2_material"] = source.name
    material["spatial_source_preserved"] = "HiRISE terrain_color.png through Level-2 Image Texture"
    return material


def _clast_material(source: bpy.types.Material, settings: dict, signature: str) -> bpy.types.Material:
    target_name = source.name.replace(CLAST_L2_PREFIX, CLAST_L3_PREFIX, 1)
    _remove_material(target_name)
    material = source.copy()
    material.name = target_name
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = next(node for node in nodes if node.type == "BSDF_PRINCIPLED")
    base_color = tuple(principled.inputs["Base Color"].default_value)
    base_roughness = float(principled.inputs["Roughness"].default_value)
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.name = "L3_ClastMetricPosition"
    dust_noise = _noise(nodes, geometry, "L3_ClastDust", 0.0035, 3.0, 0.62)
    roughness_noise = _noise(nodes, geometry, "L3_ClastRoughness", 0.0008, 2.0, 0.68)
    bump_noise = _noise(nodes, geometry, "L3_ClastBumpNoise", settings["clast_bump_wavelength_m"], 3.0, 0.72)
    dust_factor = nodes.new("ShaderNodeMath")
    dust_factor.name = "L3_ClastDustFactor"
    dust_factor.operation = "MULTIPLY"
    dust_factor.inputs[1].default_value = float(settings["dust_amount"]) * float(settings["clast_dust_fraction"])
    links.new(dust_noise.outputs["Fac"], dust_factor.inputs[0])
    dust_mix = nodes.new("ShaderNodeMixRGB")
    dust_mix.name = "L3_ClastDustAlbedo"
    dust_mix.inputs[1].default_value = base_color
    dust_mix.inputs[2].default_value = tuple(map(float, settings["dust_color"]))
    links.new(dust_factor.outputs[0], dust_mix.inputs[0])
    links.new(dust_mix.outputs["Color"], principled.inputs["Base Color"])
    roughness_map = nodes.new("ShaderNodeMapRange")
    roughness_map.name = "L3_ClastRoughnessMap"
    variation = float(settings["clast_roughness_variation"])
    roughness_map.inputs["To Min"].default_value = max(0.0, base_roughness - variation)
    roughness_map.inputs["To Max"].default_value = min(1.0, base_roughness + variation)
    links.new(roughness_noise.outputs["Fac"], roughness_map.inputs["Value"])
    links.new(roughness_map.outputs["Result"], principled.inputs["Roughness"])
    bump = nodes.new("ShaderNodeBump")
    bump.name = "L3_ClastSubgranularBump"
    bump.inputs["Strength"].default_value = float(settings["clast_bump_strength"])
    bump.inputs["Distance"].default_value = float(settings["clast_bump_distance_m"])
    links.new(bump_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    material["microterrain_level"] = 3
    material["material_signature_sha256"] = signature
    material["source_level2_material"] = source.name
    return material


def build_level3_materials(config: dict, signature: str, prototype_names: list[str]) -> tuple[bpy.types.Material, dict[str, bpy.types.Material], dict]:
    terrain = _terrain_material(config, signature)
    settings = config["microterrain"]["material"]
    level2_materials = sorted({bpy.data.objects[name].data.materials[0].name for name in prototype_names})
    for name in level2_materials:
        bpy.data.materials[name].use_fake_user = True
    mapping = {name: _clast_material(bpy.data.materials[name], settings, signature) for name in level2_materials}
    for name in prototype_names:
        obj = bpy.data.objects[name]
        source_name = obj.data.materials[0].name
        obj["level2_material_name"] = source_name
        obj["level3_material_name"] = mapping[source_name].name
        obj.data.materials[0] = mapping[source_name]
    patch = bpy.data.objects["MicroterrainPatch_L1"]
    patch["level2_material_name"] = TERRAIN_L2_MATERIAL
    patch["level3_material_name"] = terrain.name
    patch.data.materials[0] = terrain
    metrics = {
        "terrain_material": terrain.name,
        "terrain_noise_layers": 6,
        "clast_material_count": len(mapping),
        "clast_materials": [material.name for material in mapping.values()],
        "level2_clast_materials": level2_materials,
        "signature_sha256": signature,
    }
    return terrain, mapping, metrics
