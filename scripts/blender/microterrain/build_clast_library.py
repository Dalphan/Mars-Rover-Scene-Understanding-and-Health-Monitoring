"""Build a compact procedural source library for Level-2 clast instances."""

from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np

from src.microterrain.clasts import FAMILIES
from src.microterrain.core import derive_seed


LIBRARY_COLLECTION = "ClastLibrary_L2"
FAMILY_COLLECTIONS = {
    "fine_grains": "ClastLibrary_Rounded",
    "fragments": "ClastLibrary_Angular",
    "coarse_clasts": "ClastLibrary_Coarse",
}


def _remove_collection(name: str) -> None:
    collection = bpy.data.collections.get(name)
    if not collection:
        return
    for child in list(collection.children):
        _remove_collection(child.name)
    for obj in list(collection.objects):
        data = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0:
            bpy.data.meshes.remove(data)
    bpy.data.collections.remove(collection)


def _material(entry: dict) -> bpy.types.Material:
    name = f"Clast_L2_{entry['name']}"
    existing = bpy.data.materials.get(name)
    if existing:
        bpy.data.materials.remove(existing)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = tuple(map(float, entry["base_color"]))
    principled.inputs["Roughness"].default_value = float(entry["roughness"])
    principled.inputs["Metallic"].default_value = 0.0
    material["semantic_role"] = "martian_clast_material"
    return material


def _prototype_mesh(family: str, variant: int, seed: int) -> bpy.types.Mesh:
    rng = np.random.default_rng(seed)
    # Coarse rocks can fill a large part of the 10 cm validation frame. One
    # extra icosphere subdivision avoids visibly synthetic low-poly facets
    # while preserving an angular, non-smoothed silhouette.
    subdivisions = 2 if family == "fine_grains" else (1 if family == "fragments" else 3)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=0.5)
    shear = rng.uniform(-0.18, 0.18, size=3)
    if family == "fine_grains":
        base_aspect = rng.uniform([0.84, 0.84, 0.78], [1.12, 1.12, 1.05])
        perturbation = 0.045
    elif family == "fragments":
        base_aspect = rng.uniform([0.62, 0.58, 0.48], [1.35, 1.28, 1.0])
        perturbation = 0.20
    else:
        base_aspect = rng.uniform([0.70, 0.64, 0.52], [1.40, 1.30, 1.08])
        perturbation = 0.13
    for vertex in bm.verts:
        co = np.array(vertex.co, dtype=np.float64)
        radial = 1.0 + rng.uniform(-perturbation, perturbation)
        co *= radial * base_aspect
        co = np.array((co[0] + shear[0] * co[2], co[1] + shear[1] * co[0], co[2] + shear[2] * co[1]))
        vertex.co = co
    dimensions = np.array([
        max(v.co.x for v in bm.verts) - min(v.co.x for v in bm.verts),
        max(v.co.y for v in bm.verts) - min(v.co.y for v in bm.verts),
        max(v.co.z for v in bm.verts) - min(v.co.z for v in bm.verts),
    ])
    normalise = 1.0 / float(dimensions.max())
    for vertex in bm.verts:
        vertex.co *= normalise
    mesh = bpy.data.meshes.new(f"clast_{family}_{variant:02d}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    for polygon in mesh.polygons:
        polygon.use_smooth = family == "fine_grains"
    mesh.update(calc_edges=True)
    return mesh


def build_clast_library(config: dict) -> tuple[dict[str, bpy.types.Collection], dict]:
    _remove_collection(LIBRARY_COLLECTION)
    library = bpy.data.collections.new(LIBRARY_COLLECTION)
    library["semantic_role"] = "hidden_source_clast_meshes"
    palette = [_material(entry) for entry in config["microterrain"]["clasts"]["material_palette"]]
    configured_variants = config["microterrain"]["clasts"].get("library_variants")
    if configured_variants is None:
        shared = int(config["microterrain"]["clasts"]["library_variants_per_family"])
        configured_variants = {family: shared for family in FAMILIES}
    variants = {family: int(configured_variants[family]) for family in FAMILIES}
    master_seed = int(config["microterrain"]["seed"])
    collections = {}
    metrics = {"source_mesh_count": 0, "variants_per_family": variants, "families": {}, "materials": [material.name for material in palette]}
    for family_index, family in enumerate(FAMILIES):
        collection = bpy.data.collections.new(FAMILY_COLLECTIONS[family])
        library.children.link(collection)
        collections[family] = collection
        names = []
        for variant in range(variants[family]):
            seed = derive_seed(master_seed, f"clast_prototype:{family}:{variant}")
            mesh = _prototype_mesh(family, variant, seed)
            obj = bpy.data.objects.new(f"clast_{family}_{variant:02d}", mesh)
            collection.objects.link(obj)
            obj.data.materials.append(palette[(variant + family_index) % len(palette)])
            obj["clast_family"] = family
            obj["prototype_variant"] = variant
            obj["prototype_seed"] = str(seed)
            obj["unit_characteristic_size_m"] = 1.0
            names.append(obj.name)
        metrics["families"][family] = names
        metrics["source_mesh_count"] += len(names)
    return collections, metrics
