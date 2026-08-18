"""Create Level-2 point clouds and Geometry Nodes instance scatter."""

from __future__ import annotations

import bpy
import numpy as np

from src.microterrain.clasts import FAMILIES


SCATTER_COLLECTION = "Microterrain_L2"
SCATTER_OBJECTS = {
    "fine_grains": "ClastScatter_Fine_L2",
    "fragments": "ClastScatter_Fragments_L2",
    "coarse_clasts": "ClastScatter_Coarse_L2",
}


def _remove_existing() -> None:
    collection = bpy.data.collections.get(SCATTER_COLLECTION)
    if collection:
        for obj in list(collection.objects):
            data = obj.data if obj.type == "MESH" else None
            for modifier in list(obj.modifiers):
                group = modifier.node_group if modifier.type == "NODES" else None
                obj.modifiers.remove(modifier)
                if group and group.users == 0:
                    bpy.data.node_groups.remove(group)
            bpy.data.objects.remove(obj, do_unlink=True)
            if data and data.users == 0:
                bpy.data.meshes.remove(data)
        bpy.data.collections.remove(collection)


def _attribute(mesh, name: str, data_type: str, values, property_name: str) -> None:
    attribute = mesh.attributes.new(name=name, type=data_type, domain="POINT")
    attribute.data.foreach_set(property_name, np.asarray(values).ravel())


def _geometry_nodes_group(family: str, source_collection: bpy.types.Collection) -> bpy.types.GeometryNodeTree:
    group = bpy.data.node_groups.new(f"GN_ClastScatter_{family}_L2", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    group_input = nodes.new("NodeGroupInput")
    group_output = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.mode = "VERTICES"
    collection_info = nodes.new("GeometryNodeCollectionInfo")
    collection_info.inputs["Collection"].default_value = source_collection
    if "Separate Children" in collection_info.inputs:
        collection_info.inputs["Separate Children"].default_value = True
    if "Reset Children" in collection_info.inputs:
        collection_info.inputs["Reset Children"].default_value = True
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.inputs["Pick Instance"].default_value = True
    named_rotation = nodes.new("GeometryNodeInputNamedAttribute")
    named_rotation.data_type = "FLOAT_VECTOR"
    named_rotation.inputs["Name"].default_value = "instance_rotation"
    named_scale = nodes.new("GeometryNodeInputNamedAttribute")
    named_scale.data_type = "FLOAT_VECTOR"
    named_scale.inputs["Name"].default_value = "instance_scale"
    named_index = nodes.new("GeometryNodeInputNamedAttribute")
    named_index.data_type = "INT"
    named_index.inputs["Name"].default_value = "prototype_index"
    links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], instance.inputs["Points"])
    links.new(collection_info.outputs["Instances"], instance.inputs["Instance"])
    links.new(named_rotation.outputs["Attribute"], instance.inputs["Rotation"])
    links.new(named_scale.outputs["Attribute"], instance.inputs["Scale"])
    links.new(named_index.outputs["Attribute"], instance.inputs["Instance Index"])
    links.new(instance.outputs["Instances"], group_output.inputs["Geometry"])
    return group


def build_clast_scatter(families: dict, source_collections: dict[str, bpy.types.Collection], metrics: dict) -> dict[str, bpy.types.Object]:
    _remove_existing()
    collection = bpy.data.collections.new(SCATTER_COLLECTION)
    bpy.context.scene.collection.children.link(collection)
    collection["semantic_role"] = "level2_geometry_nodes_clast_instances"
    objects = {}
    for family in FAMILIES:
        values = families[family]
        positions = values["positions"]
        mesh = bpy.data.meshes.new(f"{SCATTER_OBJECTS[family]}_points")
        mesh.vertices.add(len(positions))
        mesh.vertices.foreach_set("co", positions.ravel())
        _attribute(mesh, "instance_rotation", "FLOAT_VECTOR", values["rotations"], "vector")
        _attribute(mesh, "instance_scale", "FLOAT_VECTOR", values["scales"], "vector")
        _attribute(mesh, "prototype_index", "INT", values["prototype_index"], "value")
        _attribute(mesh, "burial_fraction", "FLOAT", values["burial"], "value")
        _attribute(mesh, "characteristic_size_m", "FLOAT", values["characteristic_size_m"], "value")
        _attribute(mesh, "size_class_index", "INT", values["size_class_index"], "value")
        mesh.update()
        obj = bpy.data.objects.new(SCATTER_OBJECTS[family], mesh)
        collection.objects.link(obj)
        modifier = obj.modifiers.new(name=f"GeometryNodes_{family}_L2", type="NODES")
        modifier.node_group = _geometry_nodes_group(family, source_collections[family])
        obj["semantic_role"] = "instanced_geometric_clasts"
        obj["clast_family"] = family
        obj["instance_count"] = len(positions)
        obj["scatter_signature_sha256"] = values["signature_sha256"]
        obj["instances_realized"] = False
        objects[family] = obj
    collection["scatter_signature_sha256"] = metrics["signature_sha256"]
    collection["total_instances"] = metrics["total_instances"]
    return objects
