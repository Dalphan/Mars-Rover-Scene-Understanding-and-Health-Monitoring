# Curiosity semantic asset — anomaly detection 2

Per orientarsi tra tutte le milestone, consultare prima
[INDEX.md](INDEX.md), che indirizza al documento tecnico corretto per ogni
argomento.

This milestone starts from the verified NASA file
`24584_Curiosity_static.glb` and creates a minimally modified Blender asset in
`outputs/anomaly_detection_2/`. Generated `.blend`, `.glb`, reports and renders
remain ignored runtime output.

## Scene structure

```text
Rover
├── Body
│   └── body_suspension_assembly
├── Suspension
│   └── geometry_object = body_suspension_assembly
└── Wheels
    ├── wheel_front_left
    ├── wheel_front_right
    ├── wheel_middle_left
    ├── wheel_middle_right
    ├── wheel_rear_left
    └── wheel_rear_right
```

`Rover`, `Body`, `Suspension`, and `Wheels` are empty transform objects. The
same names are also used for collections. The `Suspension` empty explicitly
points to `body_suspension_assembly`: suspension and rocker-bogie geometry is
identified but intentionally not separated from the remaining rover mesh.

## Naming and axes

Wheel names are exact and stable. The imported model uses world `+Z` as up,
negative world `Y` as the rover front, and positive world `X` as the rover's
left side when facing forward. Every wheel is a distinct mesh object parented
to `Wheels`; its origin is the center of its world-space bounding box.

## Changes from the NASA GLB

- Removed only the default Blender cube, camera, and light.
- Used the original `wheels` material as the semantic gate. It contains 1,944
  disconnected components that form six unambiguous spatial clusters of 324
  components each.
- Separated the six clusters with Blender's native mesh separation operation.
  Meshes were not reconstructed; original face material assignments, three UV
  layers, packed textures, and custom normals are retained.
- Renamed the remaining imported mesh to `body_suspension_assembly` and added a
  small, predictable semantic hierarchy.
- Did not alter the source GLB. Its SHA-256 is checked before and after the
  build.

## Validation

Open `outputs/anomaly_detection_2/curiosity_semantic_clean.blend` and run:

```powershell
blender --background curiosity_semantic_clean.blend --python scripts/blender/validate_curiosity_semantic_asset.py
```

The validator requires exactly the six canonical wheel names, prints each
wheel's location, Euler rotation, scale, world bounding box and mesh counts,
and verifies the `wheels` material, UV layers and `Wheels` parent.

## Known limitations

- The original rover is a single highly fragmented mesh. Body, rocker-bogie,
  suspension and wheel-adjacent fixed parts remain combined in
  `body_suspension_assembly`; this avoids a destructive or visually ambiguous
  heuristic split. They are identified by object metadata and the
  `Suspension` semantic node.
- The semantic-asset milestone itself does not alter cameras, lighting or
  terrain; those are layered on by the separate Gale milestone below.

## Terrain milestones

The deterministic HiRISE Gale terrain, approximate MAHLI camera and nominal
solar lighting are documented in [gale_terrain.md](gale_terrain.md).

The deterministic two-metre Level-1 microterrain patch, its generation
contract and the 10/30 cm visual checkpoint are documented in
[microterrain_level1.md](microterrain_level1.md).

The approved Level-2 checkpoint adds three families of non-realized,
partially buried Geometry Nodes clasts and is documented in
[microterrain_level2.md](microterrain_level2.md).

The Level-3 checkpoint adds sub-granular dust, albedo, roughness and bump
shading while preserving every L2 geometry signature. It is documented in
[microterrain_level3.md](microterrain_level3.md).

The deterministic clean dataset factory, its one-render RGB/target-mask
contract, resume behavior and verified 24-sample smoke run are documented in
[clean_batch_generator.md](clean_batch_generator.md).
