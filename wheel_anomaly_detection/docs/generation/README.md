# Curiosity Semantic Asset and Dataset Generation

Start with [the generation index](INDEX.md) for the current dataset state,
configuration contracts and milestone documents. The pipeline derives a
six-wheel semantic rover scene from a verified NASA GLB, adds terrain and
imaging variation, and renders clean/paired anomalous observations.

## Required local resources

The original asset, geospatial products, derived scenes and generated
datasets are not distributed with the repository.

- NASA GLB: `24584_Curiosity_static.glb`, stored in `assets/original/`.
- Expected SHA-256:
  `86a8ee6d39fb1711ae549ba99951d2f52b44a7b8367ce4814cc81aec60e94b31`.
- HiRISE Gale DTM and imagery: identifiers and crop requirements are in
  [the terrain milestone](gale_terrain.md).
- Blender 5.2 with Eevee, plus the host packages in
  [generation requirements](../../requirements/generation.txt).

Work on imported or derived copies and preserve the source GLB.
ML packages belong in a host environment separate from Blender Python.

## Semantic asset preparation

Run commands from `wheel_anomaly_detection/`. Make the `blender` executable
available on PATH or substitute its location.

Import the verified GLB into Blender, then save the imported scene as
`outputs/anomaly_detection_2/curiosity_imported.blend`. The preparation script
expects one imported mesh named `MSL`; it does not import the GLB itself.

Prepare the semantic asset from that saved copy:

```bash
blender --background outputs/anomaly_detection_2/curiosity_imported.blend --python-exit-code 1 --python scripts/blender/prepare_curiosity_semantic_asset.py -- --source-glb assets/original/24584_Curiosity_static.glb --output-dir outputs/anomaly_detection_2
```

Validate the result:

```bash
blender --background outputs/anomaly_detection_2/curiosity_semantic_clean.blend --python-exit-code 1 --python scripts/blender/validate_curiosity_semantic_asset.py
```

The build checks the source checksum before and after processing and writes
the derived scene and its reports under `outputs/anomaly_detection_2/`.

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

Collections and transform empties use these semantic names. Body and
suspension remain one mesh: the `Suspension` node identifies the combined
geometry rather than a separate suspension object.

The imported coordinate convention is `+Z` up, `-Y` front and `+X` rover
left. Each wheel is a separate mesh parented to `Wheels`, with its origin
at the world-space bounding-box center.

## Changes from the source GLB

The original `wheels` material identifies 1,944 disconnected components:
six spatial clusters of 324 components. Native Blender separation preserves
face material assignments, three UV layers, packed textures and custom
normals. The remaining mesh becomes `body_suspension_assembly`.

Default scene objects are removed and a semantic hierarchy is added.
The source GLB remains unchanged. Wheel-adjacent fixed parts stay in the
body mesh where separation would be ambiguous.

## Generation stages

| Stage | Contract and reproduction instructions |
| --- | --- |
| Gale terrain and nominal camera/lighting | [Gale terrain](gale_terrain.md) |
| Deterministic 4 x 4 m local relief | [Microterrain Level 1](microterrain_level1.md) |
| Instanced clasts and rocks | [Microterrain Level 2](microterrain_level2.md) |
| Fine-scale material variation | [Microterrain Level 3](microterrain_level3.md) |
| Camera families and wheel roll | [Camera poses](wheel_camera_pose_pilot.md), [pose sampling](wheel_pose_sampling.md) |
| Lighting, wear and randomization | [Lighting](mars_lighting_pilot.md), [surface wear](wheel_surface_wear_pilot.md), [domain randomization](domain_randomization.md) |
| Clean and paired render batches | [Clean generator](clean_batch_generator.md), [hole anomalies](wheel_hole_anomaly.md) |
| Complete dataset plan and assembly | [Bulk plan](bulk_plan_v1.md), [final assembly](final_dataset_assembly.md) |

The current batch source is
`outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend`.
Rebuild the prerequisite stages or supply the matching derived scene before
starting a batch. Render settings come from the selected configuration;
the checked-in clean/paired defaults use 1200 x 900 RGB images.

## Validation

Use the semantic, scene and batch validators for the stage being run.
For completed batches:

```bash
python scripts/host/validate_clean_batch.py --run-dir outputs/anomaly_detection_2/clean_batch/YOUR_RUN_ID
python scripts/host/validate_anomaly_batch.py --run-dir outputs/anomaly_detection_2/anomaly_batch/YOUR_RUN_ID
```

Replace `YOUR_RUN_ID` with an existing run. Validators need the matching
source scene, configurations, manifests and raster outputs. They do not
replace visual QA. The current checkout has no `tests/` directory.

Historical output paths in milestone documents record local provenance;
a fresh Git clone does not include those artifacts. Use the
[index](INDEX.md) to identify the current pipeline.
