# Milestone 5 checkpoint — deterministic microterrain Level 1

This checkpoint adds mesoscopic geometric relief to the immutable Gale
macroterrain scene. It deliberately stops before Level 2 clasts and Level 3
shader-scale grains so that the geometric baseline can be approved in
isolation.

## Scene structure

The Level-1 scene is saved as
`outputs/anomaly_detection_2/microterrain/level1/microterrain_L1.blend` and
contains:

- `GaleTerrainVisual`: the original 257 x 257 macroterrain mesh, unchanged;
- `GaleTerrainVisual_L1`: a runtime copy with a material-space opening under
  the local patch;
- `MicroterrainPatch_L1`: the independently addressable dense terrain patch;
- the semantic rover, nominal solar lighting and existing cameras.

The patch is 4 x 4 m, centered on `wheel_middle_left`, and uses a 2.5 mm regular
grid: 2,563,201 vertices and 2,560,000 quads. Its `GeoreferencedUV` layer maps
the patch into the same macrotexture coordinate system as the HiRISE terrain.
It reuses `GaleTerrainMacroAlbedo`; the Mars-style color and UV placement are
therefore preserved without inventing a second albedo source.

The macro mesh is not cut or resampled. In L1 renders only, the runtime copy
opens the patch footprint through alpha masking and the dense patch occupies
that visual area. L0 renders hide the patch and proxy and show the untouched
macroterrain.

## Deterministic relief contract

One master seed (`42`) is split into SHA-256-derived sub-seeds for large,
medium and fine spectral bands, ripples, shallow depressions and crust-like
flattened regions. The configured wavelengths are 9 cm, 3 cm and 1 cm. This
stage makes no claim below the 2 mm grid sampling limit; sub-millimetric grains
remain deferred to shader Level 3.

The final displacement field has SHA-256
`b5a07ae03c1f5c3cea082bd63edbbe762e2b234e6aaa5acd95f5395a8847c338`.
The verified range is -4.576 to +4.254 mm, RMS 0.993 mm, with no hard-clipped
samples. A 15 cm smooth boundary feather reaches exactly zero displacement on
all patch edges.

Forty-eight shallow depressions and twelve flattened/raised crust regions are
generated for the sixteen-square-metre patch. They are continuous heightfield
features, not explicit rocks or clast objects.

## Rover contact

The rover placement samples the combined macroterrain/patch height at all six
wheel centers. The root is translated vertically only as much as needed to
preserve the configured 1 cm minimum clearance. In the verified build the
minimum clearance is 10.00001 mm and no wheel penetrates the combined terrain.
This is a deterministic placement gate, not a suspension dynamics solver.

## Visual checkpoint

The build can regenerate paired L0/L1 renders with identical lighting and
camera for rover context and terrain-only views at 10 cm and 30 cm. The
historical checkpoint images were removed after Level 1 was promoted; the
derived Blend and validation reports remain authoritative.

The close-up target is selected deterministically as the non-edge 10 cm window
with maximum displacement standard deviation, sampled every 4 cm. The camera
uses 35 degree azimuth and 40 degree elevation to reveal real surface slopes;
no diagnostic bump map or artificial contrast is added.

## Reproducible build and validation

```powershell
python scripts/host/run_microterrain_level1.py `
  --blender-executable "D:\Programmi\Blender Foundation\Blender 5.2\blender.exe"
```

The launcher rebuilds from `gale_terrain_scene.blend`, renders both levels,
saves `build_L1.json`, persists the Blend asset, then opens that saved asset in
a separate Blender process and writes `validation_L1.json`.

The validator regenerates the displacement signature, checks exact topology,
the georeferenced UV layer, the persisted material and patch metadata, the
unchanged macroterrain topology, six-wheel clearance, all render files and the
absence of Level 2/3 content. Pure deterministic logic is covered by
`tests/test_microterrain_core.py`.

## Checkpoint status

Level 1 received visual approval and is the immutable input to the Level-2
build documented in [microterrain_level2.md](microterrain_level2.md).

## Deferred after Level 1

- Level 3 shader-scale grains and micro-normal detail;
- anomaly injection, pose sampling and bulk rendering;
- dynamic wheel/soil interaction or deformable regolith.
