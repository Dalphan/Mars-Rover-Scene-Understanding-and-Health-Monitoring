# Milestone 4 — Gale terrain, approximate MAHLI and nominal lighting

This milestone builds a deterministic, georeferenced baseline from one HiRISE
stereo product. Raw products are intentionally stored outside the repository
under `D:\Daniele\Scuola\Magistrale\Tesi\3d models\gale_hirise`.

## Source and crop

- DTM: `DTEEC_010573_1755_010639_1755_U01.IMG`.
- Geometric/color reference: `PSP_010639_1755_IRB_A_01_ORTHO.JP2`
  (three-band IRB false-color, native scale about 0.25294 m/px).
- Candidate visual RGB: `PSP_010639_1755_MRGB.JP2` (three-band MRGB,
  native scale 0.5 m/px).
- Stereo observations: `PSP_010573_1755` and `PSP_010639_1755`.
- Projection: Mars equirectangular, planetocentric, central meridian 137.41°E,
  spherical radius 3,396,190 m.
- Projected crop bounds: `[-366, -266538, -110, -266282]` metres.
- Approximate crop center: 4.49450°S, 137.40598°E.
- Crop size: 256 × 256 m.
- DTM grid: 256 × 256 cells at exactly 1 m/cell.
- Color grids: 1024 × 1024 cells at exactly 0.25 m/cell.
- Height range: -4453.089 to -4447.989 m; no vertical exaggeration.

DTM and IRB use the same explicit projected bounding box. Their WKT datum
names differ, but the projection method, radius, standard latitude, central
meridian, false origins and metric units are numerically identical. The report
records that comparison instead of accepting the CRS based only on a name.
All crops are generated from this projected bounding box; there are no
hard-coded source pixel windows.

The MRGB product is north-up and covers the crop after its official CRS is
transformed from central meridian 180° to the DTM/IRB CRS at 137.41°. It is a
*merged* product: the narrow central RGB swath is embedded in the wider RED
mosaic. Only `42.48%` of this crop contains actual color according to the
configured chroma gate; the remaining part is grayscale. Coverage alone is
therefore not treated as usable color or proof of coregistration. After
standards-based reprojection to the common one-metre comparison grid, the
structural correlation is also only `0.0220` (required `0.2`) and its best
apparent shift is `[1, 8]` m (allowed at most 1 m on either axis). MRGB fails
the deterministic gate and no empirical translation is applied.

Directly displaying IRB produces a yellow surface because its bands are
near-infrared, red and blue-green mapped to display red, green and blue. HiRISE
documentation explicitly notes that dusty terrain generally appears yellow in
IRB. This is scientifically useful false color, but not the desired Mars-like
visual baseline.

The final texture keeps IRB as the only spatial source and produces a
deterministic Mars-style RGB visualization. It first applies the documented
HiRISE synthetic RGB relation `R=RED`, `G=BG`, `B=2×BG−0.3×RED` to calibrated
I/F. A shared luminance preserves the coregistered IRB morphology without
per-channel color noise. Only the robust 2nd–98th percentile *color range* of
the valid MRGB color swath is used to set the reddish-brown/tan palette; no MRGB
pixel contributes spatial structure. The report records
`mrgb_spatial_data_used=false`.

`irb_color.png`, `irb_reflectance_srgb.png`, `irb_mars_rgb.png` and
`mrgb_candidate.png` remain separate audit products. `terrain_color.png` is
the selected Mars-style result. These are macro color products, not
MAHLI-scale soil texture.

## Blender scene

`GaleTerrainVisual` is a 257 × 257 point heightfield with metric local axes:

- Blender X = projected easting;
- Blender Y = projected northing;
- Blender Z = DTM elevation relative to the crop-center elevation;
- UV U = normalized easting;
- UV V = normalized northing.

`GaleTerrainCollision` is a separate 65 × 65 point mesh at 4 m spacing, has no
material and is hidden from rendering. The semantic rover is loaded from
`curiosity_semantic_clean.blend`. A single deterministic healthy pose aligns
the rover with a least-squares plane through the six wheel-center terrain
samples, then applies the configured 1 cm minimum clearance. This is not a
dynamic collision solver or pose sampler.

The approximate MAHLI preset uses 1600 × 1200, 20 mm focal length, 13.2 mm
sensor width and 36.53° horizontal FOV. It targets `wheel_middle_left`; distance,
azimuth and elevation remain configuration values rather than a fixed camera
transform.

Nominal lighting contains one Blender Sun plus a weak World ambient term. Solar
azimuth 25.5° and elevation 34° come from the HiRISE observation metadata. Sun
energy 4.5 and ambient strength 0.18 are rendering parameters, not radiometric
reconstructions of the Martian atmosphere.

## Reproducible run

Run the launcher with the bundled scientific Python environment:

```powershell
python scripts/host/run_gale_terrain.py `
  --blender-executable "D:\Programmi\Blender Foundation\Blender 5.2\blender.exe" `
  --data-root "D:\Daniele\Scuola\Magistrale\Tesi\3d models\gale_hirise" `
  --deps-dir outputs\anomaly_detection_2\python_deps
```

The launcher verifies/downloads sources, rebuilds the crop, creates the Blender
scene and runs the persisted-scene validator. Runtime outputs are under
`outputs/anomaly_detection_2/gale_terrain/`.

`geospatial/metadata.json` is the authoritative JSON report. It records every
source path and SHA-256, native CRS, transform, orientation, pixel size,
extent, raster dimensions and band type, plus the shared crop bounds, validity
fractions, MRGB test metrics and final texture selection. Original HiRISE files
are read-only inputs under `3d models`; all GeoTIFF, PNG, NPZ, JSON, render and
Blend derivatives remain inside the milestone output directory.

## Scope intentionally deferred to the microterrain stages

- The immutable macroterrain scene contains no procedural micro-regolith,
  rocks or centimetric roughness. Level 1 is layered non-destructively and is
  documented in [microterrain_level1.md](microterrain_level1.md).
- No camera/light randomization or bulk dataset generation.
- No dynamic suspension or collision-based pose solver.
- No claim that 1 m DTM or 25 cm ortho resolves MAHLI-scale surface detail.
