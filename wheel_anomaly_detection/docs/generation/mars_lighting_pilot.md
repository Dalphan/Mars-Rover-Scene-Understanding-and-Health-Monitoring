# Mars lighting pilot

This non-destructive pilot renders one identical wheel frame under three
initial appearance presets and two refinements. It uses `wheel_middle_left`,
camera family A, roll zero and 1200 x 900 output for all variants. Camera and
wheel world matrices are compared numerically and must remain identical.

## Variants

- `current`: the existing nominal Gale Sun/World, current terrain grade and
  temporary 220 W warm camera fill.
- `mars_clear`: 60-degree solar elevation, 0.40-degree solar disc, warm direct
  illumination, stronger diffuse dusty sky, no artificial fill, AgX Medium Low
  Contrast.
- `mars_dusty_reference`: the clear setup plus a procedural deposited-dust
  layer on all rover materials and a restrained warm camera response.
- `mars_clear_refined`: the clear setup with a stronger diffuse sky and a
  restrained +0.45 EV lift to recover wheel-shadow detail.
- `mars_dusty_refined`: a lighter deposited-dust layer, less camera veil and a
  +0.30 EV lift. This preserves the dusty appearance without turning the whole
  frame into an orange filter.

The dust shader combines world-up surface orientation with 3D patch noise. It
mixes the underlying texture toward a terrain-derived dielectric layer, raises
roughness, reduces metallic response and introduces a 0.7 mm-scale micro bump.
All changes exist only during rendering; the pose-sampling Blend is never
saved.

## Automatic checks

- camera matrix maximum delta across presets: 0;
- target-wheel matrix maximum delta across presets: 0;
- source Blend checksum verified before and after rendering;
- target wheel remains fully inside frame;
- output report includes illumination, material, color-management and image
  histogram metadata.

The reference photo has different content and cannot be matched by a global
histogram alone: the synthetic frame contains a large dark wheel cavity. The
metrics are therefore diagnostic, not an optimization objective. Relative to
their initial versions, the refinements raise median luminance from 37.77 to
47.06 (`clear`) and from 60.75 to 67.75 (`dusty`) while retaining directional
shadows. Their slightly lower median saturation is intentional: it reduces
the risk of encoding a global orange cast instead of a physical dust deposit.

## Outputs

The retained report is
`outputs/anomaly_detection_2/lighting_pilot/lighting_pilot.json`. Historical
comparison renders were removed after the refined presets were promoted into
the production configuration; the command below regenerates them when needed.

```powershell
python scripts/host/run_mars_lighting_pilot.py `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe' `
  --reference-image '<optional-reference.jpg>'
```

Visual selection approved on 10 August 2026:

- primary realistic preset: `mars_dusty_refined`;
- dataset domain-variation preset: `mars_clear_refined`.

The original variants remain available as immutable visual baselines. Before
bulk generation, both selected presets must also receive an anomaly-to-wheel
photometric contrast gate so that dust or camera veil cannot make a
geometrically visible anomaly unusable.
