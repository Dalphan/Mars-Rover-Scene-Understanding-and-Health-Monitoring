# Milestone 5 checkpoint — deterministic microterrain Level 2

Level 2 starts exclusively from `microterrain_L1.blend`. It adds discrete
geometric grains and clasts without changing the Level-1 patch, HiRISE
macroterrain, rover pose, camera or nominal lighting. Level-3 shader detail
remains disabled in the L2 asset.

## Scene organization and performance

The persisted asset is
`outputs/anomaly_detection_2/microterrain/level2/microterrain_L2.blend`.
Three point-mesh objects live in `Microterrain_L2`:

- `ClastScatter_Fine_L2` — rounded grains;
- `ClastScatter_Fragments_L2` — angular lithic fragments;
- `ClastScatter_Coarse_L2` — coarse rocks with explicit size classes.

Each object has one Geometry Nodes modifier using `Mesh to Points`,
`Collection Info` and `Instance on Points`; there is no `Realize Instances`
node. The scene therefore stores 22,080 instances as 22,080 point vertices.

The hidden source library contains six deterministic fine variants, six
fragment variants and eight coarse variants: 20 meshes total. Coarse sources
use an extra icosphere subdivision because rocks can occupy much of a 10 cm
validation frame. They remain irregular, asymmetric and flat-shaded rather
than uniformly scaled spheres.

## Metric populations

For the 4 × 4 m patch, the fine and fragment densities are intentionally
reduced while coarse-rock density is preserved:

| Family | Density | Count | Characteristic size |
| --- | ---: | ---: | ---: |
| rounded fine grains | 1,200 /m² | 19,200 | 0.5–2 mm |
| angular fragments | 150 /m² | 2,400 | 1–7 mm |
| coarse clasts | 30 /m² | 480 | 6–150 mm |

The coarse population is an exact deterministic long-tail mixture:

| Coarse class | Count | Fraction | Size | Burial |
| --- | ---: | ---: | ---: | ---: |
| small | 288 | 60% | 6–15 mm | 25–60% |
| medium | 120 | 25% | 15–35 mm | 20–50% |
| large | 60 | 12.5% | 35–70 mm | 15–40% |
| very large | 12 | 2.5% | 70–150 mm | 12–32% |

Characteristic sizes are log-distributed inside each range. Point attributes
store position, surface-aligned rotation with controlled tilt, scale,
characteristic size, class index, burial fraction and prototype index.

## Spatial distribution and safety guards

A deterministic 32 cm cluster field (strength 0.55) creates clast-rich and
clast-poor areas. A weaker 0.30 weighting favors negative Level-1 displacement
without forcing every clast into depressions.

Burial is family- and class-specific. The instance center is computed from the
rotated ellipsoidal vertical extent. The persisted-scene validator measured
the highest estimated bottom at -0.0845 mm relative to the surface, so no
tested clast floats.

Three elliptical wheel-contact exclusions intersect the patch. Each ellipse
is expanded by the radius of the individual instance; the same radius-aware
guard prevents rocks crossing patch boundaries. Coarse rocks additionally use
a 0.55 minimum center-spacing factor to prevent geometric intersections. These
checks were added after an adversarial review of the larger-rock proposal.

The scatter signature is
`0c77345239666732b2b0286104648e300b12924e4adef216512d7a29f3748400`.
It includes positions, rotations, scales, characteristic sizes, size classes,
burial fractions and prototype indices.

## Render checkpoint

The build can regenerate paired L1/L2 rover and terrain-only views at 10 cm
and 30 cm, plus a full-patch distribution view. The historical checkpoint
images were removed after Level 2 was promoted; the derived Blend and
validation reports remain authoritative.

The close-up target is a representative medium clast nearest the patch center
(22.92 mm), not a member of the rare largest class. This keeps the comparison
useful without letting a 150 mm tail sample dominate every close-up. Coarse
sizes use jittered logarithmic strata per class, guaranteeing interval
coverage even for rare classes without creating uniformly spaced sizes.

## Reproducible build and validation

```powershell
python scripts/host/run_microterrain_level2.py `
  --blender-executable "D:\Programmi\Blender Foundation\Blender 5.2\blender.exe"
```

`build_L2.json` records the generated scene. A second Blender process reopens
the `.blend` and writes `validation_L2.json`. Validation regenerates the exact
scatter and checks family/class counts, source-library size, non-realized
Geometry Nodes, every point attribute, Level-1 topology/signature, radius-aware
wheel and boundary exclusions, coarse spacing, burial, non-floating placement,
materials and render artifacts.

Pure generation and report contracts are covered by
`tests/test_microterrain_clasts.py`.

## Checkpoint status

The revised L2 passed deterministic, structural and visual QA and is the input
to the regenerated Level-3 asset. Final acceptance of the new rock scale
remains a user visual decision.

Deferred work remains the final multi-distance comparison matrix, anomaly
injection, pose sampling, dataset rendering and physical granular dynamics.
