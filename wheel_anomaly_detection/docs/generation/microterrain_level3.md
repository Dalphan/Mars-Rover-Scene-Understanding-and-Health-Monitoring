# Milestone 5 checkpoint — deterministic microterrain Level 3

Level 3 starts from the visually approved `microterrain_L2.blend` and adds
only sub-granular shading. The patch topology, Level-1 displacement, Level-2
point clouds, instance transforms, prototype selection and rover pose remain
unchanged.

The persisted asset is
`outputs/anomaly_detection_2/microterrain/level3/microterrain_L3.blend`.

## Non-destructive material ablation

The Level-2 terrain material `GaleTerrainMacroAlbedo` remains present and
unchanged. `GaleTerrainMicroterrain_L3` is a copy that keeps the HiRISE image
texture upstream of Base Color, then adds the procedural layers. All four
`Clast_L2_*` materials are retained with fake users and each has a separate
`Clast_L3_*` copy. Every source prototype stores both material names, allowing
the renderer to switch L2/L3 without rebuilding geometry.

The Level-3 scene is saved in its native L3 material state. No displacement
modifier, subdivision, realized instance or new terrain mesh is introduced.

## Terrain shader

The matrix shader contains six independent metric 3D-noise bands:

| Contribution | Wavelength | Role |
| --- | ---: | --- |
| coarse albedo/dust mask | 65 mm | broad dusty variation |
| medium albedo | 12 mm | weak local tone modulation |
| fine roughness | 2.5 mm | diffuse response variation |
| very-fine roughness | 0.8 mm | second roughness scale |
| primary bump | 0.45 mm | sub-granular normal detail |
| secondary bump | 0.18 mm | finer normal variation |

The two albedo bands multiply the existing HiRISE color by at most 10% and
4.5%, then mix a dust color through a low-frequency mask with global amount
0.18. The image texture therefore remains the spatial/color baseline rather
than being replaced by synthetic noise.

Roughness is mapped around a dusty baseline of 0.94 with maximum configured
variation 0.14. The two bump fields are combined 72/28. The Bump node uses
strength 0.38 and metric distance 0.16 mm. Bump changes shading normals only;
it cannot alter silhouette or contact geometry.

## Clast shader

Each of the four L3 clast materials combines three bands:

- a 3.5 mm dust mask;
- a 0.8 mm roughness field;
- a 0.35 mm bump field.

Clasts receive 55% of the terrain dust contribution so they remain slightly
cleaner and more contrasted than the matrix. Their bump distance is limited to
0.10 mm and strength to 0.32; roughness variation is 0.11. The original rust, dark, light and cleaner L2
palette remains recognizable; metallic response is never enabled.

## Determinism and geometry preservation

The canonical material-configuration signature is
`fed526160039b3c6ee59857887191f101fc0796923237a5101633ae71ce7359e`.
The persisted Level-2 scatter signature remains
`0c77345239666732b2b0286104648e300b12924e4adef216512d7a29f3748400`.

The validator also confirms:

- patch topology remains 2,563,201 vertices and 2,560,000 faces;
- the Level-1 displacement signature is unchanged;
- all three Level-2 family signatures and point counts are unchanged;
- the 288/120/60/12 coarse size mixture and size-aware safety guards remain unchanged;
- the Geometry Nodes graphs contain no `Realize Instances` node;
- HiRISE Image Texture remains upstream of the L3 terrain Base Color;
- L2 and L3 materials coexist for deterministic ablation.

## Render checkpoint

The build can regenerate paired L2/L3 rover views and shader close-ups at 5 cm
and 10 cm with identical scene state. The historical checkpoint images were
removed after Level 3 was promoted; the derived Blend and validation reports
remain authoritative.

At 5 cm the mean absolute RGB difference is approximately 0.0078 on a 0–1
scale; at 10 cm it is approximately 0.0093. This is roughly twice the previous
profile and makes the contribution clearly readable while preserving HiRISE
color and geometric silhouettes. It appears as continuous dusty microtexture
rather than displaced geometric noise.

## Reproducible build and validation

```powershell
python scripts/host/run_microterrain_level3.py `
  --blender-executable "D:\Programmi\Blender Foundation\Blender 5.2\blender.exe"
```

The build writes `build_L3.json` and a second Blender process writes
`validation_L3.json`. The validator checks exact node counts and scales,
metric bump parameters, channel links, material signatures, L2 material
preservation, geometry signatures and every expected render.

Pure configuration and report gates are covered by
`tests/test_microterrain_material.py`.

## Deferred pending visual approval

- the final L0/L1/L2/L3 matrix at 5, 10, 20 and 30 cm;
- anomaly injection, pose sampling and dataset rendering;
- physical dust or granular dynamics.
