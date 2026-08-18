# Wheel surface-wear pilot

This pilot compares the current `mars_dusty_refined` wheel surface with two
deterministic healthy-wear levels on the same frame:

- `surface_current`: no scratch layer;
- `wear_light`: a 1.00% texture-mask coverage target;
- `wear_evident`: a 2.50% texture-mask coverage target containing the complete
  light pattern plus additional scratches.

The scratch layer changes only base color, roughness, metallic response and a
0.25 mm shader bump. It never changes source geometry or silhouette. Target
wheel mesh datablocks and materials are copied only in the temporary render
scene, preventing shared rover materials from receiving wheel wear.

The generated masks are deterministic and use a wheel-local box projection.
This avoids scale discontinuities in the NASA asset's packed UV atlas and
keeps scratches attached to the wheel during roll sampling. A geometric
exposure mask admits wear only where radius and outward-facing normal identify
the tread, or where position and axial normal identify the outboard shoulder;
hub, spokes and inward-facing cavity surfaces are excluded. Dataset policy
keeps wear independent of anomaly labels and requires the healthy/anomaly
counterfactual pair to use the same wear variant and seed. Wear masks are
audit metadata and never become anomaly masks.

Initial sampling weights, still disabled until visual approval, are 35%
current, 45% light wear and 20% evident wear. The lower weight for evident
wear limits the risk that high-contrast scratches become an anomaly proxy.

```powershell
python scripts/host/run_wheel_surface_wear_pilot.py `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

Runtime products are written to
`outputs/anomaly_detection_2/surface_wear_pilot/`. The two configured scratch
masks and the JSON report are retained; historical comparison renders were
removed and can be regenerated with the command above.

## First rendered test

The final wheel-local projection rendered successfully under
`mars_dusty_refined`. Automatic gates passed with unchanged camera, wheel
matrix, source geometry and source Blend checksum. Measured texture-mask
coverage is 1.080% for light wear and 2.550% for evident wear. At 1200 x 900,
respectively 1.27% and 2.98% of frame pixels change by more than the diagnostic
threshold after the physical surface-exposure mask is applied.

The first UV-projected attempt was rejected before review because uneven UV
texel density enlarged a fine scratch into a bright saw-tooth patch. The
wheel-local box projection removes that artifact. A later whole-wheel version
was also rejected because scratches reached the inner cavity. The final
surface-exposure mask limits wear to outward-facing tread and the outboard
shoulder, with modest texture-coverage increases to 1.00% and 2.50%.

Status: rendered, automatic gates passed, pending visual approval. Dataset
sampling remains disabled.
