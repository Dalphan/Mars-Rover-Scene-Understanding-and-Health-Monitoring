# Wheel camera pose pilot

This pilot compares wheel-centric camera families on `wheel_middle_left` without
saving changes to the Level-3 source scene. Camera positions are expressed in a
wheel-local basis (`outward`, `forward`, `up`) so that the same definitions can
be mirrored across all six wheels.

## Camera contract

- Resolution: 1600 x 1200 (4:3)
- Focal length: 21.0 mm
- Sensor width: 11.8 mm
- Diagonal field of view: 38.70 degrees
- Depth of field: disabled for the composition pilot
- Temporary warm area fill: enabled for visual inspection only; it is created
  at runtime and the source `.blend` is never saved

## Pose families

| Family | Pilot pose | Distance | Elevation | Tangential angle | Intended evidence |
| --- | --- | ---: | ---: | ---: | --- |
| A | engineering overhead | 1.50 m | 55 deg | +10 deg | tread and upper wheel surface |
| C1 | leading three-quarter | 1.52 m | 31 deg | +43 deg | tread/side transition from the leading direction |
| C2 | trailing three-quarter | 1.52 m | 31 deg | -43 deg | complementary tread/side transition |
| D | rare upper detail | 0.70 m | 45 deg | +20 deg | local high-resolution tread detail |

A, C1 and C2 must contain the full projected wheel bounding box. D uses an
intentional crop. The subsequent pose-sampling stage now applies the same hard
anomaly-visibility gate to all four families, including D.

Framing is measured from all evaluated wheel-mesh vertices rather than the
eight corners of the object's rectangular bound box, which overestimates the
frame occupancy of cylindrical wheel geometry.

## Initial sampling recommendation

Use A/C as the normal distribution and D only as a rare conditional view:

- A: 40%
- C: 56%, split evenly between C1 and C2
- D: 4%, only when the injected anomaly intersects the visible crop

The weights are a starting point for dataset design, not an approved final
distribution. Small deterministic angular and distance jitter should be added
within each family later, after validating all six wheels for self-occlusion.

## Known limitations

- The rendered pilot covers the middle-left wheel only. The non-rendering pose
  sampler subsequently audited all six wheels and rejects suspension/chassis
  occlusion by ray cast.
- The temporary fill light makes pose comparison reliable but does not define
  the final illumination randomization policy.
- A temporary terrain-only grade preserves the coregistered IRB spatial signal
  while increasing red-brown saturation and reducing the washed-out value.
- Pose D can create label noise if used outside the new fail-closed pose sampler.

## Terrain coverage

The Level-3 source now uses a 4 x 4 m detail patch. The measured ground
footprints of A, C1, C2 and D all lie inside its bounds. The most constrained
C1 corners retain approximately 0.19-0.21 m to the patch boundary, which is
greater than the configured 0.15 m feather width for the deterministic pilot.
Any future camera jitter must be re-audited against this remaining margin.
