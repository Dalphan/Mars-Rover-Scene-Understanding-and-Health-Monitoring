# Deterministic wheel-roll pose sampling

This stage derives a scriptable wheel-roll asset from the validated Level-3
scene. It does not alter the Level-3 source and it does not require render or
manual visual approval.

## Roll model

- All six wheel origins are within `1e-5 m` of their local bounding-box centre.
- Roll is therefore applied reversibly as `base_matrix_local @ RotX(angle)`;
  no pivot object, re-parenting or mesh edit is introduced.
- A single signed travel phase is applied to all six wheels so that a sample
  represents straight rover motion rather than six unrelated rotations.
- Healthy samples use eight stratified phases: `0, 45, 90, 135, 180, 225,
  270, 315` degrees.
- The equivalent signed travel is recorded as `radians(roll) * wheel_radius`.
- The persisted derived `.blend` is restored to zero roll. Every wheel stores
  its base local matrix, local roll axis, radius and configuration signature as
  custom properties.

Camera `up` and `forward` are derived from fixed world/rover vectors. They do
not use the rolling local Y/Z axes of the wheel, so rolling the tread cannot
rotate the camera with it.

## Fail-closed anomaly sampling

`select_visible_anomaly_roll()` in
`scripts/blender/wheel_pose_sampling.py` requires at least one anomaly probe.
Each probe consists of a local-space surface anchor and an outward local-space
normal. Missing metadata raises an error; an anomalous sample cannot fall back
to an unconstrained healthy roll.

For every camera pose, the sampler first places the anomaly toward the camera
within the wheel radial plane, with an upper bias. It then performs a bounded,
deterministic angular search. A candidate is accepted only when the configured
fraction of probes passes all four gates:

1. radial anchor is within 52 degrees of the upper wheel direction;
2. outward normal has camera-facing dot product of at least 0.30;
3. anchor is inside a 4% image-frame margin;
4. camera ray is not blocked by the wheel before the anchor, suspension,
   chassis, another wheel, terrain or clasts.

The current policy requires a visible-probe fraction of `1.0`. A future
anomaly injector may supply centre and boundary probes to require the whole
labelled region, rather than only its centre, to remain visible.

## Automatic validation result

The build validates eight real central-tread surface anchors for each of six
wheels, four approved camera poses and three target jitters: 576 combinations.
All 576 found a valid roll without rendering:

- 480 passed at the preferred angle;
- 32 required `-12 deg`;
- 64 required `+12 deg`;
- no candidate required a larger fallback;
- visibility failures: 0;
- physical-clearance phase failures: 0;
- maximum clearance loss relative to the assembled base pose:
  `7.45e-8 m`.

The physical gate is relative to the imported assembled pose because that pose
already contains local terrain intersections of up to about 15 mm. Pose
sampling is not authorized to translate the rover vertically; it only rejects
roll phases that worsen the base state by more than 1 mm.

## Outputs and reproduction

- Derived asset:
  `outputs/anomaly_detection_2/pose_sampling/wheel_roll_pose_sampling.blend`
- Build report:
  `outputs/anomaly_detection_2/pose_sampling/build_pose_sampling.json`
- Reopen validation:
  `outputs/anomaly_detection_2/pose_sampling/validation_pose_sampling.json`
- Derived Blend SHA-256:
  `9d33d41b643fb78b57e99a71a68a839d170b56cd96341145e384b9ad338d4030`

```powershell
python scripts/host/run_wheel_pose_sampling.py `
  --blender-executable 'D:\Programmi\Blender Foundation\Blender 5.2\blender.exe'
```

The host launcher verifies the Level-3 source checksum before and after both
the build and reopen validation. It also reads the JSON gates explicitly,
because Blender can return process exit code zero after a Python exception.

## Integration contract for anomaly injection

The injector must parent any separate anomaly carrier object to its wheel and
call `select_visible_anomaly_roll()` with wheel-local anchor/normal probes and
the carrier-object list before writing a sample. Integrated Boolean damage can
use the wheel itself as carrier. The returned pose may be rendered only when
`ok` is true. Dataset
metadata must retain the selected roll, equivalent travel, selected fallback,
per-probe gate results and camera-pose identifier.

Bulk generation remains disabled until the anomaly injector is connected to
this API; the current synthetic-anchor audit proves the pose sampler and the
six-wheel scene geometry, not a specific injected anomaly mesh.
