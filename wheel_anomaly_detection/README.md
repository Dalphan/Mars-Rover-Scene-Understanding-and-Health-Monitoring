# Curiosity Wheel Anomaly Detection

A reproducible Blender data-generation and validation pipeline for visual
anomaly detection on Curiosity rover wheels. It turns an immutable NASA rover
asset into paired clean and perforated-wheel samples, with semantic masks and
traceable generation contracts.

The data-generation pipeline is complete and documented; the reusable Python
package layout for the ML stage is in place, while a reproducible trained
baseline is still being developed.

## Highlights

- Validated separation of Curiosity's six wheel meshes from the source GLB.
- Deterministic Gale-crater terrain, camera, lighting, and wheel-pose setup.
- Domain-randomized clean and paired hole-anomaly renders with RGB and mask
  outputs.
- Dataset planning, split validation, resumable batch execution, and
  integrity checks.
- Pure Python validation logic that can be tested outside Blender.

## Dataset

The canonical local dataset is generated at
`outputs/anomaly_detection_2/datasets/curiosity_wheel_hole_v1_10000/`. It
contains 10,000 RGB renders, 10,000 target-wheel masks, and 1,250 anomaly
masks, arranged into train, validation, and test splits of 7,000, 1,000, and
2,000 images respectively.

The dataset, render outputs, and logs are local runtime artifacts and are not
committed to Git. Its provenance, contracts, and validation gates are indexed
in [`docs/generation/INDEX.md`](docs/generation/INDEX.md).

## Requirements

- Blender 5.2.0 LTS with Eevee for reproducible renders.
- A host Python environment for orchestration and validation.
- The original `24584_Curiosity_static.glb` stored locally in
  `assets/original/`. The source asset must remain unchanged.

Install the host-generation dependencies from the project directory:

```bash
cd wheel_anomaly_detection
python -m pip install -r requirements/generation.txt
```

Blender uses its bundled Python interpreter and must not import ML
dependencies. The anomaly-detection dependency set will be pinned when the
first reproducible ML baseline is defined.

## Workflow

1. Start with [`docs/generation/INDEX.md`](docs/generation/INDEX.md) to select
   the applicable generation milestone and validation procedure.
2. Run the host orchestration scripts in `scripts/host/` for planning,
   validation, and Blender invocation.
3. Run the associated Blender entry points from `scripts/blender/` only inside
   Blender.
4. Validate the generated outputs before promoting a dataset run.

## Project structure

```text
assets/original/             Local, immutable source asset placeholder
configs/blender/             Generation and rendering contracts
configs/anomaly_detection/   Reserved ML configuration area
docs/generation/             Authoritative dataset and pipeline documentation
docs/anomaly_detection/      ML protocol and benchmark documentation
scripts/blender/             Blender-side entry points
scripts/host/                Host-side orchestration and validation
src/                         Testable generation and ML package code
tests/                       Unit and validation tests
```

## Verification

Run the required host-side tests from `wheel_anomaly_detection/`:

```bash
python -m unittest discover -s tests -p "test_blender_audit*.py" -v
python -m unittest tests.test_wheel_preparation_core tests.test_wheel_preparation_validation -v
```

Generation changes also require the real-asset pipeline and the milestone
validator specified in the relevant document. JSON reports alone do not
replace visual render quality assurance.
