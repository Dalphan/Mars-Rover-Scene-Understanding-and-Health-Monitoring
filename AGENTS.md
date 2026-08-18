# Codex repository guide

Before working on files, configs, scripts, or outputs belonging to
`anomaly_detection_2`, read `docs/anomaly_detection_2/INDEX.md` first and use
its topic routing to select the milestone-specific documentation.

Files under `docs/handoff/` are a historical archive for the legacy Blender
pipeline. Read them only when working on that legacy pipeline or when a task
explicitly requires reconstructing an older decision, failure, or run. Do not
update handoff files for routine `anomaly_detection_2` milestones; update its
`INDEX.md` and the relevant milestone document instead.

## Repository boundaries

- Existing segmentation/training code under `src/`, `configs/`, and `tests/`
  predates the Blender work and must not be reorganized as part of the wheel
  pipeline.
- Host-Python entrypoints belong in `scripts/host/`. They must not import
  `bpy` and should remain testable with ordinary Python.
- Blender entrypoints belong in `scripts/blender/`. Blender-only dependencies
  such as `bpy` and `bmesh` must not leak into host modules.
- Pure Python logic and validators for the Blender stages live under
  `src/blender_audit/` and `src/wheel_preparation/`.
- Blender configuration belongs in `configs/blender/`.
- Original assets are immutable, local-only, and ignored under
  `assets/original/`. Never edit, convert, download, or commit the NASA GLB.
- Runtime output belongs under `outputs/` and logs under `logs/`; both are
  ignored. Do not commit generated `.blend` files or diagnostic renders.

## Current Blender contract

- Reproducible target: Blender 5.2.0 LTS, Eevee, 800x600, 4:3.
- The real asset is `24584_Curiosity_static.glb`, SHA-256
  `86a8ee6d39fb1711ae549ba99951d2f52b44a7b8367ce4814cc81aec60e94b31`.
- The audit verdict is B: six wheel assemblies are automatically separable
  from the single imported mesh `MSL`.
- The phase-2 default is `wheel_candidate_05`; never switch candidates
  silently. Candidate component indices are valid only when asset hash,
  Blender major/minor, imported topology, and audit report all match.
- Work only on imported copies. Every real run must verify the source checksum
  before and after.
- No ML dependencies inside Blender Python. No anomaly randomization, bulk
  dataset generation, or model training until the canonical wheel and one
  counterfactual perforation pass visual QA.

## Verification expectations

Run the focused host tests after edits:

```powershell
python -m unittest discover -s tests -p 'test_blender_audit*.py' -v
python -m unittest tests.test_wheel_preparation_core tests.test_wheel_preparation_validation -v
```

Then run the real GLB pipeline and its host validator using the exact commands
in the handoff document. Passing the JSON validator is necessary but currently
not sufficient: inspect `normal.png`, `anomaly.png`, `anomaly_mask.png`, and
`difference.png` visually before declaring phase 2 complete.

Preserve unrelated user changes and keep documentation synchronized with any
new gate, fallback, output schema, or Blender-version requirement.

## Critical reanalysis rule

Before implementing a proposed design or approach, perform a second-pass
adversarial review. Identify likely failure modes, side effects, scale and
performance risks, and data-validity risks; then revise the approach before
implementation and record any material tradeoffs.
