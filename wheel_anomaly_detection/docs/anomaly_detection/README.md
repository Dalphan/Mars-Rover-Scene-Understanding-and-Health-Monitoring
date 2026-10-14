# Visual Anomaly Detection

This guide covers dataset loading, models and evaluation for Curiosity wheel
damage. Blender scene construction and dataset provenance are documented in
the separate [generation index](../generation/INDEX.md).

## Documentation

| Topic | Document |
| --- | --- |
| Manifest schema, loading and preprocessing | [Dataloader](dataloader.md) |
| Shared metrics, diagnostics and artifacts | [Evaluation](evaluation.md) |
| Model implementation and design | [Model roadmap](model_roadmap.md) |
| Recorded experiments and efficiency measurements | [Experiment history](experiment_history.md) |
| PatchCore-specific analysis | [PatchCore results](patchcore_results.md) |
| Exploratory real-wheel evaluation | [Real zero-shot pilot](real_zero_shot_pilot.md) |

Experiment records may mention local output paths and workflows from earlier
research stages. Those runtime artifacts are excluded from Git. Available
Kaggle entrypoints are listed in the [notebook guide](../../notebooks/README.md).

## Models and evaluation

The implementation contains PatchCore, EfficientAD-S, TinyGLASS and
SuperSimpleNet. All share the dataset and evaluation interface. Training uses
clean images; validation and test include paired clean/anomalous samples.

Reported metrics include image AUROC/AP, pixel AUROC/AP and AUPRO. Cropping,
input resolution, pose selection and optional target-wheel masks change the
evaluation region. Compare experiments using their full preprocessing
contracts and validation/test split definitions.

## Running an experiment

Install the dependencies using the [project setup guide](../../README.md).
From `wheel_anomaly_detection/`, supply the extracted dataset:

```bash
python scripts/anomaly_detection/inspect_dataloaders.py dataset.root=/path/to/curiosity_wheel_hole_v1_10000 dataloader.num_workers=0
python scripts/anomaly_detection/run_experiment.py dataset.root=/path/to/curiosity_wheel_hole_v1_10000
```

The root must contain `samples.csv` and all referenced raster files.
Data and trained weights are not bundled with the repository.

The [base configuration](../../configs/anomaly_detection/config.yaml)
selects PatchCore and `A_overhead` images. Set `dataset.camera_poses=null`
for all poses, or select a [model preset](../../configs/anomaly_detection/model/)
and an [experiment preset](../../configs/anomaly_detection/experiment/):

```bash
python scripts/anomaly_detection/run_experiment.py dataset.root=/path/to/curiosity_wheel_hole_v1_10000 model=tinyglass +experiment=tinyglass_pose_a
```

Backbone weights may require a first-use download. EfficientAD additionally
requires teacher weights and authenticated ImageNet-1k access. Output paths
and resume options are controlled by the configuration. Recorded results
identify particular research runs and may use different settings from the
current presets.

## Maintenance

Authoritative ML code lives in `src/anomaly_detection/`. Kaggle notebooks
contain explicit copies so they can run without repository imports.
Synchronize affected implementations when changing the model, loader or
evaluation behavior. The current checkout does not include a unit-test suite;
data inspection and full experiment validation remain separate checks.
