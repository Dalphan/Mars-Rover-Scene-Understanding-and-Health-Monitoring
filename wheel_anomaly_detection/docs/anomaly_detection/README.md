# Visual anomaly detection

This section documents the machine-learning half of the project. Dataset
provenance and Blender generation remain in the separate
[generation index](../generation/INDEX.md).

## Start here

| Topic | Document |
| --- | --- |
| Dataset schema, loading and preprocessing | [Dataloader](dataloader.md) |
| Shared metrics, diagnostics and artifacts | [Evaluation](evaluation.md) |
| Model design and implementation plan | [Model roadmap](model_roadmap.md) |
| Complete experimental record | [Experiment history](experiment_history.md) |
| Focused PatchCore analysis | [PatchCore results](patchcore_results.md) |

## Implemented models

- PatchCore, with lightweight and reference presets;
- EfficientAD-S;
- TinyGLASS;
- SuperSimpleNet.

All models share the same dataset interface and evaluation contract. Training
uses clean images only; validation and test contain paired clean/anomalous
samples. The project reports image AUROC, image Average Precision, pixel AUROC,
pixel Average Precision and extended localization diagnostics such as AUPRO.

## Running an experiment

From `wheel_anomaly_detection/`:

```bash
python scripts/anomaly_detection/run_experiment.py \
  dataset.root=/path/to/curiosity_wheel_hole_v1_10000
```

Model and experiment presets are located in
`configs/anomaly_detection/model/` and `configs/anomaly_detection/experiment/`.
The self-contained Kaggle workflow is described in the
[notebook guide](../../notebooks/anomaly_detection/README.md).
