# Kaggle Wheel Anomaly-Detection Notebooks

This directory contains two self-contained notebooks. Each includes the
required dataset, preprocessing, model and evaluation code, so Kaggle execution
does not depend on importing `src/` from a repository checkout.

## Available notebooks

| Notebook | Purpose | Checked-in selection |
| --- | --- | --- |
| [PatchCore workflow](kaggle_wheel_anomaly_detection.ipynb) | Clean-feature extraction, memory-bank construction, calibration and synthetic evaluation. | PatchCore Light, ResNet-18, 384 x 512. |
| [Multi-model workflow](kaggle_wheel_anomaly_detection_model_roadmap.ipynb) | PatchCore, EfficientAD-S, SuperSimpleNet and TinyGLASS training/evaluation. | SuperSimpleNet, pose `A_overhead`, 256 x 256 input. |

Earlier efficiency, qualitative-comparison and real-wheel notebooks mentioned
in historical research records are not part of this directory. Their recorded
results remain documented in the
[experiment history](../docs/anomaly_detection/experiment_history.md) and
[real-wheel pilot](../docs/anomaly_detection/real_zero_shot_pilot.md).

## Kaggle setup

1. Create a Kaggle session and open the selected notebook.
2. Attach the extracted synthetic dataset. Its root must contain
   `samples.csv`, `images/` and `masks/`, including every indexed raster.
3. Enable Internet for dependencies and pretrained-weight downloads.
   Select a GPU for the full experiments.
4. Review the `Configuration` cell before running later cells.
5. Set `DRIVE_UPLOAD_ENABLED=False` for a run that only saves to Kaggle.

The dataset discovery function searches beneath `/kaggle/input/`. Attach one
compatible dataset; if multiple matching roots are present, discovery raises
an error rather than choosing one silently. Dataset files and trained
checkpoints are not included in this repository.

The PatchCore notebook has Drive upload disabled. **The multi-model notebook
has it enabled in the checked-in configuration**; disable it or configure
your own Drive destination and credentials before execution.

## Model selection and data scope

For PatchCore, `PATCHCORE_PRESET="light"` uses ResNet-18 with 384 x 512
inputs and a 384-dimensional embedding. `"reference"` uses WideResNet-50-2,
short-side resize 256, center crop 224 and a 1024-dimensional embedding.
The configuration also controls training-embedding limits and memory-bank
size; a full bank can require substantial GPU time and host memory.

In the multi-model notebook, `MODEL_NAME` accepts `patchcore`,
`efficientad_s`, `supersimplenet` or `tinyglass`.
`MODEL_CONFIGS` contains model-specific settings. Its current
`CAMERA_POSES=("A_overhead",)` selects the overhead subset; set it to
`None` to use all poses. Pose crop, input geometry and normalization are
part of the experiment definition.

Current SuperSimpleNet settings use pose crops, Perlin threshold 0.2,
wheel-restricted synthetic training anomalies and top-1% image scoring.
TinyGLASS settings include a frozen backbone, layer2 features, 384 x 384
input, sigma 1 and hole-mode local anomalies. Inspect the configuration cell
when selecting a different model; recorded experiments can use other settings.

## Model-specific resources

Pretrained backbone weights may be downloaded at first use. EfficientAD also
needs its pinned teacher weights and ImageNet-1k training data for the penalty
term. Configure the Kaggle `HF_TOKEN` Secret with access to
`ILSVRC/imagenet-1k` when training that model.

TinyGLASS downloads DTD textures only for `las_mode="texture"` or
`"mixed"`. Its checked-in `"hole"` mode does not need that texture dataset.

Training uses clean synthetic images. Validation and test keep their original
labels and pairing. Model selection and diagnostic ablations follow the
[evaluation contract](../docs/anomaly_detection/evaluation.md).

## Execution, resume and outputs

Run setup, imports and configuration first, then definition cells in order.
The later execution cells prepare and inspect the data, initialize the model,
fit, validate, test, produce diagnostics and save artifacts. Use the visual
preprocessing checks before a full fit.

The multi-model notebook supports `RESUME` and `RESUME_RUN_DIR` for
trainable models. Resume needs the matching training state and configuration.
Changing input size, model or other checkpoint-contract settings requires
a separate compatible run.

Outputs are saved under `/kaggle/working/` in the configured run directory.
They include manifests, checkpoints, metrics, scores and qualitative figures.
Download or deliberately upload the results before ending the session.

Optional Drive operations use your own Kaggle Secrets and destination:
`GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`,
`GDRIVE_REFRESH_TOKEN` and, when required, `GDRIVE_FOLDER_ID`.
The credential helper also accepts the `GOOGLE_*` aliases described in the
notebook. Existing research folder references do not grant access to weights.

## Source synchronization

Authoritative ML code is in
[`src/anomaly_detection/`](../src/anomaly_detection/). Notebook copies are
intentional: when changing loading, model behavior or evaluation, synchronize
the corresponding implementation in both locations.

The repository currently has no `tests/` directory. Full Kaggle execution,
checkpoint restoration and GPU measurements must be validated in the target
runtime. Available local entrypoints and setup instructions are in the
[project README](../README.md).
