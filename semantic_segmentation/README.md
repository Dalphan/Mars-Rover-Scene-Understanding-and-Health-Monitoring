# Mars Rover Semantic Segmentation

A PyTorch training and quantization pipeline for semantic segmentation of Mars
rover scenes. It supports S5Mars (9 classes) and the taxonomy-compatible
MarsSeg MSL/MER pair (7 classes).

## Highlights

- Configuration-driven experiments with Hydra.
- Dataset inspection with class-distribution reports and visual samples.
- SegFormer-B0, `segmentation-models-pytorch`, and native LCNet variants.
- Cross-entropy, generalized Dice, and combined segmentation losses.
- Optional image-level oversampling and train-only geometric augmentation.
- Source-test evaluation plus optional zero-shot MarsSeg MSL ↔ MER evaluation.
- Standalone ONNX PTQ and NVIDIA ModelOpt QAT entrypoints.

## Dataset

The loader expects the S5Mars fields `image`, `mask`, `width`, `height`, and
`class_labels`. The semantic classes are:

`Background`, `Bedrock`, `Hole`, `Ridge`, `Rock`, `Rover`, `Sand / Soil`,
`Sky`, and `Track`.

If access to the dataset requires authentication, export a valid `HF_TOKEN`
before running the commands below. Never commit access tokens or local secrets.

## Installation

Run all commands from `semantic_segmentation/`. Python 3.11 or 3.12 is
recommended.

```bash
cd semantic_segmentation
python -m pip install -r requirements.txt
```

## Quick start

Inspect the dataset and create summary files and visualizations:

```bash
python scripts/analyze_dataset.py
```

For a small smoke run:

```bash
python scripts/analyze_dataset.py analysis.max_samples=5 visualization.num_samples=2
```

Start the default SegFormer-B0 S5Mars experiment (same defaults as the
authoritative training notebook):

```bash
python -m src.train.train_segmentation
```

Useful Hydra overrides include:

```bash
python -m src.train.train_segmentation epochs=10 batch_size=4
python -m src.train.train_segmentation model=smp model.encoder_name=mobilenet_v2
python -m src.train.train_segmentation model=lcnet3_7 freeze=none model_analysis.enabled=true
python -m src.train.train_segmentation dataset_name=marsseg_msl run_cross_dataset_evaluation=true
python -m src.train.train_segmentation oversampling.enabled=true
```

Analysis results are written to `outputs/s5mars_analysis/`; training runs are
written under `outputs/`. Both locations are intentionally ignored by Git.

## Kaggle notebooks

The repository keeps exactly three self-contained notebooks:

- `notebooks/kaggle_segmentation_training.ipynb` trains on S5Mars, MarsSeg MSL,
  or MarsSeg MER. Set `DATASET_NAME`, `RUN_CROSS_DATASET_EVALUATION`, and
  `OVERSAMPLING_ENABLED` in its configuration cell. Cross-dataset evaluation
  is allowed only between the two compatible MarsSeg taxonomies.
- `notebooks/kaggle_s5mars_onnx_ptq.ipynb` runs the ONNX/TensorRT
  post-training quantization pipeline.
- `notebooks/kaggle_s5mars_onnx_qat.ipynb` runs selective NVIDIA ModelOpt QAT
  and the direct TensorRT comparison.

For an ordinary S5Mars run, use `DATASET_NAME = "s5mars"` and keep
`RUN_CROSS_DATASET_EVALUATION = False`. For a MarsSeg zero-shot run, choose
`marsseg_msl` or `marsseg_mer` and enable the cross-dataset flag. Oversampling
is training-only and never changes validation or test distributions.

The notebooks are authoritative. Python modules and YAML files are maintained
as a one-way mirror; nothing in the standalone pipeline rewrites a notebook.
Run the read-only synchronization gate after changing notebook constants:

```bash
python scripts/check_notebook_config_sync.py
```

## Standalone training flags

Hydra booleans use lowercase `true`/`false` on the command line.

- `dataset_name=s5mars|marsseg_msl|marsseg_mer` selects the source dataset,
  repository, taxonomy, expected split sizes, and output/checkpoint identity.
- `run_cross_dataset_evaluation=true` adds a zero-shot test only for MarsSeg:
  MSL-trained models are tested on MER, and MER-trained models on MSL. It is
  rejected for S5Mars. The target test loader is created only after training,
  source-validation checkpoint selection, and source testing.
- `oversampling.enabled=true` enables image-level, training-only sampling from
  `class_labels`. Tune it with `power`, `max_weight`,
  `num_samples_multiplier`, and `excluded_class_ids`. Replacement must remain
  `true`; validation and test are never oversampled.
- `augmentation.enabled=true` enables training-only horizontal/vertical flips
  and random 90° rotations. Their probabilities are separately configurable;
  set vertical and rotation probabilities to `0.0` for the geometry-safe
  horizontal-only variant.
- `execution.skip_train=true` runs evaluation only and requires
  `execution.load_checkpoint=true`. `execution.load_best=true` selects
  `best.ckpt`; `false` selects `last.ckpt`.
- `checkpoint.download_from_drive=true` downloads a missing evaluation
  checkpoint from the experiment-specific Drive folder when training is
  skipped. `force_download=true` replaces the local copy after validating the
  remote file size.
- `limits.train_batches`, `limits.val_batches`, and `limits.test_batches`
  provide explicit smoke-run limits; `null` means the complete split.
- `model=segformer_b0|smp|lcnet3_7|lcnet3_11`, `freeze`, `criterion.name`,
  optimizer, scheduler, workers, memory logging, and model analysis remain
  configurable in `configs/train/segmentation.yaml`.

Examples:

```bash
# Train on MSL, select the best checkpoint on MSL validation, then zero-shot MER test.
python -m src.train.train_segmentation dataset_name=marsseg_msl run_cross_dataset_evaluation=true

# S5Mars oversampling ablation, same source validation/test distribution.
python -m src.train.train_segmentation dataset_name=s5mars oversampling.enabled=true

# Evaluate an existing best checkpoint without constructing train/val loaders.
python -m src.train.train_segmentation execution.skip_train=true execution.load_checkpoint=true execution.load_best=true
```

## Standalone PTQ and QAT

GPU quantization dependencies are intentionally separate from ordinary
training:

```bash
python -m pip install -r requirements-quantization.txt
python -m src.quantization.ptq
python -m src.quantization.qat
```

PTQ stage flags live under `steps.*`. They gate the FP32 baseline, ONNX
export/benchmark, FP16 conversion, calibration preparation, static INT8 Q/DQ
quantization, TensorRT probe/benchmark, and the final hold-out test. Stages
retain their artifact dependencies: for example, INT8 quantization requires
calibration preparation in the same run, while a disabled export requires the
matching ONNX artifact to exist already. The default test flag is `false`, so
the test set is not used while choosing calibration or operator coverage.

QAT exposes `steps.restore_saved_qat`, `steps.run_trt_fp16_baseline`, and
`steps.run_final_test`, plus the two Drive-upload flags. Its training,
distillation, selective Conv/Linear quantization, export parity thresholds,
TensorRT build modes, and benchmark protocol are in
`configs/quantization/qat.yaml`.

## Configuration

The analysis configuration is in `configs/config.yaml`; the synchronized
standalone configs are `configs/train/segmentation.yaml`,
`configs/quantization/ptq.yaml`, and `configs/quantization/qat.yaml`. Model
presets remain under `configs/model/`.

## Project structure

```text
configs/       Hydra data, model, and training presets
scripts/       Dataset analysis entry point
src/data/      Dataset loading and transforms
src/models/    Segmentation model implementations
src/losses/    Loss functions
src/metrics/   Evaluation metrics
src/train/     Training and evaluation loops
src/quantization/  PTQ/QAT entrypoints and shared deployment utilities
tests/         Unit tests
```

## Tests

```bash
python -m pytest -q
```
