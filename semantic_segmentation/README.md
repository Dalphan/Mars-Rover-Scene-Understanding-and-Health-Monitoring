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

Start the default LCNet3_7 S5Mars experiment (same defaults as the
authoritative training notebook):

```bash
python -m src.train.train_segmentation
```

The prepared reference run trains LCNet from scratch for 30 epochs with batch
size 32, seed 42, and 512 x 512 inputs, without augmentation or oversampling.
It is a new controlled experiment, not a reproduction of the historical
60-epoch LCNet slide results. In Kaggle, start a fresh session and execute cells
in order through the local results summary; run the final optional Drive upload
cell separately only when desired. `LOAD_CKPT = True` and `LOAD_BEST = True`
reload the newly trained validation-selected checkpoint after training, not an
initialization checkpoint. For LCNet3_11, change only `LCNET_VARIANT` in the
notebook (or use `model=lcnet3_11` in Python). Variant-specific output directories
remain distinct, but repeated runs with the same variant/configuration overwrite
the same local directory, so preserve completed outputs before repeating one.

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

The repository keeps four self-contained notebooks:

- `notebooks/kaggle_segmentation_training.ipynb` trains on S5Mars, MarsSeg MSL,
  or MarsSeg MER. Set `DATASET_NAME`, `RUN_CROSS_DATASET_EVALUATION`, and
  `OVERSAMPLING_ENABLED` in its configuration cell. Cross-dataset evaluation
  is allowed only between the two compatible MarsSeg taxonomies.
- `notebooks/kaggle_s5mars_onnx_ptq.ipynb` runs the ONNX/TensorRT
  post-training quantization pipeline.
- `notebooks/kaggle_s5mars_onnx_qat.ipynb` runs selective NVIDIA ModelOpt QAT
  and the direct TensorRT comparison.
- `notebooks/kaggle_s5mars_qualitative_comparison.ipynb` prepares the qualitative
  comparison for thesis Figure 5.1: it samples 20 distinct S5Mars test images
  with seed 42, then downloads and evaluates each of the seven FP32 validation-
  selected `best.ckpt` checkpoints from Drive, one model at a time. It reuses
  the training architectures, 512 x 512 preprocessing, and class palette;
  no training or quantization runs are required.

For the qualitative notebook, enable Kaggle Internet, a GPU, and the same
`GDRIVE_CLIENT_ID`, `GDRIVE_CLIENT_SECRET`, and `GDRIVE_REFRESH_TOKEN` Secrets
as the training notebook (`GOOGLE_*` aliases are also accepted). `HF_TOKEN`
is optional when the dataset is accessible without authentication. All seven
Drive experiment folders are already configured, including the thesis LCNet
runs; folder IDs and `best.ckpt` availability were checked on 2026-10-06.
Its installation cell aligns `torch==2.9.1`, `torchvision==0.24.1`, and
`torchaudio==2.9.1` from the CUDA 12.8 index, matching training. Transformers
can import TorchAudio while loading SegFormer, so leaving Kaggle's preinstalled
TorchAudio alongside a downgraded PyTorch can fail with
`undefined symbol: torch_library_impl`. Run the corrected installation cell,
restart the kernel/session, then execute from configuration onward. Runtime
checks all three versions before importing the models. This notebook also pins
`transformers==4.49.0` to retain the SegFormer checkpoint layer names
(`encoder.patch_embeddings`, `encoder.block`, and `encoder.layer_norm`), which
can differ in newer 5.x releases. Both installed and in-memory Transformers
versions are checked before importing SegFormer; restart after downgrading.
Execute the cells in order. The default gallery displays 20 separate 3 x 3
comparisons; the accompanying navigator provides previous/next controls and
a **Scegli per la tesi** button. `DISPLAY_MODE = "interactive"` shows only the
navigator. `choose_comparison(7)` is the equivalent manual selection command.

Outputs are under `/kaggle/working/s5mars_qualitative/seed_42_n20/`:
`selection.json` preserves test indices and dataset identity, `samples/`
contains RGB and ground truth, `predictions/` contains class-ID/color masks
and checkpoint provenance, and `comparisons/` contains all 20 PNG/PDF grids.
Choosing a sample fills `selected/` with `figure_5_1.png`, `figure_5_1.pdf`,
individual panels, and `choice.json`. A downloadable ZIP excludes checkpoint
weights. Download the outputs before ending the Kaggle session. Completed
model predictions are reused only after identity and file-integrity checks;
changing `SEED` creates a distinct output directory. For another checkpoint
campaign, use a new `RUN_TAG`. The sampled gallery is for illustration and
does not replace full-test metrics or validation-based model selection.

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

The prepared PTQ preset recovers the missing DeepLabV3-ResNet34 FP32/FP16
baseline latency, energy, and GPU-memory measurements from the existing
`best.ckpt`. Training, QAT, the duplicate PyTorch baseline, calibration, INT8,
and hold-out testing are disabled. In Kaggle, use a fresh T4 GPU session with
Internet enabled and the `GOOGLE_REFRESH_TOKEN`, `GOOGLE_CLIENT_ID`, and
`GOOGLE_CLIENT_SECRET` secrets, then run all cells through the final summary.
The resource benchmark runs independently of INT8 and keeps one ONNX session
resident at a time, with the PyTorch model on CPU. It preserves batch size 1,
512 x 512 inputs, 10 warm-ups/50 latency forwards, and a separate efficiency
loop of at least 100 forwards and 30 seconds sampled by NVML every 20 ms.

Recovery outputs use a distinct directory so they do not replace the earlier
full PTQ campaign: `/kaggle/working/onnx/deeplabv3_baselines_recovery/` in the
notebook, or `outputs/onnx/deeplabv3_baselines_recovery/` in Python. In
`benchmark_results.json`, read `benchmarks.onnx_fp32_cuda` and
`benchmarks.onnx_fp16_cuda`: `latency.mean_ms`,
`efficiency.energy_per_image_mj`, and `efficiency.peak_gpu_memory_mib`.
The efficiency metadata records the energy source and whether memory is
process-scoped or a device-total fallback. As with other presets, rerunning
the same preset overwrites its own directory; preserve completed results
before repeating it. To restore full PTQ, re-enable calibration preparation,
INT8 quantization, the TensorRT probe and INT8 benchmark, and include `int8`
in the target precisions.

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
