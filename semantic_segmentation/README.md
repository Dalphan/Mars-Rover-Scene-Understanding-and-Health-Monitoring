# Mars Rover Semantic Segmentation

PyTorch research code for semantic segmentation of Mars rover scenes,
with S5Mars training, MarsSeg cross-mission evaluation, and separate
post-training quantization (PTQ) and quantization-aware training (QAT) workflows.

## Supported workflows

- SegFormer-B0, segmentation-models-pytorch architectures and LCNet variants.
- Cross-entropy, generalized Dice and combined losses.
- Optional training-only oversampling and geometric augmentation.
- Source-dataset evaluation and zero-shot MarsSeg MSL/MER transfer.
- ONNX export, FP32/FP16 benchmarks, INT8 PTQ and NVIDIA ModelOpt QAT.
- Self-contained Kaggle notebooks and matching standalone Python entrypoints.

## Data access

Dataset identifiers and taxonomies are defined in
[the training configuration](configs/train/segmentation.yaml):

| Dataset | Configured Hugging Face repository | Classes |
| --- | --- | ---: |
| S5Mars | `Mirali33/mb-s5mars` | 9 |
| MarsSeg MSL | `Mirali33/mb-mars_seg_msl` | 7 |
| MarsSeg MER | `Mirali33/mb-mars_seg_mer` | 7 |

S5Mars classes are Background, Bedrock, Hole, Ridge, Rock, Rover, Sand / Soil,
Sky and Track. The loader uses `image`, `mask`, `width`, `height` and
`class_labels`. Dataset files and trained checkpoints are not included.

Set `HF_TOKEN` in the environment when the selected dataset requires
authentication. Checkpoint downloads from Google Drive also require access
to the configured folders and your own credentials. Ordinary training does
not require Drive: `checkpoint.download_from_drive` defaults to `false`.

## Installation

From the repository root, enter this directory and create a dedicated
environment:

```bash
cd semantic_segmentation
python -m venv .venv
```

Activate it using the command for your shell:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Then install [the training dependencies](requirements.txt):

```bash
python -m pip install -r requirements.txt
```

The dependency file pins PyTorch 2.9.1 and torchvision 0.24.1 to match the
training notebook. GPU quantization has a separate dependency file.

## Quick start

All subsequent commands run from `semantic_segmentation/`.

Inspect a small dataset sample:

```bash
python scripts/analyze_dataset.py analysis.max_samples=5 visualization.num_samples=2 dataloader.num_workers=0
```

Start the default experiment:

```bash
python -m src.train.train_segmentation
```

The checked-in training preset uses LCNet3_7 from scratch, S5Mars, 30 epochs,
batch size 32, seed 42 and 512 x 512 inputs. Augmentation and oversampling are
disabled. This preset has a different training duration from the historical
60-epoch LCNet runs.

For a limited smoke run:

```bash
python -m src.train.train_segmentation epochs=1 batch_size=2 dataloader.train_num_workers=0 limits.train_batches=2 limits.val_batches=2 limits.test_batches=2 predictions.enabled=false
```

Examples of other configurations:

```bash
python -m src.train.train_segmentation model=lcnet3_11
python -m src.train.train_segmentation model=smp model.encoder_name=mobilenet_v2
python -m src.train.train_segmentation dataset_name=marsseg_msl run_cross_dataset_evaluation=true
python -m src.train.train_segmentation oversampling.enabled=true
```

Cross-dataset evaluation is supported between MarsSeg MSL and MER, which
share a taxonomy. It runs after source training, source-validation checkpoint
selection and source testing. S5Mars has a different taxonomy.

Analysis writes to `outputs/s5mars_analysis/`; training writes under
`outputs/`. Repeating an identical training configuration can reuse the same
output directory, so retain completed results before repeating an experiment.

## Kaggle notebooks

| Notebook | Purpose |
| --- | --- |
| [Segmentation training](notebooks/kaggle_segmentation_training.ipynb) | S5Mars/MarsSeg training and evaluation. |
| [ONNX PTQ](notebooks/kaggle_s5mars_onnx_ptq.ipynb) | FP32/FP16 baselines, ONNX export and configurable INT8 PTQ stages. |
| [ModelOpt QAT](notebooks/kaggle_s5mars_onnx_qat.ipynb) | Selective QAT and TensorRT comparison. |
| [Qualitative comparison](notebooks/kaggle_s5mars_qualitative_comparison.ipynb) | Twenty seeded S5Mars test-image comparisons from supplied checkpoints. |

Read each configuration cell before execution. Training controls include
`DATASET_NAME`, `LCNET_VARIANT`, `RUN_CROSS_DATASET_EVALUATION` and
`OVERSAMPLING_ENABLED`. Training data alone are augmented or oversampled.

The qualitative notebook requires the seven configured validation-selected
checkpoints and access to their Drive folders. These weights are not bundled
with the repository. Configure your own Drive references and Kaggle Secrets.
Dependency cells pin compatible package versions; restart the kernel when a
cell requests it after changing installed packages.

Notebooks that use Drive accept the credential names specified in their
configuration and setup cells, including `GDRIVE_CLIENT_ID`,
`GDRIVE_CLIENT_SECRET`, `GDRIVE_REFRESH_TOKEN` or `GOOGLE_*` aliases.
Drive uploads are optional; review their flags before running upload cells.
Download generated results before ending a Kaggle session.

## Standalone PTQ and QAT

Install [the additional GPU quantization dependencies](requirements-quantization.txt)
in the active environment:

```bash
python -m pip install -r requirements-quantization.txt
python -m src.quantization.ptq
python -m src.quantization.qat
```

Both workflows require a compatible trained FP32 checkpoint. Configure
checkpoint locations and Drive access before execution. QAT requires CUDA;
the runtime requirements and stage controls are recorded in
[the PTQ configuration](configs/quantization/ptq.yaml) and
[the QAT configuration](configs/quantization/qat.yaml).

The checked-in PTQ preset benchmarks DeepLabV3/ResNet34 FP32 and FP16 using an
existing checkpoint. INT8 stages and final hold-out testing are disabled in
that preset. Enable the required `steps.*` flags for a full PTQ campaign,
including calibration preparation and its dependent stages.

Recorded measurements and limitations are in
[PTQ results](docs/quantization/onnx_ptq_results.md) and
[QAT results](docs/quantization/onnx_qat_results.md).

## Configuration and verification

[Analysis](configs/config.yaml), [training](configs/train/segmentation.yaml),
[model presets](configs/model/), and the two quantization configurations
control the standalone workflows. Hydra command-line booleans use
`true` and `false`.

Evaluation-only execution uses
`execution.skip_train=true execution.load_checkpoint=true`;
`execution.load_best=true` selects `best.ckpt`. Supply the checkpoint
for the matching dataset and experiment before running this mode.
During normal training, the same load flags select the checkpoint for
evaluation after training.

Training, PTQ and QAT notebooks define the experiment specifications.
Python modules and YAML files mirror them in one direction. The
[synchronization guide](docs/notebook_python_sync.md) explains the contract.

The available read-only configuration check is:

```bash
python scripts/check_notebook_config_sync.py
```

The current checkout does not contain a `tests/` directory. The configuration
check is not a replacement for model training, inference or GPU validation.

## Project structure

```text
configs/          Dataset, architecture and experiment configurations
docs/             Synchronization contract and recorded quantization results
notebooks/        Self-contained Kaggle workflows
scripts/          Dataset inspection, configuration checks and OAuth setup
src/data/         Dataset loading and transforms
src/models/       Segmentation architectures
src/losses/       Loss functions
src/metrics/      Evaluation metrics
src/train/        Training and evaluation
src/quantization/ PTQ/QAT stages and shared runtime utilities
```
