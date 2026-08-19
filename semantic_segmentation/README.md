# Mars Rover Semantic Segmentation

A PyTorch training pipeline for semantic segmentation of Mars rover scenes.
The project uses the [Mars-Bench S5Mars dataset](https://huggingface.co/datasets/Mirali33/mb-s5mars)
and predicts nine terrain and rover classes from RGB images.

## Highlights

- Configuration-driven experiments with Hydra.
- Dataset inspection with class-distribution reports and visual samples.
- SegFormer-B0, `segmentation-models-pytorch`, and native LCNet variants.
- Cross-entropy, generalized Dice, and combined segmentation losses.
- Pixel accuracy, mean IoU, per-class IoU, checkpoints, and reproducible
  seeded runs.

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

Start the default SegFormer-B0 experiment:

```bash
python -m src.train.train_segmentation
```

Useful Hydra overrides include:

```bash
python -m src.train.train_segmentation epochs=10 batch_size=4
python -m src.train.train_segmentation model=smp model.encoder_name=mobilenet_v2
python -m src.train.train_segmentation model=lcnet3_7 freeze=none model_analysis.enabled=true
```

Analysis results are written to `outputs/s5mars_analysis/`; training runs are
written under `outputs/`. Both locations are intentionally ignored by Git.

## Configuration

The base data configuration is in `configs/config.yaml`, while training and
model presets live in `configs/train/` and `configs/model/`. The default
training setup uses 512 × 512 inputs, AdamW with cosine scheduling, and tracks
validation mean IoU for best-checkpoint selection.

## Project structure

```text
configs/       Hydra data, model, and training presets
scripts/       Dataset analysis entry point
src/data/      Dataset loading and transforms
src/models/    Segmentation model implementations
src/losses/    Loss functions
src/metrics/   Evaluation metrics
src/train/     Training and evaluation loops
tests/         Unit tests
```

## Tests

```bash
python -m pytest -q
```
