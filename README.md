# Mars-Bench S5Mars Segmentation

PyTorch utilities for semantic segmentation data loading, analysis, visualization, and SegFormer-B0 training on the Hugging Face dataset `Mirali33/mb-s5mars`.

The training path uses pure PyTorch, Hydra, and Hugging Face Transformers. It does not use PyTorch Lightning.

## Setup

Use Python 3.11 or 3.12 for the Hydra CLI path. The dependency pins in `requirements.txt` target that stable combination.

```bash
pip install -r requirements.txt
```

If the dataset requires authentication, export a Hugging Face token before running:

```bash
export HF_TOKEN=your_token_here
```

Do not commit real tokens. The Hydra config keeps only `HF_TOKEN_PLACEHOLDER`.

## Run Analysis

```bash
python scripts/analyze_dataset.py
```

Useful overrides:

```bash
python scripts/analyze_dataset.py dataset.split=val analysis.max_samples=50
python scripts/analyze_dataset.py transforms.resize.height=256 transforms.resize.width=256 dataloader.batch_size=8
python scripts/analyze_dataset.py dataset.split=partition_train_0.10x_partition analysis.max_samples=100
```

`analysis.max_samples` limits analysis and visualization loops after the configured Hugging Face split is loaded. It does not rewrite `dataset.split` to a sliced split.

Acceptance smoke test:

```bash
python scripts/analyze_dataset.py analysis.max_samples=5 visualization.num_samples=2
```

Expected outputs are written under `outputs/s5mars_analysis/`:

```text
run.log
dataset_summary.json
image_level_class_distribution.csv
mask_pixel_distribution.csv
ignore_pixel_ratio.json
visualizations/
  sample_grid.png
  sample_*.png
```

## Project Structure

```text
configs/config.yaml
src/data/
src/analysis/
src/utils/
scripts/analyze_dataset.py
notebooks/kaggle_s5mars_end_to_end.ipynb
requirements.txt
```

## Dataset Assumptions

The loader expects each sample to expose:

- `image`: PIL image
- `mask`: PIL segmentation mask
- `width`: integer
- `height`: integer
- `class_labels`: list of class names

Class IDs:

```text
0 Background
1 Bedrock
2 Hole
3 Ridge
4 Rock
5 Rover
6 Sand / Soil
7 Sky
8 Track
```

`ignore_index = 0` is used for statistics where background should be excluded from valid target-class percentages.

## Train SegFormer-B0

Install dependencies, export `HF_TOKEN` if the dataset requires authentication, then run:

```bash
python -m src.train.train_segmentation
```

Useful overrides:

```bash
python -m src.train.train_segmentation epochs=10 batch_size=4 optimizer.lr=0.00003
python -m src.train.train_segmentation splits.train=train[:100] splits.val=val[:20]
```

The model config uses `nvidia/segformer-b0-finetuned-ade-512-512`, `num_labels=9`, and `CrossEntropyLoss(ignore_index=0)`. Checkpoints are written under `outputs/segformer_b0_s5mars/`.
