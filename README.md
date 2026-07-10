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

All IDs, including label `0` (Background), are semantic classes. Training model
configs use `ignore_index = -100`; this sentinel is handled by the loss and does
not alter model outputs.

## Train Segmentation Models

Install dependencies, export `HF_TOKEN` if the dataset requires authentication, then run:

```bash
python -m src.train.train_segmentation
```

Supported model configurations:

```text
Default:
  SegFormer-B0
  model=segformer_b0
  checkpoint=nvidia/segformer-b0-finetuned-ade-512-512

SMP:
  U-Net ResNet34
  model=smp

  U-Net MobileNetV2
  model=smp model.encoder_name=mobilenet_v2

  DeepLabV3 ResNet34
  model=smp model.architecture=deeplabv3

  DeepLabV3+ MobileNetV2
  model=smp model.architecture=deeplabv3plus model.encoder_name=mobilenet_v2

LCNet (project-native, no pretrained weights):
  LCNet3_7
  model=lcnet3_7 freeze=none

  LCNet3_11
  model=lcnet3_11 freeze=none
```

Useful overrides:

```bash
python -m src.train.train_segmentation epochs=10 batch_size=4 optimizer.lr=0.00003
python -m src.train.train_segmentation splits.train=train[:100] splits.val=val[:20]
python -m src.train.train_segmentation model=smp freeze=encoder
python -m src.train.train_segmentation model=smp model.encoder_name=mobilenet_v2 freeze=encoder
python -m src.train.train_segmentation model=smp model.architecture=deeplabv3 criterion.name=combined
python -m src.train.train_segmentation model=smp model.architecture=deeplabv3plus model.encoder_name=mobilenet_v2
python -m src.train.train_segmentation num_workers=null
python -m src.train.train_segmentation model=lcnet3_7 freeze=none model_analysis.enabled=true
```

## LCNet and model analysis

`lcnet3_7` and `lcnet3_11` reimplement the architecture from Shi et al.,
“Lightweight Context-Aware Network Using Partial-Channel Transformation for
Real-Time Semantic Segmentation” (DOI: 10.1109/TITS.2023.3348631). They use all
nine S5Mars labels, return raw full-resolution logits, and do not download or
provide pretrained weights. The profiler depends on `fvcore` and uses eager
FP32 inference with input `[1, 3, 512, 512]`.

Enable the one-time analysis with `model_analysis.enabled=true`. GFLOPs is the
operation count for one inference, FPS is measured images per second, and
effective GFLOP/s is their product; it is not the hardware's theoretical peak.
Unsupported operations reported by `fvcore` are logged, so the GFLOP count may
be a partial count.

The default model config uses `nvidia/segformer-b0-finetuned-ade-512-512`, `num_labels=9`, and `ignore_index=-100`. `model=smp` defaults to U-Net ResNet34; change `model.architecture` and `model.encoder_name` for other SMP variants. Checkpoints are written under `outputs/${model.run_name}_s5mars/`. When `num_workers=null`, the DataLoader chooses a worker count from available CPU cores. If two GPUs are visible, training uses `torch.nn.DataParallel` on GPU `0` and `1`.

Training choices:

```text
freeze:
  none        train all parameters
  encoder     freeze pretrained backbone, train decoder/head
  classifier  train only the final segmentation head

criterion.name:
  cross_entropy     CE with ignore_index=-100 (all dataset labels remain valid)
  generalized_dice  Dice over all semantic classes, including class 0
  combined          alpha * CE + (1 - alpha) * Dice

criterion.weight_type:
  uniform  equal weight for classes present in the batch
  simple   inverse target volume
  square   inverse squared target volume
```
