# Mars Rover Terrain Segmentation

Baseline semantic segmentation pipeline for Mars rover imagery using PyTorch Lightning and Hydra. The current stable path trains SegFormer on a shared core taxonomy with three classes: bedrock, loose regolith, and rock. Dataset adapters remap native AI4MARS and Mars-Bench labels to this taxonomy and use `255` as `ignore_index`.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

### AI4MARS (local)

The loader can discover common layouts under `data/ai4mars`, including `images/masks`, `edr/mxy/rng`, and nested split folders. The legacy split-file layout is still supported through [configs/data/ai4mars.yaml](configs/data/ai4mars.yaml):

```
data/ai4mars/
	images/
	masks/
	splits/
		train.txt
		val.txt
		test.txt
```

Each split file is a text file with one sample per line:

```
relative/image.png relative/mask.png
```

If only one path is provided, the same filename is used for the mask.

### Mars-Bench (Hugging Face)

The loader supports these Hugging Face datasets:

- `Mirali33/mb-mars_seg_mer`
- `Mirali33/mb-mars_seg_msl`
- `Mirali33/mb-s5mars`

Set `hf_dataset`, `hf_config`, and the `image_key`/`mask_key` in the data config if needed.

## Training

Train SegFormer-B0 on AI4MARS:

```bash
python -m train.train
```

Run the explicit core baseline config:

```bash
python -m train.train --config-name exp_segformer_core
```

Train DeepLabV3+ MobileNetV3:

```bash
python -m train.train model=deeplabv3plus_mnv3
```

Adjust batch size, image size, and augmentations in the data config files.

## Evaluation

Evaluate a checkpoint (example on Mars-Bench):

```bash
python -m eval.eval ckpt_path=outputs/2026-05-17/12-00-00/last.ckpt data=mars_bench
```

## Audit

Audit configured dataset splits before training:

```bash
python -m scripts.audit_datasets data=ai4mars
python -m scripts.audit_datasets data=mars_bench
```

## Notes

- Dataset-specific labels are mapped centrally in [taxonomies.py](taxonomies.py).
- The first tranche intentionally implements only the core taxonomy. Rover, track, sky, shadow, hazard, background, and null pixels are ignored for core training.
- `REPO_VERSION.txt` is a simple numeric counter. Increase it before each commit with `python scripts/update_repo_version.py` so Kaggle clones can tell which repo version they have.
