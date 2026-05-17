# Mars Rover AI Thesis – Knowledge Base

## Project overview

This thesis explores AI for planetary rovers, with the long-term goal of combining:

1. semantic segmentation of Mars rover imagery,
2. path planning / traversability estimation,
3. rover self-health monitoring.

For the current development phase, the focus is **Mars terrain segmentation** using **Mars-Bench** and **AI4MARS**. The segmentation module will be the foundation for later traversal planning and health-monitoring work.

## Current scope

Build a strong and reproducible segmentation pipeline that can:

* segment Martian terrain from rover images,
* distinguish traversable and non-traversable terrain elements,
* provide a clean baseline for later path-planning experiments,
* support future rover self-inspection / anomaly detection modules.

## Datasets selected

### AI4MARS

Use this as the main large-scale training dataset.

* Contains rover imagery from Mars missions.
* Includes pixel-level terrain annotations.
* Useful for training and benchmarking semantic segmentation models on Martian terrain.
* Best suited for learning terrain categories such as soil, bedrock, sand, rock, shadow, sky, and tracks.

### Mars-Bench

Use this as the more recent benchmark and evaluation reference.

* Contains Mars-related benchmark tasks and segmentation subsets.
* Useful for testing generalization and comparing against more recent Mars vision baselines.
* Ideal for validating whether the model trained on AI4MARS transfers well to a newer benchmark.

## Main research question

Can a modern segmentation model trained on public Mars rover datasets learn robust terrain segmentation that generalizes across missions and image conditions, and later serve as the basis for traversability-aware navigation?

## Why segmentation first

Segmentation provides the visual understanding layer needed for health monitoring and self-inspection later.

## Intended thesis direction

The thesis should start with **single-image semantic segmentation** and later extend to:

* terrain cost mapping,
* path suggestion / traversability estimation,
* rover self-health monitoring as a future extension.

The first milestone is therefore a reliable segmentation system, not a full autonomous rover stack.

## Preferred model families

Start with efficient and well-known segmentation architectures in PyTorch.
Recommended candidates:

* **SegFormer** as the main baseline because it is strong and relatively lightweight,
* **DeepLabv3+** as a classic baseline,
* **Mask2Former** only if instance-level or more advanced segmentation is needed.

The first experiments should remain focused and simple:

* train a SegFormer baseline,
* compare against a simpler CNN-based baseline,
* evaluate cross-dataset generalization.

## Data strategy

### Training strategy

* Train primarily on **AI4MARS**.
* Validate on an internal split of AI4MARS.
* Test transferability on **Mars-Bench**.

### Possible experiments

* Train on AI4MARS only, test on Mars-Bench.
* Train on AI4MARS and fine-tune on a small Mars-Bench subset.
* Compare with a model trained from scratch on Mars-Bench.

### Augmentation

Use augmentation heavily, but keep it realistic for Mars:

* random rotations,
* flips when physically acceptable,
* brightness and contrast jitter,
* Gaussian noise,
* blur,
* dust / haze-like perturbations if useful,
* random crop / resize.

Avoid augmentations that destroy the planetary geometry or create unrealistic terrain textures.

## Output classes

Use a compact class set at the beginning.
Suggested labels:

* rock
* soil / sand
* bedrock
* shadow
* sky
* track / wheel mark
* rover (if present in the image)

Do not make the label taxonomy too large in the first phase.

## Evaluation metrics

Primary metrics:

* mean IoU
* per-class IoU
* pixel accuracy
* F1 score if relevant

Secondary metrics:

* inference latency
* parameter count
* model size
* memory usage

For later lightweight deployment, efficiency metrics matter as much as accuracy.

## Development priorities

1. Build a clean dataset loader for AI4MARS and Mars-Bench (Use Hugging Face for Mars-Bench)
2. Implement a train/val/test pipeline in PyTorch-Lightning.
3. Train a baseline segmentation model.
4. Evaluate on AI4MARS.
5. Test transfer to Mars-Bench.
6. Document failure cases and class confusion.
7. Only after segmentation is stable, move to traversability estimation.

## Expected challenges

* domain shift between datasets,
* strong lighting differences,
* shadows that look like terrain obstacles,
* small rocks and thin boundaries,
* imbalance between common terrain classes and rare classes.

## Suggested technical approach

* Use PyTorch-Lightning for training and inference.
* Use Hydra for config management.
* Start with a simple and reproducible codebase.
* Keep the training loop readable and modular.
* Save configs in YAML or JSON.
* Log metrics and qualitative predictions consistently.
* I have to work in a Jupyter notebook, but write also clean Python scripts for training and evaluation that can be run from the command line.

## Recommended repository structure

```text
project/
├── data/
├── datasets/
├── models/
├── train/
├── eval/
├── utils/
├── configs/
├── notebooks/
├── outputs/
└── README.md
```

## Coding conventions for Codex

When generating code, follow these rules:

* write modular PyTorch code,
* prefer readability over cleverness,
* keep dataset-specific logic isolated,
* make config-driven experiments,
* include clear docstrings and comments,
* save checkpoints and metrics automatically,
* make scripts runnable from the command line.

## What Codex should optimize for

* reproducibility,
* clean code structure,
* easy dataset swapping between AI4MARS and Mars-Bench,
* strong baseline results,
* a future path toward navigation and rover health monitoring.

## Long-term vision

This segmentation module is only the first layer of a broader planetary rover intelligence stack:

* segmentation → terrain understanding,
* terrain understanding → traversability / path planning,
* rover-focused vision → self-health monitoring.

For now, everything should be built so that these future modules can plug into the segmentation output later.
