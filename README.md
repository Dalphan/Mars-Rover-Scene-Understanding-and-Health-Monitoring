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

## Blender GLB wheel audit

The Blender pipeline is deliberately isolated from the training code. Host-side
launching and validation live under `scripts/host/`; scripts that import `bpy`
live under `scripts/blender/`; Blender configuration is in
`configs/blender/audit.json`. Original assets belong in `assets/original/` or may
be passed from another local path. They are treated as immutable and ignored by
Git.

This first phase only audits editability. It does not create anomalies, perform
domain randomization, generate a dataset, run PatchCore, or install ML
dependencies into Blender.

### Requirements

- Blender 5.2.0 LTS;
- the original NASA `.glb` already present locally;
- a regular host Python for the optional launcher, validator and tests;
- no additional Python package is required by the Blender audit.

The default render configuration is Eevee at 800x600 (4:3). The audit records
the actual render backend and warns if no hardware GPU context is detected.

### Direct headless command

From the repository root in PowerShell:

```powershell
& 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --background `
  --python scripts/blender/audit_asset.py `
  -- `
  --asset '..\24584_Curiosity_static.glb' `
  --output-dir 'outputs\blender_audit\curiosity_static'
```

The equivalent generic form is:

```text
blender --background --python <script_audit> -- --asset <path_asset.glb> --output-dir <directory_output>
```

To override the checked-in standard-library-only configuration, append:

```text
--config configs/blender/audit.json
```

### Host launcher and validation

The host launcher runs Blender and validates all outputs when Blender exits:

```powershell
python scripts/host/run_blender_audit.py `
  --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --asset '..\24584_Curiosity_static.glb' `
  --output-dir 'outputs\blender_audit\curiosity_static'
```

Validation can also be rerun independently:

```powershell
python scripts/host/validate_blender_audit.py `
  --output-dir 'outputs\blender_audit\curiosity_static'
```

Run only the host-side unit tests (standard library only; they never import
`bpy`):

```powershell
python -m unittest discover `
  -s tests `
  -p 'test_blender_audit*.py' `
  -v
```

### Output layout

```text
outputs/blender_audit/<run>/
  reports/
    audit.json
    audit.md
  diagnostics/
    imported_asset.blend
  renders/
    rover/                 # six axis-aligned overview directions
    candidates/            # at least three close-ups per candidate
    topology/              # wireframe/topology overview
  logs/
    audit.log
```

The report inventories collections, hierarchy, parents, transforms, bounding
boxes, topology, materials, UVs, disconnected components, normals and
non-manifold edges. Candidate wheels are scored from independent signals:
names, cylindrical proportions, scene position and repeated
geometry/topology. The final category is:

- A: separate wheel directly usable;
- B: wheel automatically separable from a larger mesh;
- C: minimal manual correction required;
- D: unsuitable for the requested geometry edits.

The source GLB is hashed before and after execution. Only the imported in-memory
copy and the diagnostic `.blend` are written. A `.gltf` is never downloaded,
generated or substituted automatically. If Blender imports unresolved
textures/resources, the report sets `try_gltf_required` and explains which
separate URI or buffer would need inspection; otherwise it explicitly states
that trying glTF is unnecessary.

## Blender wheel preparation handoff

The repository also contains the second-stage prototype that extracts
`wheel_candidate_05`, creates a canonical wheel, evaluates repair options, and
renders a deterministic normal/perforation pair:

```powershell
python scripts/host/run_wheel_preparation.py `
  --blender 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe' `
  --asset '<path>\24584_Curiosity_static.glb' `
  --audit-report 'outputs\blender_audit\curiosity_static\reports\audit.json' `
  --candidate-id wheel_candidate_05 `
  --output-dir 'outputs\wheel_preparation\curiosity_middle_right'
```

This stage currently passes structural validation but has not passed final
visual QA and must not yet be used for bulk dataset generation. Before
continuing, read [AGENTS.md](AGENTS.md) and the complete
[Blender anomaly-detection handoff](docs/handoff/blender_anomaly_detection.md).
The reproducible target is Blender 5.2.0; using 5.1 requires regenerating the
audit report with 5.1 rather than reusing a 5.2 report.
