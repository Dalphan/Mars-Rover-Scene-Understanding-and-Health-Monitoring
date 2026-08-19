# Mars Rover Scene Understanding and Health Monitoring

Research code for a thesis on Mars rover perception and wheel-condition
monitoring. The repository contains two independent Python projects that can
be installed, tested, and run from their own directories.

## Projects

| Project | Purpose |
| --- | --- |
| [`semantic_segmentation`](semantic_segmentation/README.md) | PyTorch semantic segmentation of Mars rover scenes from S5Mars. |
| [`wheel_anomaly_detection`](wheel_anomaly_detection/README.md) | Blender-based generation and validation of a Curiosity wheel dataset for visual anomaly detection. |

## Getting started

Choose a project, change into its directory, and follow the setup instructions
in its README. Project-specific virtual environments and dependencies are kept
separate so that Blender's Python environment remains isolated from ML
dependencies.

```bash
cd semantic_segmentation
# or
cd wheel_anomaly_detection
```

## Repository layout

```text
semantic_segmentation/     S5Mars data analysis, model training, and evaluation
wheel_anomaly_detection/   Blender data generation and anomaly-detection research
```

Original 3D assets, generated datasets, checkpoints, logs, and other runtime
artifacts remain local and are excluded from version control.
