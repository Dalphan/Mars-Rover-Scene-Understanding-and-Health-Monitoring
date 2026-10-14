# Mars Rover Scene Understanding and Health Monitoring

Research code and dissertation materials for **Lightweight Scene Understanding
and Synthetic Data-Driven Damage Inspection for Martian Rovers**. The work
covers semantic segmentation of Martian terrain and visual anomaly detection
of Curiosity rover wheel damage.

## Projects

| Directory | Purpose | Start here |
| --- | --- | --- |
| `semantic_segmentation/` | S5Mars and MarsSeg segmentation, training, evaluation, PTQ and QAT. | [Segmentation guide](semantic_segmentation/README.md) |
| `wheel_anomaly_detection/` | Blender scene generation, paired wheel-damage data and anomaly-detection experiments. | [Wheel anomaly-detection guide](wheel_anomaly_detection/README.md) |
| `thesis/` | LaTeX dissertation, reusable figures and presentation materials. | [Thesis guide](thesis/README.md) |

The two code projects are independent. Run commands from the project directory
and use a separate Python environment for each project. Blender uses its own
bundled Python; ML dependencies belong in the host environment.

## Getting started

1. Choose one of the project guides above.
2. Install its dependencies in a dedicated environment.
3. Supply the datasets or checkpoints required by the selected workflow.
4. Run the data inspection command before starting a full experiment.

For example, from the repository root:

```bash
cd semantic_segmentation
# or
cd wheel_anomaly_detection
```

The checked-in Kaggle notebooks are self-contained. Their configuration cells
define model selection, data access, output locations and optional Google
Drive operations. Configure those cells for your own environment before
execution.

## Repository contents

This repository includes source code, experiment configurations, documentation,
notebooks, the dissertation sources and PDF, and presentation/figure materials.

Datasets, original 3D/geospatial assets, trained checkpoints, caches and runtime
logs are local resources excluded from Git. A fresh clone therefore requires
data preparation or a compatible dataset supplied separately. Configured
Google Drive references identify research runs; they do not provide public
access to the associated checkpoints.

## Documentation

- [Segmentation notebook/Python synchronization](semantic_segmentation/docs/notebook_python_sync.md)
- [Segmentation PTQ results](semantic_segmentation/docs/quantization/onnx_ptq_results.md)
- [Segmentation QAT results](semantic_segmentation/docs/quantization/onnx_qat_results.md)
- [Wheel dataset generation and provenance](wheel_anomaly_detection/docs/generation/INDEX.md)
- [Anomaly-detection evaluation and experiment history](wheel_anomaly_detection/docs/anomaly_detection/README.md)
- [Dissertation and presentation](thesis/README.md)

Recorded experiments have their own preprocessing, training budgets and
hardware conditions. Use the linked reports and run configurations when
interpreting results or reproducing a particular experiment.

## License and attribution

A project-wide open-source license has not been specified. The bundled
Sapthesis class retains its [upstream license and attribution notice](thesis/latex/README).
Third-party datasets, model weights, illustrations and presentation materials
retain their respective terms and attribution.
