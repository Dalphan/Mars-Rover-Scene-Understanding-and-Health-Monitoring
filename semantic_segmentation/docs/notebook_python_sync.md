# Notebook → Python synchronization contract

The three notebooks are the authoritative experiment specifications. The
standalone code is a one-way mirror:

| Authoritative notebook | Standalone configuration | Entrypoint |
|---|---|---|
| `notebooks/kaggle_segmentation_training.ipynb` | `configs/train/segmentation.yaml` | `python -m src.train.train_segmentation` |
| `notebooks/kaggle_s5mars_onnx_ptq.ipynb` | `configs/quantization/ptq.yaml` | `python -m src.quantization.ptq` |
| `notebooks/kaggle_s5mars_onnx_qat.ipynb` | `configs/quantization/qat.yaml` | `python -m src.quantization.qat` |

The only structural difference is configuration placement: notebook constants
stay in their configuration cells, while the Python entrypoints read YAML via
Hydra. Host-relative output paths are YAML values rather than Kaggle literals.
The Python code must never be used to generate or overwrite notebook cells.

## Standalone module layout

The command modules are intentionally thin: they load Hydra configuration and
delegate to a pipeline. The synchronized implementation is split by
responsibility rather than copied into one generated Python file. Files may be
up to roughly 500 lines when the contained operations are cohesive.

| Pipeline | Main modules |
|---|---|
| Training | `run_context.py` resolves dataset/run identity; `experiment.py` builds loaders, model and loss; `training_loop.py` and `training_stage.py` train and validate; `checkpoint_stage.py` handles checkpoint I/O; `evaluation_stage.py` performs source and cross-dataset evaluation; `reporting.py` writes metadata and predictions. |
| PTQ | `core/` contains shared data/model/ONNX/metric utilities; `ptq_baselines.py` handles FP32/FP16; `ptq_calibration.py` owns calibration; `ptq_graph_rewrite.py`, `ptq_int8_audit.py` and `ptq_int8.py` create and verify INT8; `ptq_tensorrt.py` probes TensorRT; `ptq_final.py` owns the final hold-out. |
| QAT | `qat_setup.py` builds runtime state; `qat_stage.py` coordinates quantizer preparation and training; `qat_export.py`, `qat_onnx_audit.py` and `qat_fp16_export.py` own ONNX artifacts; `qat_engine.py` and `qat_benchmark.py` own TensorRT execution and measurement; `qat_final.py` owns final evaluation and artifact upload. |

`pipeline.py`, `ptq_pipeline.py`, and `qat_pipeline.py` only coordinate those
stages. `qat_tensorrt.py` is a small compatibility facade that re-exports the
engine and benchmark API; it contains no duplicate implementation.

## Update procedure

1. Change and validate the relevant notebook first.
2. Copy the changed behavior into the matching Python module and copy constants
   into its YAML file.
3. Run `python scripts/check_notebook_config_sync.py`. It parses literal
   notebook assignments without executing cells and compares them with YAML.
4. Run `python -m pytest -q` and the appropriate smoke run. A mismatch is fixed
   in Python/YAML; the checker never edits the notebook.

## Adversarial review and trade-offs

- **Taxonomy validity:** S5Mars has 9 classes; MarsSeg has 7. Model output
  channels are resolved from the selected source dataset. Cross-dataset mode is
  accepted only for MSL↔MER after exact taxonomy comparison.
- **Leakage prevention:** cross-target data are constructed only after source
  training, source-validation checkpoint selection, and source testing. The
  target test split is never used for tuning.
- **Background validity:** `ignore_index=-100` lies outside every mask range,
  so Background is a real class in loss and mIoU. Metrics mask the ignore index
  only when it is a valid class ID.
- **Memory/scale:** training workers return compact NumPy `uint8` arrays and
  normalization occurs in the main process. Oversampling reads image-level
  metadata without decoding every mask and caps inverse-rarity weights.
- **Checkpoint safety:** dataset, repository, architecture, class count,
  sampling, loss weighting, and augmentation tags are checked before loading
  weights. Old checkpoints without dataset identity are rejected because their
  source cannot be verified safely. PTQ and QAT downloads use experiment-scoped
  directories so identically named `best.ckpt` files cannot collide.
- **Optional GPU stack:** TensorRT, ONNX Runtime GPU, NVML, and ModelOpt are
  isolated in `requirements-quantization.txt` and imported lazily. Training can
  run without them. PTQ reports CPU fallback explicitly; QAT intentionally
  requires CUDA.
- **Deployment validity:** PTQ checks ONNX structure, calibration provenance,
  Q/DQ coverage, TensorRT execution, parity, latency, and NVML efficiency. QAT
  audits selected ModelOpt quantizers, folds constant INT8 weights, checks
  export/TensorRT parity, and keeps the test split behind an explicit final
  gate.
- **Module boundaries:** data construction, optimization, checkpoint I/O,
  graph mutation, runtime execution, and reporting are kept separate. This
  adds a few explicit stage interfaces, but prevents the notebook mirror from
  becoming a single hard-to-review script.

The price of a notebook-authoritative workflow is deliberate duplication. The
read-only sync gate catches constant drift; behavior changes still require a
code review and smoke test because equivalence cannot be proven from literals
alone.
