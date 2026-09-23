"""Verify the one-way notebook -> Python configuration contract.

This command is intentionally read-only.  Notebook constants are parsed with
``ast.literal_eval`` and compared with their standalone YAML counterparts.
When a notebook changes, this check must fail until the Python config/modules
are deliberately synchronized from that notebook.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from omegaconf import OmegaConf


TRAINING_FIELDS = {
    "EXPECTED_TORCH_VERSION": "runtime.expected_torch_version",
    "DATASET_NAME": "dataset_name",
    "RUN_CROSS_DATASET_EVALUATION": "run_cross_dataset_evaluation",
    "HF_TOKEN": "huggingface.token",
    "TRAIN_SPLIT": "splits.train",
    "VAL_SPLIT": "splits.val",
    "TEST_SPLIT": "splits.test",
    "SEED": "seed",
    "IMAGE_SIZE": "image_size",
    "IGNORE_INDEX": "ignore_index",
    "EPOCHS": "epochs",
    "BATCH_SIZE": "batch_size",
    "NUM_WORKERS": "dataloader.train_num_workers",
    "EVAL_NUM_WORKERS": "dataloader.eval_num_workers",
    "DATALOADER_PIN_MEMORY": "dataloader.pin_memory",
    "DATALOADER_PREFETCH_FACTOR": "dataloader.prefetch_factor",
    "DATALOADER_PERSISTENT_WORKERS": "dataloader.persistent_workers",
    "TORCH_MULTIPROCESSING_SHARING_STRATEGY": "runtime.torch_multiprocessing_sharing_strategy",
    "LOG_MEMORY_USAGE": "memory.log_usage",
    "MEMORY_LOG_INTERVAL_BATCHES": "memory.log_interval_batches",
    "FREEZE": "freeze",
    "LR": "optimizer.lr",
    "WEIGHT_DECAY": "optimizer.weight_decay",
    "WARMUP_EPOCHS": "scheduler.warmup_epochs",
    "CRITERION_NAME": "criterion.name",
    "CRITERION_ALPHA": "criterion.alpha",
    "CRITERION_WEIGHT_TYPE": "criterion.weight_type",
    "CRITERION_SMOOTH": "criterion.smooth",
    "TRAIN_AUGMENTATION_ENABLED": "augmentation.enabled",
    "TRAIN_HORIZONTAL_FLIP_PROB": "augmentation.horizontal_flip_prob",
    "TRAIN_VERTICAL_FLIP_PROB": "augmentation.vertical_flip_prob",
    "TRAIN_RANDOM_ROTATE90_PROB": "augmentation.random_rotate90_prob",
    "OVERSAMPLING_ENABLED": "oversampling.enabled",
    "OVERSAMPLING_POWER": "oversampling.power",
    "OVERSAMPLING_MAX_WEIGHT": "oversampling.max_weight",
    "OVERSAMPLING_NUM_SAMPLES_MULTIPLIER": "oversampling.num_samples_multiplier",
    "OVERSAMPLING_REPLACEMENT": "oversampling.replacement",
    "OVERSAMPLING_EXCLUDED_CLASS_IDS": "oversampling.excluded_class_ids",
    "MODEL_ANALYSIS_ENABLED": "model_analysis.enabled",
    "MODEL_ANALYSIS_WARMUP_ITERATIONS": "model_analysis.warmup_iterations",
    "MODEL_ANALYSIS_MEASUREMENT_ITERATIONS": "model_analysis.measurement_iterations",
    "LIMIT_TRAIN_BATCHES": "limits.train_batches",
    "LIMIT_VAL_BATCHES": "limits.val_batches",
    "SKIP_TRAIN": "execution.skip_train",
    "LOAD_CKPT": "execution.load_checkpoint",
    "LOAD_BEST": "execution.load_best",
    "DOWNLOAD_CHECKPOINT_FROM_DRIVE": "checkpoint.download_from_drive",
    "FORCE_CHECKPOINT_DOWNLOAD": "checkpoint.force_download",
    "DRIVE_CHECKPOINT_FOLDER_IDS": "checkpoint.drive_folder_ids",
    "DRIVE_CHECKPOINT_FOLDER_ID_OVERRIDE": "checkpoint.drive_folder_id_override",
}


PTQ_FIELDS = {
    "REPO_ID": "dataset.repo_id",
    "CALIBRATION_SPLIT": "dataset.calibration_split",
    "EVALUATION_SPLIT": "dataset.evaluation_split",
    "TEST_SPLIT": "dataset.test_split",
    "IMAGE_SIZE": "dataset.image_size",
    "NUM_CLASSES": "dataset.num_classes",
    "IMAGE_MEAN": "dataset.image_mean",
    "IMAGE_STD": "dataset.image_std",
    "IGNORE_INDEX": "dataset.ignore_index",
    "CLASS_NAMES": "dataset.class_names",
    "MODEL_NAME": "model.name",
    "PRETRAINED_NAME": "model.pretrained_name",
    "SMP_ARCHITECTURE": "model.smp_architecture",
    "SMP_ENCODER_NAME": "model.smp_encoder_name",
    "SMP_ENCODER_WEIGHTS": "model.smp_encoder_weights",
    "DRIVE_CHECKPOINT_FOLDERS": "checkpoint.drive_folders",
    "CHECKPOINT_EXPERIMENT": "checkpoint.experiment",
    "CHECKPOINT_FOLDER_ID_OVERRIDE": "checkpoint.folder_id_override",
    "CHECKPOINT_FILENAME": "checkpoint.filename",
    "BATCH_SIZE": "data.batch_size",
    "NUM_WORKERS": "data.num_workers",
    "MAX_CALIBRATION_SAMPLES": "data.max_calibration_samples",
    "CALIBRATION_DRY_RUN_SAMPLES": "data.calibration_dry_run_samples",
    "MAX_EVALUATION_SAMPLES": "data.max_evaluation_samples",
    "MAX_TEST_SAMPLES": "data.max_test_samples",
    "CALIBRATION_SELECTION": "data.calibration_selection",
    "SUBSET_SEED": "seed",
    "PREFER_GPU": "runtime.prefer_gpu",
    "PREFER_TENSORRT": "runtime.prefer_tensorrt",
    "TARGET_PRECISIONS": "runtime.target_precisions",
    "RUN_SMOKE_TEST": "steps.run_smoke_test",
    "RUN_FP32_BASELINE": "steps.run_fp32_baseline",
    "RUN_INT8_CALIBRATION_PREPARATION": "steps.run_int8_calibration_preparation",
    "RUN_ONNX_EXPORT": "steps.run_onnx_export",
    "RUN_ONNX_FP32_BENCHMARK": "steps.run_onnx_fp32_benchmark",
    "RUN_ONNX_FP16": "steps.run_onnx_fp16",
    "RUN_ONNX_INT8_QUANTIZATION": "steps.run_onnx_int8_quantization",
    "RUN_TRT_INT8_PROBE": "steps.run_trt_int8_probe",
    "RUN_ONNX_INT8_BENCHMARK": "steps.run_onnx_int8_benchmark",
    "RUN_FINAL_TEST_EVALUATION": "steps.run_final_test_evaluation",
    "LATENCY_WARMUP_RUNS": "benchmark.latency_warmup_runs",
    "LATENCY_MEASURED_RUNS": "benchmark.latency_measured_runs",
    "EFFICIENCY_WARMUP_RUNS": "benchmark.efficiency_warmup_runs",
    "EFFICIENCY_MIN_RUNS": "benchmark.efficiency_min_runs",
    "EFFICIENCY_MIN_SECONDS": "benchmark.efficiency_min_seconds",
    "NVML_SAMPLE_INTERVAL_SECONDS": "benchmark.nvml_sample_interval_seconds",
    "ONNX_OPSET_VERSION": "onnx.opset_version",
    "FP16_KEEP_IO_TYPES": "onnx.fp16_keep_io_types",
    "INT8_QUANT_FORMAT": "int8.quant_format",
    "INT8_CALIBRATION_METHOD": "int8.calibration_method",
    "INT8_ACTIVATION_TYPE": "int8.activation_type",
    "INT8_WEIGHT_TYPE": "int8.weight_type",
    "INT8_PER_CHANNEL": "int8.per_channel",
    "INT8_REDUCE_RANGE": "int8.reduce_range",
    "INT8_SYMMETRIC": "int8.symmetric",
    "INT8_OP_TYPES": "int8.op_types",
    "INT8_DEDICATED_QDQ_PAIR": "int8.dedicated_qdq_pair",
    "INT8_QUANTIZE_BIAS": "int8.quantize_bias",
    "INT8_DECOMPOSE_CONV_BIAS": "int8.decompose_conv_bias",
    "INT8_CALIBRATION_CHUNK_SIZE": "int8.calibration_chunk_size",
    "INT8_CALIBRATION_PERCENTILE": "int8.calibration_percentile",
    "TRT_DEVICE_ID": "tensorrt.device_id",
    "TRT_INT8_ENABLE": "tensorrt.int8_enable",
    "TRT_FP16_ENABLE": "tensorrt.fp16_enable",
}


QAT_FIELDS = {
    "REPO_ID": "dataset.repo_id",
    "TRAIN_SPLIT": "dataset.train_split",
    "VAL_SPLIT": "dataset.val_split",
    "TEST_SPLIT": "dataset.test_split",
    "IMAGE_SIZE": "dataset.image_size",
    "IMAGE_MEAN": "dataset.image_mean",
    "IMAGE_STD": "dataset.image_std",
    "NUM_CLASSES": "dataset.num_classes",
    "IGNORE_INDEX": "dataset.ignore_index",
    "CLASS_NAMES": "dataset.class_names",
    "MODEL_NAME": "model.name",
    "PRETRAINED_NAME": "model.pretrained_name",
    "EXPERIMENT": "model.experiment",
    "QAT_VARIANT": "model.variant",
    "INT8_LINEAR_GROUPS": "model.int8_linear_groups",
    "FP16_FALLBACK_MODULE_PREFIXES": "model.fp16_fallback_module_prefixes",
    "CHECKPOINT_FILENAME": "checkpoint.filename",
    "CHECKPOINT_FOLDER_ID": "checkpoint.folder_id",
    "SEED": "seed",
    "TRAIN_BATCH_SIZE": "training.train_batch_size",
    "EVAL_BATCH_SIZE": "training.eval_batch_size",
    "CALIBRATION_BATCH_SIZE": "training.calibration_batch_size",
    "GRADIENT_ACCUMULATION_STEPS": "training.gradient_accumulation_steps",
    "NUM_WORKERS": "training.num_workers",
    "CALIBRATION_SAMPLES": "training.calibration_samples",
    "EPOCHS": "training.epochs",
    "EARLY_STOPPING_PATIENCE": "training.early_stopping_patience",
    "LEARNING_RATE": "training.learning_rate",
    "WEIGHT_DECAY": "training.weight_decay",
    "WARMUP_EPOCHS": "training.warmup_epochs",
    "USE_AMP": "training.use_amp",
    "SUPERVISED_WEIGHT": "training.supervised_weight",
    "DISTILLATION_WEIGHT": "training.distillation_weight",
    "DISTILLATION_TEMPERATURE": "training.distillation_temperature",
    "COMBINED_LOSS_ALPHA": "training.combined_loss_alpha",
    "DICE_SMOOTH": "training.dice_smooth",
    "MAX_TRAIN_SAMPLES": "training.max_train_samples",
    "MAX_VAL_SAMPLES": "training.max_val_samples",
    "MAX_TEST_SAMPLES": "training.max_test_samples",
    "RESTORE_SAVED_QAT": "steps.restore_saved_qat",
    "RUN_TRT_FP16_BASELINE": "steps.run_trt_fp16_baseline",
    "RUN_FINAL_TEST": "steps.run_final_test",
    "UPLOAD_BEST_EACH_EPOCH": "steps.upload_best_each_epoch",
    "UPLOAD_FINAL_ARTIFACTS": "steps.upload_final_artifacts",
    "SEGFORMER_QAT_CONV_ONLY_REFERENCE": "reference.segformer_qat_conv_only",
    "MAX_VALIDATION_MIOU_DROP_VS_CONV_ONLY": "reference.max_validation_miou_drop_vs_conv_only",
    "QAT_DRIVE_PARENT_FOLDER_ID": "output.drive_parent_folder_id",
    "ONNX_OPSET": "export.onnx_opset",
    "FP16_FALLBACK_MIN_PREDICTION_AGREEMENT": "export.fp16_fallback_min_prediction_agreement",
    "EXPORT_MIN_PREDICTION_AGREEMENT": "export.export_min_prediction_agreement",
    "TRT_MIN_PREDICTION_AGREEMENT": "export.trt_min_prediction_agreement",
    "TRT_DEVICE_ID": "tensorrt.device_id",
    "TRT_WORKSPACE_GIB": "tensorrt.workspace_gib",
    "TRT_BUILD_MODE": "tensorrt.build_mode",
    "TRT_FP16_BUILD_MODE": "tensorrt.fp16_build_mode",
    "TRT_BUILDER_OPTIMIZATION_LEVEL": "tensorrt.builder_optimization_level",
    "LATENCY_WARMUP_RUNS": "benchmark.latency_warmup_runs",
    "LATENCY_MEASURED_RUNS": "benchmark.latency_measured_runs",
    "EFFICIENCY_WARMUP_RUNS": "benchmark.efficiency_warmup_runs",
    "EFFICIENCY_MIN_RUNS": "benchmark.efficiency_min_runs",
    "EFFICIENCY_MIN_SECONDS": "benchmark.efficiency_min_seconds",
    "NVML_SAMPLE_INTERVAL_SECONDS": "benchmark.nvml_sample_interval_seconds",
}


SPECS = (
    (
        "notebooks/kaggle_segmentation_training.ipynb",
        "configs/train/segmentation.yaml",
        TRAINING_FIELDS,
    ),
    (
        "notebooks/kaggle_s5mars_onnx_ptq.ipynb",
        "configs/quantization/ptq.yaml",
        PTQ_FIELDS,
    ),
    (
        "notebooks/kaggle_s5mars_onnx_qat.ipynb",
        "configs/quantization/qat.yaml",
        QAT_FIELDS,
    ),
)


def notebook_literals(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])), filename=str(path))
        except SyntaxError:
            # Installation cells contain IPython !/% syntax and no runtime
            # constants relevant to the standalone configuration.
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            try:
                literal = _safe_literal(node.value, values)
            except (ValueError, TypeError, KeyError):
                continue
            if isinstance(target, ast.Name):
                values[target.id] = literal
            elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(
                literal, (tuple, list)
            ):
                names = [item for item in target.elts if isinstance(item, ast.Name)]
                if len(names) == len(target.elts) == len(literal):
                    values.update(
                        {name.id: item for name, item in zip(names, literal)}
                    )
    return values


def _safe_literal(node, known):
    """Evaluate only literal containers and references to earlier literals."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return copy.deepcopy(known[node.id])
    if isinstance(node, ast.Dict):
        return {
            _safe_literal(key, known): _safe_literal(value, known)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.List):
        return [_safe_literal(item, known) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(item, known) for item in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_literal(item, known) for item in node.elts}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_literal(node.operand, known)
        return -value if isinstance(node.op, ast.USub) else +value
    raise ValueError(f"Non-literal expression: {ast.dump(node)}")


def normalized(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (tuple, list)):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    return value


def check_sync(root: Path) -> list[str]:
    errors = []
    for notebook_name, config_name, fields in SPECS:
        notebook_path, config_path = root / notebook_name, root / config_name
        literals = notebook_literals(notebook_path)
        config = OmegaConf.load(config_path)
        if config.sync.direction != "notebook_to_python":
            errors.append(f"{config_name}: sync.direction is not notebook_to_python")
        if str(config.sync.source_notebook) != notebook_name:
            errors.append(f"{config_name}: wrong sync.source_notebook")
        for constant, config_path_key in fields.items():
            if constant not in literals:
                errors.append(f"{notebook_name}: literal {constant} not found")
                continue
            configured = OmegaConf.select(config, config_path_key)
            if normalized(literals[constant]) != normalized(configured):
                errors.append(
                    f"{notebook_name}:{constant}={literals[constant]!r} != "
                    f"{config_name}:{config_path_key}={configured!r}"
                )
        if notebook_name == "notebooks/kaggle_segmentation_training.ipynb":
            notebook_datasets = literals.get("DATASET_CONFIGS", {})
            for dataset_name, notebook_dataset in notebook_datasets.items():
                configured_dataset = config.datasets.get(dataset_name)
                if configured_dataset is None:
                    errors.append(f"{config_name}: missing dataset {dataset_name}")
                    continue
                for key in (
                    "family",
                    "repo_id",
                    "display_name",
                    "class_names",
                    "expected_split_sizes",
                    "cross_dataset_name",
                ):
                    expected_value = notebook_dataset[key]
                    actual_value = configured_dataset.get(key)
                    if normalized(expected_value) != normalized(actual_value):
                        errors.append(
                            f"{config_name}:datasets.{dataset_name}.{key} differs "
                            "from DATASET_CONFIGS"
                        )
                if int(configured_dataset.num_classes) != len(
                    notebook_dataset["class_names"]
                ):
                    errors.append(
                        f"{config_name}:datasets.{dataset_name}.num_classes differs "
                        "from the notebook taxonomy"
                    )

            model_name = literals.get("MODEL_NAME")
            defaults = normalized(config.defaults)
            if {"/model": model_name} not in defaults:
                errors.append(
                    f"{config_name}: default model differs from MODEL_NAME={model_name!r}"
                )
            model_checks = (
                (
                    root / "configs/model/segformer_b0.yaml",
                    {
                        "name": literals.get("MODEL_NAME"),
                        "pretrained_name": literals.get("PRETRAINED_NAME"),
                    },
                ),
                (
                    root / "configs/model/smp.yaml",
                    {
                        "architecture": literals.get("SMP_ARCHITECTURE"),
                        "encoder_name": literals.get("SMP_ENCODER_NAME"),
                        "encoder_weights": literals.get("SMP_ENCODER_WEIGHTS"),
                    },
                ),
                (
                    root / "configs/model/lcnet3_7.yaml",
                    {
                        "variant": literals.get("LCNET_VARIANT"),
                        "base_channels": literals.get("LCNET_BASE_CHANNELS"),
                        "partial_rate": literals.get("LCNET_PARTIAL_RATE"),
                    },
                ),
            )
            for model_path, expected_fields in model_checks:
                model_config = OmegaConf.load(model_path)
                for key, expected_value in expected_fields.items():
                    if normalized(model_config.get(key)) != normalized(expected_value):
                        errors.append(
                            f"{model_path.relative_to(root)}:{key} differs from notebook"
                        )
    return errors


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    errors = check_sync(root)
    if errors:
        raise SystemExit("Notebook -> Python sync failed:\n- " + "\n- ".join(errors))
    print("Notebook -> Python configuration sync: OK")


if __name__ == "__main__":
    main()
