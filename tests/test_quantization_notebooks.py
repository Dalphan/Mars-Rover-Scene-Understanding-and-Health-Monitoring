from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_NOTEBOOK = ROOT / "notebooks" / "kaggle_s5mars_segformer_training.ipynb"
PTQ_NOTEBOOK = ROOT / "notebooks" / "kaggle_s5mars_ptq.ipynb"


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_quantization_notebook_cells_parse():
    for path in (TRAINING_NOTEBOOK, PTQ_NOTEBOOK):
        notebook = _load_notebook(path)
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell["source"])
            if cell["cell_type"] != "code" or source.lstrip().startswith("!"):
                continue
            ast.parse(source, filename=f"{path.name}:cell{index}")


def test_training_notebook_exposes_guarded_qat_and_keeps_metric_outputs():
    code = _code(_load_notebook(TRAINING_NOTEBOOK))
    assert any(
        f'QUANTIZATION_MODE = "{mode}"' in code
        for mode in ("none", "qat_int8")
    )
    assert 'QUANTIZATION_MODE == "qat_int8"' in code
    assert "mtq.INT8_DEFAULT_CFG" in code
    assert "QAT_QUANTIZER_EXCLUSIONS_BY_MODEL" in code
    assert '("smp", "deeplabv3", "resnet34")' in code
    assert '"*encoder.layer3*"' in code
    assert '"*encoder.layer4*"' in code
    assert '("smp", "deeplabv3plus", "mobilenet_v2")' in code
    for feature_index in range(14, 19):
        assert f'"*encoder.features.{feature_index}*"' in code
    assert "qat_quantize_config" in code
    assert '"excluded_quantizers": QAT_EXCLUDED_QUANTIZERS' in code
    assert "mto.save(" in code
    assert "model_int8_qat_qdq.onnx" in code
    assert "engine_int8_qat.plan" in code
    assert '"history.json"' in code
    assert '"test_metrics.json"' in code
    assert "DRIVE_UPLOAD_ENABLED = False" in code
    assert "def segmentation_collate_fn(" in code
    assert "collate_fn=segmentation_collate_fn" in code
    assert "collate_samples" not in code


def test_ptq_notebook_has_separate_precision_paths_and_train_only_calibration():
    code = _code(_load_notebook(PTQ_NOTEBOOK))
    assert 'TRAIN_SPLIT = "train"' in code
    assert "LoaderCalibrationReader" in code
    assert "def get_first(self)" in code
    assert "model_fp32.onnx" in code
    assert "model_fp16.onnx" in code
    assert "model_int8_qdq.onnx" in code
    assert "ONNXRuntimeRunner" in code
    assert "APPLY_UNET_PTQ_FALLBACK = True" in code
    assert 'model_cfg.get("architecture", "").lower() == "unet"' in code
    assert 'if "segmentation_head" in node.name' in code
    assert "use_zero_point=ptq_use_zero_point" in code
    assert "nodes_to_exclude=ptq_nodes_to_exclude" in code
    assert "ORT_VALIDATION_PROVIDERS = [\"CPUExecutionProvider\"]" in code
    assert "ort.GraphOptimizationLevel.ORT_DISABLE_ALL" in code
    assert 'int8_ort_optimization_diagnostic.json' in code
    assert 'RUN_FP16_ONNX = True' in code
    assert 'val_metrics_onnx_{precision}.json' in code
    assert 'onnx_validation_comparison.json' in code
    assert 'efficiency_estimates.json' in code
    assert '"speedup_claimed": False' in code
    assert "engine_fp32.plan" in code
    assert "engine_fp16.plan" in code
    assert "engine_int8.plan" in code
    assert 'RUN_FP32_TRT = False' in code
    assert 'RUN_FP16_TRT = False' in code
    assert 'RUN_INT8_TRT = False' in code
    assert 'if RUN_INT8_TRT:' in code
    assert "RUN_FINAL_TEST = False" in code
    assert "source_checkpoint_sha256" in code
    assert "QUANTIZATION_MODE" not in code
