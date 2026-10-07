import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import pytest

import scripts.check_notebook_config_sync as sync_module
from scripts.check_notebook_config_sync import check_sync


def test_notebook_constants_match_python_configs():
    root = Path(__file__).resolve().parents[1]
    assert check_sync(root) == []


@pytest.mark.parametrize("preset", ["segformer_b0", "smp", "lcnet3_7", "lcnet3_11"])
def test_training_sync_resolves_active_model_preset(tmp_path, monkeypatch, preset):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(sync_module, "SPECS", sync_module.SPECS[:1])
    shutil.copytree(root / "configs", tmp_path / "configs")
    notebook_path = tmp_path / "notebooks/kaggle_segmentation_training.ipynb"
    notebook_path.parent.mkdir()
    notebook = json.loads(
        (root / "notebooks/kaggle_segmentation_training.ipynb").read_text(
            encoding="utf-8"
        )
    )
    model_name = "lcnet" if preset.startswith("lcnet") else preset
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        for index, line in enumerate(cell.get("source", [])):
            if line.startswith("MODEL_NAME = "):
                cell["source"][index] = f'MODEL_NAME = "{model_name}"\n'
            elif preset.startswith("lcnet") and line.startswith("LCNET_VARIANT = "):
                cell["source"][index] = f'LCNET_VARIANT = "{preset}"\n'
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    config_path = tmp_path / "configs/train/segmentation.yaml"
    config = OmegaConf.load(config_path)
    config.defaults[0]["/model"] = preset
    OmegaConf.save(config, config_path)

    assert check_sync(tmp_path) == []

    config.defaults[0]["/model"] = "smp" if preset != "smp" else "segformer_b0"
    OmegaConf.save(config, config_path)
    assert any("default model differs" in error for error in check_sync(tmp_path))
