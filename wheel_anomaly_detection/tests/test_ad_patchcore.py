import json
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from src.anomaly_detection.models import PatchCore, build_model
from src.anomaly_detection.training import build_optimizer, build_scheduler


class PatchCoreTests(unittest.TestCase):
    def _config(self):
        config_dir = Path(__file__).parents[1] / "configs" / "anomaly_detection"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            config = compose(config_name="config")
        config.model.pretrained = False
        return config

    def test_factory_builds_frozen_patch_embeddings(self) -> None:
        model = build_model(self._config())
        embeddings = model(torch.randn(2, 3, 64, 64))

        self.assertIsInstance(model, PatchCore)
        self.assertEqual(embeddings.shape, (2, 384, 8, 8))
        self.assertFalse(model.training)
        self.assertFalse(model.is_fitted)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_patchcore_has_no_optimizer_or_scheduler(self) -> None:
        config = self._config()
        model = build_model(config)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        self.assertIsNone(optimizer)
        self.assertIsNone(scheduler)

    def test_rejects_invalid_patchcore_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            PatchCore(pretrained=False, pool_kernel_size=2)

    def test_notebook_contains_self_contained_patchcore(self) -> None:
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "anomaly_detection"
            / "kaggle_wheel_anomaly_detection.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertNotIn("src.anomaly_detection", source)
        self.assertIn("MODEL_NAME = \"patchcore\"", source)
        self.assertIn("class AnomalyDetector(nn.Module)", source)
        self.assertIn("class PatchCore(AnomalyDetector)", source)
        self.assertIn("OPTIMIZER_NAME = \"none\"", source)
        self.assertIn("SCHEDULER_NAME = \"none\"", source)
        self.assertIn("model = build_model()", source)


if __name__ == "__main__":
    unittest.main()
