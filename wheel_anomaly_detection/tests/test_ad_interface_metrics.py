import json
import tempfile
import unittest
from pathlib import Path

import torch

from hydra import compose, initialize_config_dir

from src.anomaly_detection.evaluation import (
    AnomalyMetrics,
    ExactBinaryMetrics,
    build_metrics,
)
from src.anomaly_detection.models import AnomalyPrediction, PatchCore
from src.anomaly_detection.utils import save_json_atomic


class AnomalyInterfaceMetricsTests(unittest.TestCase):
    def _config(self):
        config_dir = (
            Path(__file__).parents[1] / "configs" / "anomaly_detection"
        ).resolve()
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return compose(config_name="config")

    def test_prediction_enforces_common_shapes_and_range(self) -> None:
        prediction = AnomalyPrediction(
            anomaly_score=torch.tensor([0.1, 0.9]),
            anomaly_map=torch.rand(2, 1, 4, 5),
        )
        self.assertEqual(prediction.anomaly_score.shape, (2,))

        with self.assertRaisesRegex(ValueError, "normalized"):
            AnomalyPrediction(
                anomaly_score=torch.tensor([1.1]),
                anomaly_map=torch.zeros(1, 1, 2, 2),
            )

    def test_essential_metrics_are_perfect_for_perfect_ranking(self) -> None:
        prediction = AnomalyPrediction(
            anomaly_score=torch.tensor([0.1, 0.9]),
            anomaly_map=torch.tensor(
                [
                    [[[0.1, 0.2], [0.1, 0.2]]],
                    [[[0.1, 0.9], [0.2, 0.8]]],
                ]
            ),
        )
        anomaly_masks = torch.tensor(
            [
                [[[0, 0], [0, 0]]],
                [[[0, 255], [0, 255]]],
            ],
            dtype=torch.uint8,
        )
        metrics = AnomalyMetrics(histogram_bins=32)
        metrics.update(prediction, torch.tensor([0, 1]), anomaly_masks)

        result = metrics.compute()
        self.assertEqual(
            set(result),
            {
                "image_auroc",
                "image_average_precision",
                "pixel_auroc",
                "pixel_average_precision",
            },
        )
        for value in result.values():
            self.assertAlmostEqual(value, 1.0)

    def test_image_metrics_preserve_ranking_inside_one_histogram_bin(self) -> None:
        metrics = ExactBinaryMetrics()
        metrics.update(
            torch.tensor([0.5001, 0.5004]),
            torch.tensor([0, 1]),
        )

        result = metrics.compute()
        self.assertEqual(result, {"auroc": 1.0, "average_precision": 1.0})

    def test_checkpoint_restores_dynamic_patchcore_memory_bank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            model = PatchCore(pretrained=False)
            model.memory_bank = torch.rand(5, 8)
            model.save(path, metadata={"run_name": "test"})

            restored = PatchCore(pretrained=False)
            metadata = restored.load(path)

            self.assertTrue(torch.equal(restored.memory_bank, model.memory_bank))
            self.assertEqual(metadata, {"run_name": "test"})

            incompatible = PatchCore(pretrained=False, patch_size=5)
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                incompatible.load(path)

    def test_results_are_saved_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_json_atomic({"image_auroc": 0.75}, Path(directory) / "metrics.json")
            self.assertIn('"image_auroc": 0.75', path.read_text(encoding="utf-8"))

    def test_hydra_builds_only_the_essential_metrics(self) -> None:
        metrics = build_metrics(self._config())
        self.assertIsInstance(metrics.image, ExactBinaryMetrics)
        self.assertEqual(metrics.pixel.num_bins, 2048)

    def test_notebook_duplicates_metrics_and_drive_integration(self) -> None:
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

        self.assertIn("class AnomalyPrediction", source)
        self.assertIn("class ExactBinaryMetrics", source)
        self.assertIn("class AnomalyMetrics", source)
        self.assertIn("def plot_anomaly_visualizations", source)
        self.assertIn("def save_anomaly_visualizations", source)
        self.assertIn("class AnomalyDiagnostics", source)
        self.assertIn("class PerRegionOverlap", source)
        self.assertIn("random_pixel_ap_baseline", source)
        self.assertIn("self.image = ExactBinaryMetrics()", source)
        self.assertIn("def persist_run_artifacts", source)
        self.assertIn('get_kaggle_secret("GDRIVE_FOLDER_ID", required=True)', source)
        self.assertIn('detector.save(OUTPUT_DIR / "model.ckpt"', source)
        self.assertIn('OUTPUT_DIR / "metrics.json"', source)
        self.assertIn('OUTPUT_DIR / f"{split}_examples.png"', source)
        self.assertIn('"model_config": self.checkpoint_config()', source)
        self.assertIn('"manifest_sha256": sha256_file(DATASET_ROOT / "samples.csv")', source)


if __name__ == "__main__":
    unittest.main()
