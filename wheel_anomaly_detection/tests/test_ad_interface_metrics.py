import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hydra import compose, initialize_config_dir

from src.anomaly_detection.evaluation import (
    AnomalyMetrics,
    ExactBinaryMetrics,
    aggregate_patch_scores,
    build_metrics,
    evaluate_gaussian_sigma_ablation,
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
                "target_wheel_pixel_auroc",
                "target_wheel_pixel_average_precision",
                "normalized_image_scores_at_zero",
                "normalized_image_scores_at_one",
                "normalized_image_score_saturation_fraction",
            },
        )
        for name in (
            "image_auroc",
            "image_average_precision",
            "pixel_auroc",
            "pixel_average_precision",
            "target_wheel_pixel_auroc",
            "target_wheel_pixel_average_precision",
        ):
            self.assertAlmostEqual(result[name], 1.0)
        self.assertEqual(result["normalized_image_scores_at_zero"], 0)
        self.assertEqual(result["normalized_image_scores_at_one"], 0)
        self.assertEqual(result["normalized_image_score_saturation_fraction"], 0.0)

    def test_raw_image_scores_override_saturated_normalized_scores(self) -> None:
        prediction = AnomalyPrediction(
            anomaly_score=torch.tensor([1.0, 1.0]),
            anomaly_map=torch.tensor([
                [[[0.1, 0.1], [0.1, 0.1]]],
                [[[0.1, 0.9], [0.1, 0.9]]],
            ]),
        )
        masks = torch.tensor([
            [[[0, 0], [0, 0]]],
            [[[0, 255], [0, 255]]],
        ])
        target_wheel = torch.tensor([
            [[[0, 1], [0, 1]]],
            [[[0, 1], [0, 1]]],
        ])
        metrics = AnomalyMetrics(histogram_bins=32)

        metrics.update(
            prediction,
            torch.tensor([0, 1]),
            masks,
            image_scores=torch.tensor([2.0, 5.0]),
            target_wheel_mask=target_wheel,
        )
        result = metrics.compute()

        self.assertEqual(result["image_auroc"], 1.0)
        self.assertEqual(result["image_average_precision"], 1.0)
        self.assertEqual(result["normalized_image_scores_at_one"], 2)
        self.assertEqual(result["normalized_image_score_saturation_fraction"], 1.0)
        self.assertEqual(result["target_wheel_pixel_auroc"], 1.0)

    def test_image_metrics_preserve_ranking_inside_one_histogram_bin(self) -> None:
        metrics = ExactBinaryMetrics()
        metrics.update(
            torch.tensor([0.5001, 0.5004]),
            torch.tensor([0, 1]),
        )

        result = metrics.compute()
        self.assertEqual(result, {"auroc": 1.0, "average_precision": 1.0})

    def test_patch_score_aggregations(self) -> None:
        patch_scores = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ])

        self.assertTrue(torch.equal(
            aggregate_patch_scores(patch_scores, {"mode": "max"}),
            torch.tensor([4.0, 8.0]),
        ))
        self.assertTrue(torch.equal(
            aggregate_patch_scores(
                patch_scores, {"mode": "topk_mean", "topk": 2}
            ),
            torch.tensor([3.5, 7.5]),
        ))
        self.assertTrue(torch.equal(
            aggregate_patch_scores(
                patch_scores, {"mode": "quantile", "quantile": 0.5}
            ),
            torch.tensor([2.5, 6.5]),
        ))
        with self.assertRaisesRegex(ValueError, "topk"):
            aggregate_patch_scores(
                patch_scores, {"mode": "topk_mean", "topk": 5}
            )

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
        config = self._config()
        metrics = build_metrics(config)
        self.assertIsInstance(metrics.image, ExactBinaryMetrics)
        self.assertEqual(metrics.pixel.num_bins, 2048)
        candidates = config.evaluation.image_score_aggregation.candidates
        self.assertEqual(candidates[0].name, "max")
        self.assertEqual(candidates[1].mode, "topk_mean")
        self.assertTrue(config.evaluation.gaussian_sigma_ablation.enabled)
        self.assertEqual(
            list(config.evaluation.gaussian_sigma_ablation.candidates),
            [0.0, 1.0, 2.0, 4.0],
        )

    def test_sigma_ablation_is_validation_only_and_restores_sigma(self) -> None:
        class Detector:
            gaussian_sigma = 4.0

        class Diagnostics:
            def compute(self):
                return {
                    "pro": {
                        "aupro": 0.75,
                        "aupro_by_max_fpr": {"0.05": 0.5, "0.30": 0.75},
                    }
                }

        detector = Detector()
        metrics = {
            "image_auroc": 0.8,
            "image_average_precision": 0.7,
            "pixel_auroc": 0.9,
            "pixel_average_precision": 0.1,
        }
        with patch(
            "src.anomaly_detection.evaluation.runner.evaluate_anomaly_detector",
            return_value=metrics,
        ):
            result = evaluate_gaussian_sigma_ablation(
                detector,
                [],
                [0, 1, 2, 4],
                device="cpu",
                diagnostics_factory=Diagnostics,
                reference_image_metrics=metrics,
            )

        self.assertEqual(detector.gaussian_sigma, 4.0)
        self.assertEqual(result["split"], "validation")
        self.assertEqual(result["selection"], "none")
        self.assertNotIn("test", result)
        self.assertEqual(result["validation"]["sigma_2"]["aupro"], 0.75)

    def test_sigma_ablation_selects_on_validation_and_tests_only_winner(self) -> None:
        class Detector:
            gaussian_sigma = 4.0

        class Diagnostics:
            def compute(self):
                return {
                    "pro": {
                        "aupro": 0.75,
                        "aupro_by_max_fpr": {"0.05": 0.5, "0.30": 0.75},
                    }
                }

        detector = Detector()
        wheel_ap = {0.0: 0.1, 1.0: 0.4, 2.0: 0.3, 4.0: 0.2}

        def metrics_for_sigma(*args, **kwargs):
            return {
                "image_auroc": 0.8,
                "image_average_precision": 0.7,
                "pixel_auroc": 0.9,
                "pixel_average_precision": 0.1,
                "target_wheel_pixel_average_precision": wheel_ap[
                    float(detector.gaussian_sigma)
                ],
            }

        reference = {"image_auroc": 0.8, "image_average_precision": 0.7}
        with patch(
            "src.anomaly_detection.evaluation.runner.evaluate_anomaly_detector",
            side_effect=metrics_for_sigma,
        ) as evaluate:
            result = evaluate_gaussian_sigma_ablation(
                detector, [], [0, 1, 2, 4], device="cpu",
                diagnostics_factory=Diagnostics,
                reference_image_metrics=reference,
                test_loader=[], test_diagnostics_factory=Diagnostics,
                reference_test_image_metrics=reference,
                selection_metric="target_wheel_pixel_average_precision",
            )

        self.assertEqual(detector.gaussian_sigma, 4.0)
        self.assertEqual(evaluate.call_count, 5)
        self.assertEqual(result["selected"], "sigma_1")
        self.assertEqual(result["test"]["sigma_1"]["sigma"], 1.0)
        self.assertFalse(result["test_used_for_selection"])

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
