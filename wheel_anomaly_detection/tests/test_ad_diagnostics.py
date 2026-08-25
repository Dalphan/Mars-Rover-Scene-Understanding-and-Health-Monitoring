import tempfile
import unittest
from pathlib import Path

import torch

from src.anomaly_detection.data import PreprocessingConfig, WheelPreprocessor
from src.anomaly_detection.evaluation import (
    AnomalyDiagnostics,
    AnomalyMetrics,
    PerRegionOverlap,
    save_anomaly_diagnostics,
)
from src.anomaly_detection.models import AnomalyPrediction


class AnomalyDiagnosticsTests(unittest.TestCase):
    def test_existing_metrics_class_name_is_preserved(self) -> None:
        self.assertEqual(AnomalyMetrics.__name__, "AnomalyMetrics")

    def test_aupro_is_one_for_perfect_region_separation(self) -> None:
        scores = torch.zeros(1, 1, 8, 8)
        anomaly_mask = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
        anomaly_mask[:, :, 2:6, 2:6] = 255
        scores[anomaly_mask > 0] = 1.0
        metric = PerRegionOverlap(num_bins=32, max_fpr=0.30)

        metric.update(scores, anomaly_mask)
        result = metric.compute()

        self.assertAlmostEqual(result["aupro"], 1.0)
        self.assertEqual(
            set(result["aupro_by_max_fpr"]), {"0.05", "0.10", "0.30"}
        )
        self.assertTrue(
            all(value == 1.0 for value in result["aupro_by_max_fpr"].values())
        )
        self.assertEqual(result["num_regions"], 1)
        self.assertGreater(len(result["curve"]), 2)

    def test_grouped_pixel_ap_prevalence_and_artifacts(self) -> None:
        rows = [
            {
                "image_id": "clean_0",
                "pair_id": "pair_0",
                "condition": "clean",
                "severity": "",
                "camera_pose": "pose_a",
                "lighting": "light_a",
                "wear": "wear_a",
            },
            {
                "image_id": "hole_0",
                "pair_id": "pair_0",
                "condition": "hole",
                "severity": "medium",
                "camera_pose": "pose_a",
                "lighting": "light_a",
                "wear": "wear_a",
            },
        ]
        images = torch.rand(2, 3, 8, 8)
        anomaly_masks = torch.zeros(2, 1, 8, 8, dtype=torch.uint8)
        anomaly_masks[1, :, 2:6, 2:6] = 255
        anomaly_maps = torch.zeros(2, 1, 8, 8)
        anomaly_maps[1, :, 2:6, 2:6] = 1.0
        batch = {
            "image": images,
            "target_mask": torch.full((2, 1, 8, 8), 255, dtype=torch.uint8),
            "anomaly_mask": anomaly_masks,
            "label": torch.tensor([0, 1]),
            "metadata": {
                key: [row[key] for row in rows]
                for key in (
                    "image_id",
                    "pair_id",
                    "condition",
                    "severity",
                    "camera_pose",
                    "lighting",
                    "wear",
                )
            },
        }
        prediction = AnomalyPrediction(
            anomaly_score=torch.tensor([0.1, 0.9]),
            anomaly_map=anomaly_maps,
        )
        diagnostics = AnomalyDiagnostics(
            rows,
            preprocessor=WheelPreprocessor(
                PreprocessingConfig(augmentations_enabled=False)
            ),
            histogram_bins=32,
            pro_bins=32,
            num_extreme_examples=1,
        )

        diagnostics.update(prediction, torch.tensor([0.2, 1.2]), batch)
        result = diagnostics.compute()

        severity = result["pixel_average_precision_by_group"]["severity"][
            "medium"
        ]
        self.assertAlmostEqual(severity["pixel_average_precision"], 1.0)
        self.assertEqual(severity["num_images"], 2)
        image_severity = result["image_metrics_by_group"]["severity"]["medium"]
        self.assertAlmostEqual(image_severity["image_auroc"], 1.0)
        self.assertAlmostEqual(image_severity["image_average_precision"], 1.0)
        self.assertEqual(image_severity["num_clean_images"], 1)
        self.assertEqual(image_severity["num_anomalous_images"], 1)
        self.assertAlmostEqual(result["anomalous_pixel_fraction"], 16 / 128)
        self.assertAlmostEqual(result["random_pixel_ap_baseline"], 16 / 128)
        self.assertEqual(diagnostics.top_clean[0]["image_id"], "clean_0")
        self.assertEqual(diagnostics.bottom_hole[0]["image_id"], "hole_0")

        with tempfile.TemporaryDirectory() as directory:
            summary, paths = save_anomaly_diagnostics(
                diagnostics,
                directory,
                split="validation",
                dpi=50,
            )
            self.assertIn("pro", summary)
            self.assertTrue(all(Path(path).is_file() for path in paths))
            self.assertTrue(
                (Path(directory) / "validation_image_scores.csv").is_file()
            )


    def test_grouped_image_metrics_are_null_for_one_class(self) -> None:
        diagnostics = object.__new__(AnomalyDiagnostics)
        diagnostics.records = [
            {
                "condition": "clean",
                "normalized_image_score": 0.2,
                "raw_image_score": 1.0,
                "camera_pose": "clean_only",
            }
        ]
        diagnostics.valid_pixels = 1
        diagnostics.anomalous_pixels = 0
        diagnostics.group_fields = ("camera_pose",)
        diagnostics.group_metrics = {"camera_pose": {}}
        diagnostics.group_image_counts = {"camera_pose": {}}
        diagnostics.pro = None

        # Exercise only the grouping branch without requiring a valid PRO input.
        original_pro = diagnostics.pro
        class StubPro:
            @staticmethod
            def compute():
                return {"aupro": 0.0}

        diagnostics.pro = StubPro()
        result = diagnostics.compute()
        group = result["image_metrics_by_group"]["camera_pose"]["clean_only"]
        self.assertIsNone(group["image_auroc"])
        self.assertIsNone(group["image_average_precision"])
        self.assertEqual(group["num_clean_images"], 1)
        self.assertEqual(group["num_anomalous_images"], 0)
        diagnostics.pro = original_pro



if __name__ == "__main__":
    unittest.main()
