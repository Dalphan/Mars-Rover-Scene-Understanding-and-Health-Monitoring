import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.anomaly_detection.evaluation.real_pilot import (
    evaluate_real_wheel_pilot,
    plot_real_wheel_predictions,
    save_real_wheel_pilot_results,
)
from src.anomaly_detection.models import AnomalyDetector, AnomalyPrediction


class _PerfectDetector(AnomalyDetector):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    @property
    def is_fitted(self) -> bool:
        return True

    def predict_with_raw(self, images):
        anomaly_map = images[:, :1].clamp(0, 1)
        raw_score = anomaly_map.flatten(1).amax(1)
        prediction = AnomalyPrediction(raw_score, anomaly_map)
        return prediction, raw_score, anomaly_map

    def predict(self, images):
        return self.predict_with_raw(images)[0]


def _batch(image_id: str, label: int, mask: torch.Tensor) -> dict:
    image = torch.zeros(1, 3, 8, 8)
    image[:, :1] = mask.float() / 255
    return {
        "image": image,
        "anomaly_mask": mask,
        "label": torch.tensor([label]),
        "metadata": {
            "image_id": [image_id],
            "sol": [100 + label],
            "condition": ["hole" if label else "no_visible_hole"],
        },
    }


class RealWheelPilotEvaluationTests(unittest.TestCase):
    def test_metrics_visualizations_and_artifacts_share_one_prediction_pass(self):
        clean_mask = torch.zeros(1, 1, 8, 8, dtype=torch.uint8)
        hole_mask = clean_mask.clone()
        hole_mask[:, :, 2:6, 2:6] = 255
        loader = [
            _batch("clean", 0, clean_mask),
            _batch("hole", 1, hole_mask),
        ]
        detector = _PerfectDetector()

        result = evaluate_real_wheel_pilot(
            detector,
            loader,
            device="cpu",
            image_for_display=lambda image: image,
            histogram_bins=32,
            pro_bins=32,
        )

        for name in (
            "image_auroc",
            "image_average_precision",
            "pixel_auroc",
            "pixel_average_precision",
            "aupro",
        ):
            self.assertAlmostEqual(result["metrics"][name], 1.0)
        self.assertEqual(result["metrics"]["num_images"], 2)
        self.assertFalse(result["metrics"]["target_wheel_metrics_available"])
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(len(result["samples"]), 2)

        figure = plot_real_wheel_predictions(
            result["samples"], title="Real pilot"
        )
        self.assertGreaterEqual(len(figure.axes), 6)

        with tempfile.TemporaryDirectory() as directory:
            figures, paths = save_real_wheel_pilot_results(
                result,
                directory,
                context={"model_name": "perfect"},
                rows_per_figure=1,
                dpi=50,
            )
            self.assertEqual(len(figures), 2)
            self.assertTrue(Path(paths["metrics"]).is_file())
            self.assertTrue(Path(paths["image_scores"]).is_file())
            self.assertTrue(Path(paths["pro_curve"]).is_file())
            self.assertTrue(all(Path(path).is_file() for path in paths["figures"]))

    def test_batch_size_greater_than_one_is_rejected(self):
        mask = torch.zeros(2, 1, 8, 8, dtype=torch.uint8)
        batch = {
            "image": torch.zeros(2, 3, 8, 8),
            "anomaly_mask": mask,
            "label": torch.tensor([0, 1]),
            "metadata": {},
        }
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            evaluate_real_wheel_pilot(
                _PerfectDetector(),
                [batch],
                device="cpu",
                image_for_display=lambda image: image,
            )


if __name__ == "__main__":
    unittest.main()
