import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir

from src.anomaly_detection.models import PatchCore, build_model
from src.anomaly_detection.data import PreprocessingConfig, WheelPreprocessor
from src.anomaly_detection.evaluation import (
    collect_anomaly_visualization_samples,
    evaluate_anomaly_detector,
    plot_anomaly_visualizations,
    save_anomaly_visualizations,
)
from src.anomaly_detection.training import build_optimizer, build_scheduler
from src.anomaly_detection.training import run_anomaly_experiment


class PatchCoreTests(unittest.TestCase):
    def _config(self, overrides=None):
        config_dir = Path(__file__).parents[1] / "configs" / "anomaly_detection"
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            config = compose(config_name="config", overrides=overrides or [])
        config.model.pretrained = False
        return config

    @staticmethod
    def _tiny_model() -> PatchCore:
        return PatchCore(
            pretrained=False,
            coreset_sampling_ratio=0.5,
            num_neighbors=2,
            max_patches_per_image=4,
            max_training_embeddings=8,
            max_memory_bank_size=2,
            projection_dim=4,
            calibration_batches=1,
            distance_query_chunk_size=16,
            distance_bank_chunk_size=2,
            gaussian_sigma=1.0,
        )

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
            PatchCore(pretrained=False, patch_size=2)

    def test_complete_fit_and_predict_contract(self) -> None:
        loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.zeros(2, dtype=torch.long),
            }
        ]
        model = self._tiny_model().fit(loader, device="cpu")
        prediction = model.predict(torch.randn(2, 3, 64, 64))

        self.assertTrue(model.is_fitted)
        self.assertEqual(model.memory_bank.shape, (2, 384))
        self.assertEqual(prediction.anomaly_score.shape, (2,))
        self.assertEqual(prediction.anomaly_map.shape, (2, 1, 64, 64))
        self.assertTrue((prediction.anomaly_score >= 0).all())
        self.assertTrue((prediction.anomaly_score <= 1).all())
        self.assertTrue((prediction.anomaly_map >= 0).all())
        self.assertTrue((prediction.anomaly_map <= 1).all())

    def test_fitted_checkpoint_preserves_predictions(self) -> None:
        loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.zeros(2, dtype=torch.long),
            }
        ]
        images = torch.randn(1, 3, 64, 64)
        model = self._tiny_model().fit(loader, device="cpu")
        expected = model.predict(images)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.ckpt"
            model.save(checkpoint)
            restored = self._tiny_model()
            restored.load(checkpoint, map_location="cpu")
            actual = restored.predict(images)

        self.assertTrue(torch.equal(actual.anomaly_score, expected.anomaly_score))
        self.assertTrue(torch.equal(actual.anomaly_map, expected.anomaly_map))

    def test_evaluation_runner_returns_only_essential_metrics(self) -> None:
        train_loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.zeros(2, dtype=torch.long),
            }
        ]
        anomaly_masks = torch.zeros(2, 1, 64, 64, dtype=torch.uint8)
        anomaly_masks[1, :, 24:40, 24:40] = 255
        evaluation_loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.tensor([0, 1]),
                "target_mask": torch.full(
                    (2, 1, 64, 64), 255, dtype=torch.uint8
                ),
                "anomaly_mask": anomaly_masks,
            }
        ]
        model = self._tiny_model().fit(train_loader, device="cpu")
        metrics = evaluate_anomaly_detector(
            model,
            evaluation_loader,
            device="cpu",
            histogram_bins=32,
        )

        self.assertEqual(
            set(metrics),
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

    def test_fit_rejects_anomalous_training_images(self) -> None:
        loader = [
            {
                "image": torch.randn(1, 3, 64, 64),
                "label": torch.ones(1, dtype=torch.long),
            }
        ]
        with self.assertRaisesRegex(ValueError, "only clean"):
            self._tiny_model().fit(loader, device="cpu")

    def test_predict_requires_fitted_memory_bank(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            self._tiny_model().predict(torch.randn(1, 3, 64, 64))

    def test_light_reference_and_256_presets(self) -> None:
        default_config = self._config()
        light_config = self._config(["model=patchcore_light"])
        reference_config = self._config(["model=patchcore_reference"])
        lower_resolution_config = self._config(["model=patchcore_256"])

        self.assertEqual(tuple(default_config.model.input_size), (384, 512))
        self.assertEqual(tuple(light_config.model.input_size), (384, 512))
        self.assertEqual(light_config.model.target_embed_dimension, 384)
        self.assertEqual(reference_config.model.backbone, "wide_resnet50_2")
        self.assertEqual(tuple(reference_config.model.input_size), (224, 224))
        self.assertEqual(reference_config.model.resize_shorter_side, 256)
        self.assertEqual(tuple(reference_config.model.center_crop), (224, 224))
        self.assertEqual(reference_config.model.target_embed_dimension, 1024)
        self.assertEqual(reference_config.model.num_neighbors, 1)
        self.assertEqual(tuple(lower_resolution_config.model.input_size), (256, 256))
        self.assertEqual(default_config.model.max_memory_bank_size, 2048)
        self.assertEqual(lower_resolution_config.model.max_memory_bank_size, 2048)

    def test_reference_embedding_has_expected_patch_grid_and_dimension(self) -> None:
        config = self._config(["model=patchcore_reference"])
        model = build_model(config)
        embeddings = model(torch.randn(1, 3, 224, 224))

        self.assertEqual(embeddings.shape, (1, 1024, 28, 28))

    def test_terminal_experiment_saves_checkpoint_and_metrics(self) -> None:
        config = self._config()
        config.evaluation.visualization.enabled = False
        config.evaluation.diagnostics.enabled = False
        config.model.coreset_sampling_ratio = 0.5
        config.model.num_neighbors = 2
        config.model.max_patches_per_image = 4
        config.model.max_training_embeddings = 8
        config.model.max_memory_bank_size = 2
        config.model.projection_dim = 4
        config.model.calibration_batches = 1
        config.model.distance_query_chunk_size = 16
        config.model.distance_bank_chunk_size = 2
        config.model.gaussian_sigma = 1.0
        train_loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.zeros(2, dtype=torch.long),
            }
        ]
        anomaly_masks = torch.zeros(2, 1, 64, 64, dtype=torch.uint8)
        anomaly_masks[1, :, 24:40, 24:40] = 255
        evaluation_loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.tensor([0, 1]),
                "target_mask": torch.full(
                    (2, 1, 64, 64), 255, dtype=torch.uint8
                ),
                "anomaly_mask": anomaly_masks,
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            config.output.root = directory
            with patch(
                "src.anomaly_detection.training.experiment.build_dataloaders",
                return_value=(train_loader, evaluation_loader, evaluation_loader),
            ):
                result = run_anomaly_experiment(config)

            self.assertTrue(result["checkpoint_path"].is_file())
            self.assertTrue(result["metrics_path"].is_file())
            metrics = json.loads(result["metrics_path"].read_text(encoding="utf-8"))
            self.assertEqual(set(metrics), {"validation", "test"})

    def test_qualitative_visualization_balances_clean_and_anomalous_samples(self) -> None:
        train_loader = [
            {
                "image": torch.rand(2, 3, 64, 64),
                "label": torch.zeros(2, dtype=torch.long),
            }
        ]
        evaluation_loader = [
            {
                "image": torch.rand(2, 3, 64, 64),
                "label": torch.tensor([0, 1]),
                "target_mask": torch.full(
                    (2, 1, 64, 64), 255, dtype=torch.uint8
                ),
                "anomaly_mask": torch.stack(
                    (
                        torch.zeros(1, 64, 64, dtype=torch.uint8),
                        torch.nn.functional.pad(
                            torch.full((1, 16, 16), 255, dtype=torch.uint8),
                            (24, 24, 24, 24),
                        ),
                    )
                ),
                "metadata": {"image_id": ["clean_0", "hole_0"]},
            }
        ]
        model = self._tiny_model().fit(train_loader, device="cpu")
        preprocessor = WheelPreprocessor(
            PreprocessingConfig(augmentations_enabled=False)
        )

        samples = collect_anomaly_visualization_samples(
            model,
            evaluation_loader,
            device="cpu",
            preprocessor=preprocessor,
            num_clean=1,
            num_anomalous=1,
        )
        figure = plot_anomaly_visualizations(samples, title="Test examples")

        self.assertEqual([sample["label"] for sample in samples], [0, 1])
        self.assertEqual(len(figure.axes), 9)
        with tempfile.TemporaryDirectory() as directory:
            output_path = save_anomaly_visualizations(
                model,
                evaluation_loader,
                Path(directory) / "examples.png",
                device="cpu",
                preprocessor=preprocessor,
                title="Test examples",
                num_clean=1,
                num_anomalous=1,
                dpi=50,
            )
            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)

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
        self.assertIn("def fit(self, train_loader", source)
        self.assertIn("def predict(self, images)", source)
        self.assertIn("def predict_with_raw", source)
        self.assertIn("def _build_coreset", source)
        self.assertIn("def _nearest_memory", source)
        self.assertIn("model.fit(train_loader, device=DEVICE)", source)
        self.assertIn("model = build_model()", source)

    def test_notebook_definitions_precede_ordered_runtime_cells(self) -> None:
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "anomaly_detection"
            / "kaggle_wheel_anomaly_detection.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell_ids = [cell.get("id") for cell in notebook["cells"]]
        definition_ids = (
            "data-preparation-definitions",
            "preprocessing-audit-definition",
            "dataloader-smoke-definition",
            "anomaly-visualization-definitions",
            "anomaly-diagnostics-definitions",
            "artifact-drive-integration",
        )
        runtime_ids = (
            "run-initialization",
            "data-preparation-run",
            "dataloader-smoke-run",
            "preprocessing-audit-run",
            "model-build-run",
            "patchcore-fit-run",
            "validation-run",
            "test-run",
            "diagnostics-run",
            "visualization-run",
            "artifact-persistence-run",
        )

        self.assertLess(
            max(cell_ids.index(cell_id) for cell_id in definition_ids),
            min(cell_ids.index(cell_id) for cell_id in runtime_ids),
        )
        self.assertEqual(
            [cell_id for cell_id in cell_ids if cell_id in runtime_ids],
            list(runtime_ids),
        )
        definition_source = "\n".join(
            "".join(notebook["cells"][cell_ids.index(cell_id)].get("source", []))
            for cell_id in definition_ids
        )
        self.assertNotIn("DATASET_ROOT = find_dataset_root()", definition_source)
        self.assertNotIn("model = build_model()", definition_source)
        self.assertNotIn("model.fit(", definition_source)

    def test_notebook_patchcore_runtime_smoke(self) -> None:
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "anomaly_detection"
            / "kaggle_wheel_anomaly_detection.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell.get("id"): "".join(cell.get("source", [])) for cell in notebook["cells"]}
        namespace: dict[str, object] = {}
        exec(cells["f3f66a9a"], namespace)
        exec(cells["common-model-interface"], namespace)
        patchcore_class = cells["patchcore-model"].split("\ndef build_model", 1)[0]
        exec(patchcore_class, namespace)

        notebook_model = namespace["PatchCore"](
            pretrained=False,
            coreset_sampling_ratio=0.5,
            num_neighbors=2,
            max_patches_per_image=4,
            max_training_embeddings=8,
            max_memory_bank_size=2,
            projection_dim=4,
            calibration_batches=1,
            distance_query_chunk_size=16,
            distance_bank_chunk_size=2,
            gaussian_sigma=1.0,
        )
        loader = [
            {
                "image": torch.randn(2, 3, 64, 64),
                "label": torch.zeros(2, dtype=torch.long),
            }
        ]
        notebook_model.fit(loader, device="cpu")
        prediction = notebook_model.predict(torch.randn(1, 3, 64, 64))

        self.assertEqual(prediction.anomaly_score.shape, (1,))
        self.assertEqual(prediction.anomaly_map.shape, (1, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
