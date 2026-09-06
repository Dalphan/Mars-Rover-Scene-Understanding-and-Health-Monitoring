import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from src.anomaly_detection.models import EfficientAD, SuperSimpleNet, TinyGLASS
from src.anomaly_detection.evaluation import (
    aggregate_patch_scores,
    evaluate_image_score_aggregations,
)
from src.anomaly_detection.models.factory import build_model
from src.anomaly_detection.models.supersimplenet import (
    SuperSimpleAnomalyGenerator,
)
from src.anomaly_detection.models.tinyglass import (
    TinyGLASSFeatureExtractor,
    TinyGLASSLAS,
    _build_validation_feature_cache,
    _cached_validation_diagnostics,
    _validation_diagnostics,
)


def clean_loader(batch_size=2):
    return [{
        "image": torch.rand(batch_size, 3, 64, 64),
        "label": torch.zeros(batch_size, dtype=torch.long),
    }]


def validation_loader():
    return [{
        "image": torch.rand(2, 3, 64, 64),
        "label": torch.tensor([0, 1]),
    }]


class InterruptingEpochLoader:
    def __init__(self, *, fail_after: int | None = None):
        self.fail_after = fail_after
        self.iterations = 0
        self.generator = torch.Generator().manual_seed(42)

    def __iter__(self):
        if self.fail_after is not None and self.iterations >= self.fail_after:
            raise RuntimeError("simulated pause")
        self.iterations += 1
        return iter(clean_loader())


class ModelRoadmapTests(unittest.TestCase):
    def test_efficientad_paper_training_predict_and_checkpoint(self):
        kwargs = {
            "require_teacher_weights": False, "channels": 8, "max_steps": 1,
            "checkpoint_interval": 1,
        }
        train = [{
            "image": torch.rand(1, 3, 256, 256),
            "label": torch.zeros(1, dtype=torch.long),
        }]
        validation = [{
            "image": torch.rand(1, 3, 256, 256),
            "label": torch.zeros(1, dtype=torch.long),
        }]
        penalty = [{"image": torch.rand(1, 3, 256, 256)}]
        model = EfficientAD(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            model.fit(
                train, device="cpu", validation_loader=validation,
                penalty_loader=penalty, work_dir=directory,
            )
            prediction = model.predict(torch.rand(1, 3, 256, 256))
            self.assertEqual(prediction.anomaly_score.shape, (1,))
            self.assertEqual(prediction.anomaly_map.shape, (1, 1, 256, 256))
            self.assertTrue((Path(directory) / "efficientad_training.ckpt").is_file())
            self.assertEqual(model.fit_summary["selected_step"], 1)
            self.assertFalse(model.fit_summary["stopped_early"])
            checkpoint = Path(directory) / "efficientad.ckpt"
            model.save(checkpoint)
            restored = EfficientAD(**kwargs)
            restored.load(checkpoint, map_location="cpu")
            self.assertTrue(restored.is_fitted)

    def test_efficientad_uses_monotonic_map_normalization_without_clipping(self):
        source_path = (
            Path(__file__).parents[1]
            / "src"
            / "anomaly_detection"
            / "models"
            / "trainable.py"
        )
        source = source_path.read_text(encoding="utf-8")
        prediction = source[source.index("def predict_with_raw"):]

        self.assertIn("torch.atan(10.0 * raw_map)", prediction)
        self.assertNotIn("anomaly_map.clamp(0, 1)", prediction)
        self.assertIn("raw_score = self._aggregate_image_score(", prediction)
        self.assertIn("selected_topk_fraction", prediction)

    def test_efficientad_384_geometry_matches_all_branches(self):
        model = EfficientAD(
            input_size=(384, 384), require_teacher_weights=False,
            channels=8, max_steps=1,
        )
        images = torch.rand(1, 3, 384, 384)
        with torch.no_grad():
            teacher = model.teacher(images)
            student = model.student(images)
            autoencoder = model.autoencoder(images)
        self.assertEqual(teacher.shape, (1, 8, 96, 96))
        self.assertEqual(student.shape, (1, 16, 96, 96))
        self.assertEqual(autoencoder.shape, teacher.shape)
        self.assertEqual(model.static_roi.shape, (1, 1, 96, 96))
        self.assertEqual(
            model.autoencoder.decoder_sizes, (5, 13, 23, 49, 95, 191)
        )
        with self.assertRaises(ValueError):
            EfficientAD(
                input_size=(384, 512), require_teacher_weights=False,
                channels=8, max_steps=1,
            )

    def test_efficientad_calibrates_native_maps_before_upsampling(self):
        source_path = (
            Path(__file__).parents[1]
            / "src"
            / "anomaly_detection"
            / "models"
            / "trainable.py"
        )
        source = source_path.read_text(encoding="utf-8")
        calibration = source[
            source.index("[EfficientAD] Optimization complete")
            : source.index("self.fitted.fill_(True)", source.index(
                "[EfficientAD] Optimization complete"
            ))
        ]

        self.assertIn("map_st, map_ae = self._raw_maps(images)", calibration)
        self.assertIn("st_maps.append(map_st.cpu())", calibration)
        self.assertNotIn("map_st = F.interpolate", calibration)
        self.assertIn("self._build_spatial_calibration(", calibration)

    def test_efficientad_resume_loads_rng_state_on_cpu(self):
        source_path = (
            Path(__file__).parents[1]
            / "src"
            / "anomaly_detection"
            / "models"
            / "trainable.py"
        )
        source = source_path.read_text(encoding="utf-8")

        self.assertIn('checkpoint_path, map_location="cpu"', source)
        self.assertIn(
            'resume_train_epoch_state, dtype=torch.uint8, device="cpu"',
            source,
        )
        self.assertIn(
            'checkpoint["torch_rng_state"], dtype=torch.uint8, device="cpu"',
            source,
        )
        self.assertIn("and step < self.max_steps", source)
        self.assertIn("efficientad_training_v4_spatial_calibration", source)
        self.assertIn("ReduceLROnPlateau", source)
        self.assertIn("self._clean_validation_loss(", source)
        self.assertIn("self.load_state_dict(best_state)", source)

    def test_efficientad_hard_quantile_presets_are_documented(self):
        config = (
            Path(__file__).parents[1]
            / "configs"
            / "anomaly_detection"
            / "model"
            / "efficientad_s.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn("official=0.999", config)
        self.assertIn("balanced=0.995", config)
        self.assertIn("broad=0.99", config)
        self.assertIn("hard_quantile_preset: balanced", config)
        factory = (
            Path(__file__).parents[1]
            / "src"
            / "anomaly_detection"
            / "models"
            / "factory.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"official": 0.999', factory)
        self.assertIn('"balanced": 0.995', factory)
        self.assertIn('"broad": 0.99', factory)
        self.assertIn("hard_quantile=_efficientad_hard_quantile(cfg)", factory)
        self.assertIn("spatial_calibration_enabled: false", config)
        self.assertIn("fixed_training_duration: true", config)
        self.assertIn("mixed_precision: true", config)
        self.assertIn("checkpoint_interval: 5000", config)
        self.assertIn("image_score_topk_candidates: [0.0]", config)
        experiment = (
            Path(__file__).parents[1] / "configs" / "anomaly_detection"
            / "experiment" / "efficientad_pose_a_spatial.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("camera_poses: [A_overhead]", experiment)
        self.assertIn("spatial_calibration_enabled: true", experiment)
        self.assertIn("spatial_calibration_q_low", factory)
        self.assertIn("image_score_topk_candidates", factory)
        self.assertIn("fixed_training_duration=bool", factory)
        self.assertIn("mixed_precision=bool", factory)
        training = (
            Path(__file__).parents[1] / "src" / "anomaly_detection"
            / "models" / "trainable.py"
        ).read_text(encoding="utf-8")
        self.assertIn("torch.autocast(", training)
        self.assertIn("student_all = self.student(", training)
        self.assertIn("scaler.scale(loss).backward()", training)
        self.assertIn('"grad_scaler": scaler.state_dict()', training)
        global_experiment = (
            Path(__file__).parents[1] / "configs" / "anomaly_detection"
            / "experiment" / "efficientad_pose_a_global.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("A_overhead: [90, 240, 720, 720]", global_experiment)
        self.assertIn("fixed_training_duration: true", global_experiment)
        self.assertIn("spatial_calibration_enabled: false", global_experiment)
        self.assertIn("image_score_topk_candidates: [0.0]", global_experiment)

    def test_efficientad_spatial_score_uses_only_static_roi(self):
        model = EfficientAD(
            require_teacher_weights=False,
            channels=8,
            max_steps=1,
            spatial_calibration_enabled=True,
        )
        anomaly_map = torch.tensor([[[[100.0, 4.0], [3.0, 2.0]]]])
        model.static_roi = torch.tensor(
            [[[[False, True], [True, True]]]], dtype=torch.bool
        )
        maximum = model._aggregate_image_score(
            anomaly_map, use_static_roi=True, topk_fraction=0.0
        )
        top_half = model._aggregate_image_score(
            anomaly_map, use_static_roi=True, topk_fraction=0.5
        )
        self.assertEqual(maximum.item(), 4.0)
        self.assertEqual(top_half.item(), 3.5)

    def test_efficientad_diagnostic_modes_are_transient_and_fixed_to_max(self):
        model = EfficientAD(
            require_teacher_weights=False,
            channels=8,
            max_steps=1,
            spatial_calibration_enabled=True,
        )
        state_keys = set(model.state_dict())
        model.configure_diagnostic_inference("global_roi")
        self.assertFalse(model._diagnostic_spatial_override)
        self.assertTrue(model._diagnostic_static_roi_override)
        self.assertEqual(model._diagnostic_topk_fraction_override, 0.0)
        self.assertIsNone(model._diagnostic_global_topk_pixels_override)
        model.configure_diagnostic_inference("global")
        model.configure_diagnostic_global_topk_pixels(4)
        self.assertEqual(model._diagnostic_global_topk_pixels_override, 4)
        self.assertEqual(state_keys, set(model.state_dict()))
        model.clear_diagnostic_inference()
        self.assertIsNone(model._diagnostic_spatial_override)
        self.assertIsNone(model._diagnostic_static_roi_override)
        self.assertIsNone(model._diagnostic_topk_fraction_override)
        self.assertIsNone(model._diagnostic_global_topk_pixels_override)
        with self.assertRaises(ValueError):
            model.configure_diagnostic_inference("spatial_without_roi")
        with self.assertRaises(ValueError):
            model.configure_diagnostic_global_topk_pixels(0)

    def test_efficientad_accepts_only_the_nelson_teacher_layout(self):
        source = EfficientAD(require_teacher_weights=False, channels=8, max_steps=1)
        reverse_keys = {value: key for key, value in source._NELSON_TO_LOCAL.items()}
        nelson_state = {
            reverse_keys[key]: value.detach().clone()
            for key, value in source.teacher.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher_small.pth"
            torch.save(nelson_state, path)
            loaded = EfficientAD(
                teacher_weights_path=str(path), channels=8, max_steps=1
            )
        for expected, actual in zip(
            source.teacher.parameters(), loaded.teacher.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_supersimplenet_fit_and_frozen_backbone(self):
        kwargs = dict(
            backbone="resnet18", pretrained=False,
            weights_name="IMAGENET1K_V1", input_size=(64, 64),
            epochs=1, validation_interval=1, validation_batches=1,
            max_samples_per_epoch=2, gaussian_sigma=0.0,
            fixed_training_duration=True,
        )
        model = SuperSimpleNet(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            model.fit(
                clean_loader(), device="cpu", validation_loader=validation_loader(),
                work_dir=directory,
            )
            training_checkpoint = Path(directory) / "supersimplenet_training.ckpt"
            self.assertTrue(training_checkpoint.is_file())
            payload = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 1)
            self.assertIn("optimizer", payload)
            self.assertIn("scheduler", payload)
            resumed = SuperSimpleNet(**kwargs)
            resumed.fit(
                clean_loader(), device="cpu", validation_loader=validation_loader(),
                work_dir=directory, resume=True,
            )
            self.assertTrue(resumed.is_fitted)
        prediction = model.predict(torch.rand(1, 3, 64, 64))
        self.assertEqual(prediction.anomaly_map.shape, (1, 1, 64, 64))
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertFalse(model.features.training)
        self.assertIsNone(model.fit_summary["selected_validation_image_auroc"])

    def test_supersimplenet_synthetic_masks_stay_inside_target_wheel(self):
        generator = SuperSimpleAnomalyGenerator(
            noise_std=0.015, threshold=0.6
        )
        features = torch.zeros(2, 4, 8, 8)
        adapted = torch.zeros_like(features)
        target_masks = torch.zeros(2, 1, 8, 8)
        target_masks[:, :, 2:6, 1:7] = 1
        generated_masks = torch.ones(4, 1, 8, 8)
        with patch.object(generator, "_masks", return_value=generated_masks):
            noisy_features, noisy_adapted, masks = generator(
                features, adapted, target_masks=target_masks
            )
        expected = torch.cat((target_masks, target_masks), dim=0)
        self.assertTrue(torch.equal(masks, expected))
        outside = 1 - expected
        self.assertEqual((noisy_features * outside).abs().sum().item(), 0)
        self.assertEqual((noisy_adapted * outside).abs().sum().item(), 0)

    def test_supersimplenet_image_score_candidates_include_classifier(self):
        model = SuperSimpleNet(
            backbone="resnet18", pretrained=False, input_size=(64, 64),
            epochs=1, gaussian_sigma=0.0,
        )
        model.fitted.fill_(True)
        patch_scores = torch.tensor([
            [[[1.0, 1.0], [1.0, 1.0]]],
            [[[0.0, 0.0], [0.0, 0.0]]],
        ])
        classification_scores = torch.tensor([0.0, 1.0])
        candidates = (
            {"name": "classification_head", "mode": "classification_head"},
            {"name": "map_max", "mode": "max"},
        )
        with patch.object(
            model, "_image_score_components",
            return_value=(patch_scores, classification_scores),
        ):
            results = evaluate_image_score_aggregations(
                model, validation_loader(), candidates, device="cpu"
            )
        self.assertEqual(results["classification_head"]["image_auroc"], 1.0)
        self.assertEqual(results["map_max"]["image_auroc"], 0.0)

    def test_topk_fraction_uses_ceiling_patch_count(self):
        patch_scores = torch.arange(1, 101, dtype=torch.float32).reshape(1, 1, 10, 10)
        scores = aggregate_patch_scores(
            patch_scores,
            {"mode": "topk_fraction_mean", "fraction": 0.021},
        )
        self.assertTrue(torch.equal(scores, torch.tensor([99.0])))
        with self.assertRaisesRegex(ValueError, "topk fraction"):
            aggregate_patch_scores(
                patch_scores,
                {"mode": "topk_fraction_mean", "fraction": 0.0},
            )

    def test_supersimplenet_prediction_uses_configured_topk_fraction(self):
        model = SuperSimpleNet(
            backbone="resnet18", pretrained=False, input_size=(64, 64),
            epochs=1, gaussian_sigma=0.0,
            image_score_mode="topk_fraction_mean", image_score_fraction=0.25,
        )
        model.fitted.fill_(True)
        native_map = torch.tensor([
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[-3.0, -2.0], [-1.0, 0.0]]],
        ])
        classifier = torch.tensor([100.0, -100.0])
        with patch.object(model, "_logits", return_value=(native_map, classifier)):
            prediction, raw_score, raw_map = model.predict_with_raw(
                torch.rand(2, 3, 64, 64)
            )
        self.assertTrue(torch.equal(raw_score, torch.tensor([4.0, 0.0])))
        self.assertTrue(torch.allclose(prediction.anomaly_score, raw_score.sigmoid()))
        self.assertEqual(raw_map.shape, (2, 1, 64, 64))

    def test_supersimplenet_early_stopping_is_checkpointed_and_resumable(self):
        kwargs = dict(
            backbone="resnet18", pretrained=False,
            weights_name="IMAGENET1K_V1", input_size=(64, 64),
            epochs=5, validation_interval=1, validation_batches=None,
            max_samples_per_epoch=2, gaussian_sigma=0.0,
            fixed_training_duration=False, early_stopping_patience=1,
            early_stopping_min_delta=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            model = SuperSimpleNet(**kwargs)
            with patch(
                "src.anomaly_detection.models.supersimplenet._validation_auc",
                return_value=0.5,
            ):
                model.fit(
                    clean_loader(), device="cpu",
                    validation_loader=validation_loader(), work_dir=directory,
                )
            self.assertTrue(model.fit_summary["stopped_early"])
            self.assertEqual(model.fit_summary["epochs_completed"], 2)
            self.assertEqual(model.fit_summary["selected_epoch"], 1)
            self.assertEqual(len(model.fit_summary["validation_history"]), 2)
            checkpoint = torch.load(
                Path(directory) / "supersimplenet_training.ckpt",
                map_location="cpu", weights_only=False,
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertTrue(checkpoint["stopped_early"])
            self.assertEqual(checkpoint["no_improvement_validations"], 1)

            resumed = SuperSimpleNet(**kwargs)
            resumed.fit(
                clean_loader(), device="cpu",
                validation_loader=validation_loader(),
                work_dir=directory, resume=True,
            )
            self.assertTrue(resumed.fit_summary["stopped_early"])
            self.assertEqual(resumed.fit_summary["epochs_completed"], 2)
            self.assertEqual(resumed.fit_summary["selected_epoch"], 1)

    def test_supersimplenet_hydra_config_enables_early_stopping(self):
        config_path = (
            Path(__file__).parents[1] / "configs" / "anomaly_detection"
            / "model" / "supersimplenet.yaml"
        )
        cfg = OmegaConf.create({"model": OmegaConf.load(config_path)})
        cfg.model.pretrained = False
        model = build_model(cfg)
        self.assertFalse(model.fixed_training_duration)
        self.assertIsNone(model.validation_batches)
        self.assertEqual(model.validation_interval, 4)
        self.assertEqual(model.early_stopping_patience, 5)
        self.assertEqual(model.early_stopping_min_delta, 0.001)
        self.assertEqual(model.perlin_threshold, 0.2)
        self.assertTrue(model.restrict_synthetic_anomalies_to_target_mask)
        self.assertEqual(model.image_score_mode, "topk_fraction_mean")
        self.assertEqual(model.image_score_fraction, 0.01)
        self.assertTrue(cfg.model.pose_crop_enabled)

    def test_tinyglass_deployment_matches_eager_map(self):
        with tempfile.TemporaryDirectory() as directory:
            texture_root = Path(directory) / "dtd" / "images" / "banded"
            texture_root.mkdir(parents=True)
            from PIL import Image
            Image.new("RGB", (64, 64), (128, 64, 192)).save(
                texture_root / "texture.png"
            )
            model = TinyGLASS(
                pretrained=False, weights_name="IMAGENET1K_V1",
                input_size=(64, 64), epochs=1, learning_rate=1e-4,
                weight_decay=1e-2, validation_interval=1,
                validation_batches=None, max_samples_per_epoch=2,
                gaussian_sigma=0.0, texture_root=str(texture_root.parent),
                require_texture_dataset=True, gas_steps=1,
                fixed_training_duration=False,
            )
            model.fit(
                clean_loader(), device="cpu",
                validation_loader=validation_loader() * 2,
                work_dir=directory,
            )
            training_checkpoint = Path(directory) / "tinyglass_training.ckpt"
            self.assertTrue(training_checkpoint.is_file())
            payload = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 1)
            self.assertIn("optimizer", payload)
            self.assertIn("features", payload["model"])
            images = torch.rand(1, 3, 64, 64)
            with torch.no_grad():
                expected = model._patch_scores(images)
                actual = model.to_deployment_module()(images)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
            self.assertEqual(actual.shape, (1, 1, 4, 4))
            self.assertIsNotNone(
                model.fit_summary["selected_validation_image_auroc"]
            )
            self.assertEqual(model.fit_summary["selected_epoch"], 1)
            self.assertEqual(len(model.fit_summary["validation_history"]), 1)
            diagnostics = model.fit_summary["validation_history"][0]
            distribution = diagnostics["raw_image_score_distribution"]["all"]
            self.assertEqual(distribution["count"], 4)
            self.assertIn("greater_than_0_99_fraction", distribution)
            self.assertIn("greater_than_0_999_fraction", distribution)
            self.assertTrue(model.fit_summary["validation_feature_cache_enabled"])
            self.assertEqual(model.fit_summary["validation_cache_images"], 4)
            self.assertGreater(model.fit_summary["validation_cache_bytes"], 0)
            self.assertEqual(model.fit_summary["center_recomputations"], 1)
            self.assertEqual(
                model.fit_summary["center_patches_total"],
                model.fit_summary["center_patches"],
            )
            self.assertEqual(payload["best_epoch"], 1)
            self.assertEqual(len(payload["validation_history"]), 1)
            comparison_validation = validation_loader() * 2
            cache = _build_validation_feature_cache(
                model, comparison_validation, torch.device("cpu"), None
            )
            direct = _validation_diagnostics(
                model, comparison_validation, torch.device("cpu"), None
            )
            cached = _cached_validation_diagnostics(
                model, cache, torch.device("cpu")
            )
            self.assertEqual(cached["image_auroc"], direct["image_auroc"])
            self.assertAlmostEqual(
                cached["raw_image_score_distribution"]["all"]["mean"],
                direct["raw_image_score_distribution"]["all"]["mean"],
            )
            final_checkpoint = Path(directory) / "tinyglass.ckpt"
            model.save(final_checkpoint)
            inference_only = TinyGLASS(
                pretrained=False, weights_name="IMAGENET1K_V1",
                input_size=(64, 64), epochs=1, learning_rate=1e-4,
                weight_decay=1e-2, validation_interval=1,
                validation_batches=None, max_samples_per_epoch=2,
                gaussian_sigma=0.0, texture_root=None,
                require_texture_dataset=True, gas_steps=1,
                fixed_training_duration=False,
            )
            inference_only.load(final_checkpoint, map_location="cpu")
            self.assertTrue(inference_only.is_fitted)
            inference_only.fit(
                clean_loader(), device="cpu", validation_loader=validation_loader(),
                work_dir=directory, resume=True,
            )

    def test_tinyglass_early_stopping_is_checkpointed_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            texture_root = Path(directory) / "dtd" / "images" / "banded"
            texture_root.mkdir(parents=True)
            from PIL import Image
            Image.new("RGB", (64, 64), (128, 64, 192)).save(
                texture_root / "texture.png"
            )
            kwargs = dict(
                pretrained=False, input_size=(64, 64), epochs=5,
                max_samples_per_epoch=2, texture_root=str(texture_root.parent),
                gas_steps=1, gaussian_sigma=0.0,
                fixed_training_duration=False, early_stopping_patience=1,
            )
            distribution = {
                "count": 2, "min": 0.5, "max": 0.5,
                "mean": 0.5, "std": 0.0,
                "greater_than_0_99_fraction": 0.0,
                "greater_than_0_999_fraction": 0.0,
            }
            diagnostics = {
                "image_auroc": 0.5,
                "raw_image_score_distribution": {
                    "all": distribution, "clean": distribution,
                    "anomaly": distribution,
                },
            }
            model = TinyGLASS(**kwargs)
            with patch(
                "src.anomaly_detection.models.tinyglass."
                "_cached_validation_diagnostics",
                return_value=diagnostics,
            ):
                model.fit(
                    clean_loader(), device="cpu",
                    validation_loader=validation_loader(), work_dir=directory,
                )
            self.assertTrue(model.fit_summary["stopped_early"])
            self.assertEqual(model.fit_summary["epochs_completed"], 2)
            self.assertEqual(model.fit_summary["selected_epoch"], 1)
            checkpoint = torch.load(
                Path(directory) / "tinyglass_training.ckpt",
                map_location="cpu", weights_only=False,
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertTrue(checkpoint["stopped_early"])

            resumed = TinyGLASS(**{**kwargs, "texture_root": None})
            resumed.fit(
                clean_loader(), device="cpu",
                validation_loader=validation_loader(),
                work_dir=directory, resume=True,
            )
            self.assertTrue(resumed.fit_summary["stopped_early"])
            self.assertEqual(resumed.fit_summary["epochs_completed"], 2)

    def test_tinyglass_hydra_config_accepts_full_validation(self):
        config_path = (
            Path(__file__).parents[1] / "configs" / "anomaly_detection"
            / "model" / "tinyglass.yaml"
        )
        cfg = OmegaConf.create({"model": OmegaConf.load(config_path)})
        cfg.model.pretrained = False
        model = build_model(cfg)
        self.assertIsNone(model.validation_batches)
        self.assertTrue(model.cache_validation_features)
        self.assertEqual(model.learning_rate, 5e-5)
        self.assertEqual(model.early_stopping_patience, 40)
        self.assertTrue(model.freeze_backbone)
        self.assertEqual(model.backbone_learning_rate, 1e-5)
        self.assertTrue(model.las_restrict_to_target_mask)
        self.assertEqual(model.las_mode, "hole")
        self.assertEqual(model.las_hole_luminance_range, (0.18, 0.32))
        self.assertEqual(model.las_hole_severity_weights, (0.7, 0.25, 0.05))
        self.assertFalse(model.las.requires_textures)
        self.assertFalse(model.hypersphere_projection)
        self.assertEqual(model.feature_grid_resolution, "layer2")
        self.assertEqual(model.input_size, (384, 384))
        self.assertEqual(model.gaussian_sigma, 1.0)

    def test_tinyglass_layer2_grid_doubles_spatial_resolution(self):
        low_resolution = TinyGLASSFeatureExtractor(
            pretrained=False, weights_name="IMAGENET1K_V1",
            feature_grid_resolution="layer3",
        )
        high_resolution = TinyGLASSFeatureExtractor(
            pretrained=False, weights_name="IMAGENET1K_V1",
            feature_grid_resolution="layer2",
        )
        images = torch.rand(1, 3, 64, 64)
        self.assertEqual(low_resolution(images).shape, (1, 128, 4, 4))
        self.assertEqual(high_resolution(images).shape, (1, 128, 8, 8))
    def test_tinyglass_gas_uses_one_joint_discriminator_forward(self):
        class RecordingDiscriminator(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.batch_sizes = []

            def forward(self, features):
                self.batch_sizes.append(features.shape[0])
                return torch.sigmoid(features.mean(1))

        model = TinyGLASS(
            pretrained=False, input_size=(64, 64), epochs=1,
            require_texture_dataset=False, gas_steps=1,
        )
        discriminator = RecordingDiscriminator()
        model.discriminator = discriminator
        true_features = torch.randn(2, 128, 4, 4)
        fake_features = torch.randn_like(true_features)
        feature_mask = torch.ones(2, 1, 4, 4)
        loss = model._training_loss(
            true_features, fake_features, feature_mask
        )
        self.assertIsNotNone(loss)
        self.assertEqual(discriminator.batch_sizes, [4, 4, 2])

    def test_tinyglass_wheel_aware_las_stays_inside_target_mask(self):
        las = TinyGLASSLAS(input_size=(8, 8))
        images = torch.zeros(2, 3, 8, 8)
        textures = torch.ones_like(images)
        target_masks = torch.zeros(2, 1, 8, 8)
        target_masks[:, :, 2:6, 2:6] = 1
        with patch(
            "src.anomaly_detection.models.tinyglass._rand_perlin_2d",
            return_value=torch.ones(8, 8),
        ):
            augmented, masks = las(
                images, textures, target_masks=target_masks
            )
        self.assertTrue(masks.any())
        self.assertEqual((masks * (1 - target_masks)).sum().item(), 0)
        self.assertEqual(
            (augmented * (1 - target_masks)).sum().item(), 0
        )

    def test_tinyglass_hole_las_is_local_dark_and_texture_free(self):
        torch.manual_seed(7)
        las = TinyGLASSLAS(
            input_size=(64, 64), mode="hole",
            hole_luminance_range=(0.18, 0.32),
        )
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        images = (torch.full((2, 3, 64, 64), 0.7) - mean) / std
        target_masks = torch.zeros(2, 1, 64, 64)
        target_masks[:, :, 8:56, 8:56] = 1
        augmented, masks = las(images, target_masks=target_masks)
        self.assertFalse(las.requires_textures)
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue((masks.sum(dim=(1, 2, 3)) >= 6).all())
        self.assertEqual((masks * (1 - target_masks)).sum().item(), 0)
        unchanged = (1 - masks).expand_as(images).bool()
        self.assertTrue(torch.equal(augmented[unchanged], images[unchanged]))
        augmented_rgb = augmented * std + mean
        original_rgb = images * std + mean
        changed = masks.expand_as(images).bool()
        self.assertLess(
            augmented_rgb[changed].mean().item(),
            original_rgb[changed].mean().item(),
        )

    def test_tinyglass_hole_mode_does_not_require_dtd_during_fit(self):
        model = TinyGLASS(
            pretrained=False, input_size=(64, 64), epochs=1,
            max_samples_per_epoch=2, texture_root=None,
            require_texture_dataset=True, las_restrict_to_target_mask=True,
            las_mode="hole", gas_steps=0, gaussian_sigma=0.0,
            fixed_training_duration=True,
        )
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "label": torch.zeros(2, dtype=torch.long),
            "target_mask": torch.ones(2, 1, 64, 64),
        }
        model.fit([batch], device="cpu")
        self.assertEqual(model.fit_summary["las_mode"], "hole")
        self.assertFalse(model.fit_summary["las_requires_textures"])

    def test_tinyglass_hole_mode_requires_target_restriction(self):
        with self.assertRaisesRegex(ValueError, "requires las_restrict"):
            TinyGLASS(
                pretrained=False, input_size=(64, 64),
                require_texture_dataset=False, las_mode="hole",
            )

    def test_tinyglass_hole_severity_weights_are_validated_and_normalized(self):
        las = TinyGLASSLAS(
            input_size=(64, 64), mode="hole",
            hole_severity_weights=(70, 25, 5),
        )
        self.assertEqual(las.hole_severity_weights, (0.7, 0.25, 0.05))
        with self.assertRaisesRegex(ValueError, "three finite"):
            TinyGLASSLAS(
                input_size=(64, 64), mode="hole",
                hole_severity_weights=(0, 0, 0),
            )

    def test_tinyglass_finetuning_recomputes_center_each_epoch(self):
        model = TinyGLASS(
            pretrained=False, input_size=(64, 64), epochs=2,
            max_samples_per_epoch=2, require_texture_dataset=False,
            gas_steps=0, gaussian_sigma=0.0, fixed_training_duration=True,
            freeze_backbone=False, cache_validation_features=False,
        )
        model.textures.paths = [Path("synthetic-texture")]
        batch = {
            "image": torch.rand(2, 3, 64, 64),
            "label": torch.zeros(2, dtype=torch.long),
        }
        with (
            patch.object(model, "_compute_center", return_value=32) as compute,
            patch.object(
                model.textures, "sample",
                return_value=torch.rand(2, 3, 64, 64),
            ),
            patch.object(
                model.las, "forward",
                return_value=(
                    torch.rand(2, 3, 64, 64),
                    torch.ones(2, 1, 64, 64),
                ),
            ),
        ):
            model.fit([batch], device="cpu")
        self.assertEqual(compute.call_count, 2)
        self.assertEqual(model.fit_summary["center_recomputations"], 2)
        self.assertEqual(model.fit_summary["center_patches_total"], 64)

    def test_tinyglass_optional_backbone_fine_tuning_contract(self):
        frozen = TinyGLASS(
            pretrained=False, input_size=(64, 64), epochs=1,
            require_texture_dataset=False,
        )
        frozen.train(True)
        self.assertTrue(all(
            not parameter.requires_grad for parameter in frozen.features.parameters()
        ))
        self.assertFalse(frozen.features.training)
        self.assertTrue(frozen.cache_validation_features)

        with self.assertWarnsRegex(UserWarning, "cache disabled"):
            fine_tuned = TinyGLASS(
                pretrained=False, input_size=(64, 64), epochs=1,
                require_texture_dataset=False, freeze_backbone=False,
                backbone_learning_rate=1e-5, cache_validation_features=True,
            )
        fine_tuned.train(True)
        self.assertTrue(all(
            parameter.requires_grad for parameter in fine_tuned.features.parameters()
        ))
        self.assertTrue(fine_tuned.features.training)
        self.assertFalse(fine_tuned.cache_validation_features)
        self.assertIn("features", fine_tuned._training_state())
        self.assertFalse(
            fine_tuned.checkpoint_config()["freeze_backbone"]
        )
        self.assertEqual(
            fine_tuned.checkpoint_config()["backbone_learning_rate"], 1e-5
        )

    def test_tinyglass_checkpoint_rejects_a_different_freeze_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "tinyglass.ckpt"
            frozen = TinyGLASS(
                pretrained=False, input_size=(64, 64), epochs=1,
                require_texture_dataset=False,
            )
            frozen.save(checkpoint)
            fine_tuned = TinyGLASS(
                pretrained=False, input_size=(64, 64), epochs=1,
                require_texture_dataset=False, freeze_backbone=False,
                cache_validation_features=False,
            )
            with self.assertRaisesRegex(ValueError, "configuration does not match"):
                fine_tuned.load(checkpoint, map_location="cpu")

    def test_tinyglass_las_mask_uses_max_pooling(self):
        image_mask = torch.zeros(1, 1, 8, 8)
        image_mask[0, 0, 1, 1] = 1
        feature_mask = TinyGLASSLAS.downsample_mask(image_mask, (2, 2))
        nearest = torch.nn.functional.interpolate(image_mask, (2, 2), mode="nearest")
        self.assertEqual(feature_mask.sum().item(), 1)
        self.assertEqual(nearest.sum().item(), 0)

    def test_epoch_checkpoints_resume_after_simulated_pause(self):
        ssn_kwargs = dict(
            backbone="resnet18", pretrained=False,
            weights_name="IMAGENET1K_V1", input_size=(64, 64),
            epochs=2, max_samples_per_epoch=2, gaussian_sigma=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            interrupted = SuperSimpleNet(**ssn_kwargs)
            with self.assertRaisesRegex(RuntimeError, "simulated pause"):
                interrupted.fit(
                    InterruptingEpochLoader(fail_after=1),
                    device="cpu", work_dir=directory,
                )
            resumed = SuperSimpleNet(**ssn_kwargs)
            resumed.fit(
                InterruptingEpochLoader(), device="cpu",
                work_dir=directory, resume=True,
            )
            self.assertEqual(resumed.fit_summary["steps"], 2)

        with tempfile.TemporaryDirectory() as directory:
            texture_root = Path(directory) / "dtd" / "images" / "banded"
            texture_root.mkdir(parents=True)
            from PIL import Image
            Image.new("RGB", (64, 64), (128, 64, 192)).save(
                texture_root / "texture.png"
            )
            tiny_kwargs = dict(
                pretrained=False, weights_name="IMAGENET1K_V1",
                input_size=(64, 64), epochs=2, max_samples_per_epoch=2,
                texture_root=str(texture_root.parent), gas_steps=1,
                gaussian_sigma=0.0, fixed_training_duration=True,
            )
            interrupted = TinyGLASS(**tiny_kwargs)
            with self.assertRaisesRegex(RuntimeError, "simulated pause"):
                interrupted.fit(
                    InterruptingEpochLoader(fail_after=2),
                    device="cpu", work_dir=directory,
                )
            resumed = TinyGLASS(**tiny_kwargs)
            resumed.fit(
                InterruptingEpochLoader(), device="cpu",
                work_dir=directory, resume=True,
            )
            self.assertEqual(resumed.fit_summary["steps"], 2)

    def test_trainable_models_reject_anomalous_train_samples(self):
        model = SuperSimpleNet(
            backbone="resnet18", pretrained=False,
            weights_name="IMAGENET1K_V1", input_size=(64, 64),
            epochs=1, max_samples_per_epoch=1, gaussian_sigma=0.0,
        )
        with self.assertRaisesRegex(ValueError, "only clean"):
            model.fit([{
                "image": torch.rand(1, 3, 64, 64),
                "label": torch.ones(1, dtype=torch.long),
            }], device="cpu")

    def test_new_notebook_is_self_contained(self):
        notebook_path = (
            Path(__file__).parents[1] / "notebooks" / "anomaly_detection"
            / "kaggle_wheel_anomaly_detection_model_roadmap.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            compilable_source = "".join(
                line for line in cell.get("source", [])
                if not line.lstrip().startswith(("!", "%"))
            )
            compile(compilable_source, f"notebook-cell-{index}", "exec")
        self.assertNotIn("src.anomaly_detection", source)
        self.assertIn('MODEL_NAME = "supersimplenet"', source)
        self.assertNotIn('MODEL_PRESET = "smoke"', source)
        self.assertIn('"max_steps": 70_000', source)
        self.assertIn("nelson1425/EfficientAD", source)
        self.assertIn("class ImageNetPenaltyDataset", source)
        self.assertIn("class EfficientAD", source)
        self.assertIn("class SuperSimpleNet", source)
        self.assertIn("class TinyGLASS", source)
        self.assertIn("def to_deployment_module", source)
        self.assertNotIn("class SyntheticFeatureDetector", source)
        self.assertIn("FIXED_TRAINING_DURATION = True", source)
        self.assertIn("TINYGLASS_FIXED_TRAINING_DURATION = False", source)
        self.assertIn("greater_than_0_999_fraction", source)
        self.assertIn("_tinyglass_build_validation_feature_cache", source)
        self.assertIn('"learning_rate": 5e-5, "weight_decay": 1e-2', source)
        self.assertIn('"early_stopping_patience": 40', source)
        self.assertIn("IMAGENET1K_V1", source)
        self.assertIn("dtd-r1.0.1.tar.gz", source)
        self.assertIn("Expected 5,640 DTD R1.0.1 images", source)
        self.assertIn("supersimplenet_training.ckpt", source)
        self.assertIn("tinyglass_training.ckpt", source)
        self.assertIn("adaptive_max_pool2d", source)
        self.assertIn("TINYGLASS_IMAGE_SCORE_CANDIDATES", source)
        self.assertIn("SUPERSIMPLENET_IMAGE_SCORE_CANDIDATES", source)
        self.assertIn('"name": "classification_head"', source)
        self.assertIn('SUPERSIMPLENET_FIXED_IMAGE_SCORE_NAME = "map_topk_mean_1pct"', source)
        self.assertIn('"mode": "topk_fraction_mean"', source)
        self.assertIn('"image_score_fraction": 0.01', source)
        self.assertIn('selection_strategy = "fixed_pre_registered"', source)
        self.assertIn('"restrict_synthetic_anomalies_to_target_mask": True', source)
        self.assertIn('f"{MODEL_NAME}_image_score_aggregation.json"', source)
        self.assertIn("EVALUATION_ONLY_FROM_DRIVE = False", source)
        self.assertIn("DRIVE_CHECKPOINT_FILE_ID = None", source)
        self.assertIn("DRIVE_CHECKPOINT_RUN_FOLDER_ID = None", source)
        self.assertIn("def download_drive_checkpoint", source)
        self.assertIn("MediaIoBaseDownload", source)
        self.assertIn("def restore_supersimplenet_from_drive", source)
        self.assertIn("Drive checkpoint evaluation contract mismatch", source)
        self.assertIn(
            'constructor_config["image_score_mode"] = '
            'SELECTED_MODEL_CONFIG["image_score_mode"]',
            source,
        )
        self.assertIn("Training skipped; evaluating restored", source)
        self.assertIn('"freeze_backbone": True', source)
        self.assertIn('"backbone_learning_rate": 1e-5', source)
        self.assertIn('"las_restrict_to_target_mask": True', source)
        self.assertIn('"las_mode": "hole"', source)
        self.assertIn('"las_hole_luminance_range": (0.18, 0.32)', source)
        self.assertIn('"las_hole_severity_weights": (0.70, 0.25, 0.05)', source)
        self.assertIn("TINYGLASS_INPUT_SIZE = (384, 384)", source)
        self.assertIn('"input_size": TINYGLASS_INPUT_SIZE', source)
        self.assertIn('"gaussian_sigma": 1.0', source)
        self.assertIn("TINYGLASS_FINE_TUNE_EXPERIMENT = False", source)
        self.assertIn("if self.freeze_backbone", source)
        self.assertIn("joint_scores = self.discriminator", source)
        self.assertIn("center_recomputations", source)
        self.assertIn('MODEL_NAME = "supersimplenet"', source)
        self.assertIn('"A_overhead": (90, 240, 720, 720)', source)
        self.assertIn("build_efficientad_train_calibration_loaders", source)
        self.assertIn("EFFICIENTAD_IMAGE_SCORE_TOPK_CANDIDATES", source)
        self.assertIn("selected_spatial_calibration", source)
        self.assertIn('"features": self.features.state_dict()', source)
        self.assertIn("def aggregate_patch_scores", source)
        self.assertIn("selection_split\": \"validation", source)
        self.assertIn("TINYGLASS_GAUSSIAN_SIGMA_CANDIDATES", source)
        self.assertIn("SUPERSIMPLENET_GAUSSIAN_SIGMA_CANDIDATES", source)
        self.assertIn(
            'SUPERSIMPLENET_GAUSSIAN_SIGMA_SELECTION_METRIC', source
        )
        self.assertIn("evaluate_gaussian_sigma_ablation", source)
        self.assertIn("tinyglass_gaussian_sigma_ablation.json", source)
        self.assertIn("supersimplenet_gaussian_sigma_ablation.json", source)
        self.assertIn("EFFICIENTAD_FIXED_TRAINING_DURATION = True", source)
        self.assertIn("EFFICIENTAD_SPATIAL_CALIBRATION_ENABLED = False", source)
        self.assertIn("EFFICIENTAD_CALIBRATION_ABLATION_ENABLED = False", source)
        self.assertIn("EFFICIENTAD_GLOBAL_TOPK_NATIVE_PIXELS = (2, 4, 8, 16)", source)
        self.assertIn("validation was not used for model selection", source)
        self.assertIn('"input_size": (384, 384)', source)
        self.assertIn('if key != "batch_size"', source)
        self.assertIn("evaluate_efficientad_calibration_ablation", source)
        self.assertIn("efficientad_calibration_ablation.json", source)
        self.assertIn("evaluate_efficientad_global_topk_ablation", source)
        self.assertIn("efficientad_global_topk_ablation.json", source)
        self.assertIn('"test_used_for_selection": False', source)
        self.assertNotIn("resume is not implemented", source)


if __name__ == "__main__":
    unittest.main()
