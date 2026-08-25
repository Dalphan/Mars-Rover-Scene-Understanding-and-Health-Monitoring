import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.anomaly_detection.models import EfficientAD, SuperSimpleNet, TinyGLASS


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
            checkpoint = Path(directory) / "efficientad.ckpt"
            model.save(checkpoint)
            restored = EfficientAD(**kwargs)
            restored.load(checkpoint, map_location="cpu")
            self.assertTrue(restored.is_fitted)

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
        model = SuperSimpleNet(
            backbone="resnet18", pretrained=False,
            weights_name="IMAGENET1K_V1", input_size=(64, 64),
            epochs=1, validation_interval=1, validation_batches=1,
            max_samples_per_epoch=2, gaussian_sigma=0.0,
            fixed_training_duration=True,
        )
        model.fit(clean_loader(), device="cpu", validation_loader=validation_loader())
        prediction = model.predict(torch.rand(1, 3, 64, 64))
        self.assertEqual(prediction.anomaly_map.shape, (1, 1, 64, 64))
        self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
        self.assertFalse(model.features.training)
        self.assertIsNone(model.fit_summary["selected_validation_image_auroc"])

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
                validation_batches=1, max_samples_per_epoch=2,
                gaussian_sigma=0.0, texture_root=str(texture_root.parent),
                require_texture_dataset=True, gas_steps=1,
                fixed_training_duration=True,
            )
            model.fit(
                clean_loader(), device="cpu",
                validation_loader=validation_loader(),
            )
            images = torch.rand(1, 3, 64, 64)
            with torch.no_grad():
                expected = model._patch_scores(images)
                actual = model.to_deployment_module()(images)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))
            self.assertEqual(actual.shape, (1, 1, 4, 4))
            self.assertIsNone(model.fit_summary["selected_validation_image_auroc"])

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
        self.assertNotIn("src.anomaly_detection", source)
        self.assertIn('MODEL_NAME = "efficientad_s"', source)
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
        self.assertIn("IMAGENET1K_V1", source)
        self.assertIn("dtd-r1.0.1.tar.gz", source)
        self.assertIn("Expected 5,640 DTD R1.0.1 images", source)


if __name__ == "__main__":
    unittest.main()
