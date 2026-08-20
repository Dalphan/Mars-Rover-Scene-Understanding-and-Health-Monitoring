import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from PIL import Image

from src.anomaly_detection.data import (
    CuriosityWheelDataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessingConfig,
    WheelPreprocessor,
    build_dataloaders,
)


COLUMNS = [
    "image_id",
    "split",
    "condition",
    "image_path",
    "target_mask_path",
    "anomaly_mask_path",
    "pair_id",
    "target_wheel",
    "camera_pose",
    "lighting",
    "wear",
    "severity",
    "surface",
    "sector",
]


class AnomalyDataloaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rows = [
            self._make_sample("train_clean", "train", "clean"),
            self._make_sample("validation_clean", "validation", "clean", "pair_val"),
            self._make_sample("validation_hole", "validation", "hole", "pair_val"),
            self._make_sample("test_clean", "test", "clean", "pair_test"),
            self._make_sample("test_hole", "test", "hole", "pair_test"),
        ]
        with (self.root / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _dataloader_config(self, batch_size: int = 4):
        config_path = (
            Path(__file__).parents[1]
            / "configs"
            / "anomaly_detection"
            / "config.yaml"
        )
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_path.parent.resolve()),
        ):
            config = compose(config_name="config")
        config.dataset.root = str(self.root)
        config.dataloader.batch_size = batch_size
        config.dataloader.num_workers = 0
        config.dataloader.pin_memory = False
        return config

    def _make_sample(
        self,
        image_id: str,
        split: str,
        condition: str,
        pair_id: str = "",
    ) -> dict[str, str]:
        image_path = Path("images") / split / condition / f"{image_id}.png"
        target_path = Path("masks") / "target_wheel" / split / condition / f"{image_id}.png"
        anomaly_path = (
            Path("masks") / "anomaly" / split / "hole" / f"{image_id}.png"
            if condition == "hole"
            else None
        )

        for path in (image_path, target_path, anomaly_path):
            if path is not None:
                (self.root / path).parent.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (4, 3), color=(10, 20, 30)).save(self.root / image_path)
        Image.new("L", (4, 3), color=255).save(self.root / target_path)
        if anomaly_path is not None:
            Image.new("L", (4, 3), color=255).save(self.root / anomaly_path)

        return {
            "image_id": image_id,
            "split": split,
            "condition": condition,
            "image_path": image_path.as_posix(),
            "target_mask_path": target_path.as_posix(),
            "anomaly_mask_path": anomaly_path.as_posix() if anomaly_path else "",
            "pair_id": pair_id,
            "target_wheel": "wheel_front_left",
            "camera_pose": "A_overhead",
            "lighting": "mars_dusty_refined",
            "wear": "wear_light",
            "severity": "small" if condition == "hole" else "",
            "surface": "tread" if condition == "hole" else "",
            "sector": "leading" if condition == "hole" else "",
        }

    def test_clean_sample_has_stable_zero_mask_schema(self) -> None:
        sample = CuriosityWheelDataset(self.root, "train")[0]

        self.assertEqual(sample["image"].shape, (3, 3, 4))
        self.assertEqual(sample["target_mask"].shape, (1, 3, 4))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertAlmostEqual(sample["image"][0, 0, 0].item(), 10 / 255)
        self.assertEqual(sample["label"].item(), 0)
        self.assertFalse(sample["has_anomaly_mask"])
        self.assertEqual(torch.count_nonzero(sample["anomaly_mask"]).item(), 0)

    def test_hole_sample_loads_anomaly_mask_and_metadata(self) -> None:
        dataset = CuriosityWheelDataset(self.root, "validation")
        sample = dataset[1]

        self.assertEqual(sample["label"].item(), 1)
        self.assertTrue(sample["has_anomaly_mask"])
        self.assertEqual(torch.unique(sample["anomaly_mask"]).tolist(), [255])
        self.assertEqual(sample["metadata"]["pair_id"], "pair_val")

    def test_builds_three_collatable_dataloaders(self) -> None:
        train_loader, validation_loader, test_loader = build_dataloaders(
            self._dataloader_config(batch_size=2)
        )

        self.assertEqual(len(train_loader.dataset), 1)
        self.assertEqual(len(validation_loader.dataset), 2)
        self.assertEqual(len(test_loader.dataset), 2)
        validation_batch = next(iter(validation_loader))
        self.assertEqual(validation_batch["image"].shape, (2, 3, 256, 256))
        self.assertEqual(validation_batch["label"].tolist(), [0, 1])
        self.assertEqual(validation_batch["has_anomaly_mask"].tolist(), [False, True])

    def test_resize_is_aligned_and_preserves_mask_values(self) -> None:
        preprocessing = WheelPreprocessor(PreprocessingConfig(resize=(6, 8)))
        sample = CuriosityWheelDataset(
            self.root,
            "validation",
            preprocessing=preprocessing,
        )[1]

        self.assertEqual(sample["image"].shape, (3, 6, 8))
        self.assertEqual(sample["target_mask"].shape, (1, 6, 8))
        self.assertEqual(sample["anomaly_mask"].shape, (1, 6, 8))
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(torch.unique(sample["anomaly_mask"]).tolist(), [255])

    def test_normalization_is_optional_and_uses_rgb_statistics(self) -> None:
        preprocessing = WheelPreprocessor(
            PreprocessingConfig(
                normalize_mean=(0.0, 0.0, 0.0),
                normalize_std=(1.0, 1.0, 1.0),
            )
        )
        sample = CuriosityWheelDataset(
            self.root,
            "train",
            preprocessing=preprocessing,
        )[0]

        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertAlmostEqual(sample["image"][0, 0, 0].item(), 10 / 255)
        self.assertEqual(sample["target_mask"].dtype, torch.uint8)

    def test_normalized_image_can_be_restored_for_display(self) -> None:
        raw = CuriosityWheelDataset(self.root, "train")[0]["image"]
        preprocessing = WheelPreprocessor(
            PreprocessingConfig(
                normalize_mean=IMAGENET_MEAN,
                normalize_std=IMAGENET_STD,
            )
        )
        normalized, _, _ = preprocessing(
            raw,
            torch.zeros((1, 3, 4), dtype=torch.uint8),
            torch.zeros((1, 3, 4), dtype=torch.uint8),
        )

        self.assertTrue(torch.allclose(preprocessing.image_for_display(normalized), raw))

    def test_augmentations_change_rgb_but_not_masks(self) -> None:
        torch.manual_seed(7)
        preprocessing = WheelPreprocessor(
            PreprocessingConfig(
                brightness=0.1,
                contrast=0.1,
                gamma=0.1,
                saturation=0.1,
                sensor_noise=0.005,
                gaussian_noise=0.005,
                gaussian_blur=(0.1, 0.5),
            )
        )
        raw = CuriosityWheelDataset(self.root, "validation")[1]
        augmented = CuriosityWheelDataset(
            self.root,
            "validation",
            preprocessing=preprocessing,
        )[1]

        self.assertEqual(augmented["image"].dtype, torch.float32)
        self.assertFalse(
            torch.equal(
                augmented["image"],
                raw["image"],
            )
        )
        self.assertTrue(torch.equal(augmented["target_mask"], raw["target_mask"]))
        self.assertTrue(torch.equal(augmented["anomaly_mask"], raw["anomaly_mask"]))

    def test_rejects_incomplete_normalization_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be set together"):
            PreprocessingConfig(normalize_mean=(0.5, 0.5, 0.5))

    def test_train_and_evaluation_can_use_different_preprocessing(self) -> None:
        cfg = self._dataloader_config(batch_size=1)
        cfg.preprocessing.train.augmentations_enabled = True
        cfg.preprocessing.train.gaussian_noise = 0.01
        train_loader, validation_loader, _ = build_dataloaders(cfg)

        train_image = next(iter(train_loader))["image"]
        validation_image = next(iter(validation_loader))["image"]
        self.assertEqual(train_image.dtype, torch.float32)
        self.assertEqual(validation_image.dtype, torch.float32)
        self.assertFalse(torch.equal(train_image, validation_image))

    def test_patchcore_owns_resize_and_disables_train_augmentation(self) -> None:
        cfg = self._dataloader_config(batch_size=1)
        train_loader, validation_loader, _ = build_dataloaders(cfg)

        self.assertEqual(tuple(cfg.model.input_size), (256, 256))
        self.assertFalse(cfg.model.train_augmentations_enabled)
        self.assertFalse(train_loader.dataset.preprocessing.config.augmentations_enabled)
        self.assertFalse(
            validation_loader.dataset.preprocessing.config.augmentations_enabled
        )

    def test_train_and_evaluation_share_imagenet_numerical_contract(self) -> None:
        cfg = self._dataloader_config(batch_size=2)
        for split_config in (cfg.preprocessing.train, cfg.preprocessing.evaluation):
            split_config.normalize_mean = IMAGENET_MEAN
            split_config.normalize_std = IMAGENET_STD
        cfg.preprocessing.train.gaussian_noise = 0.001
        train_loader, validation_loader, test_loader = build_dataloaders(cfg)

        for loader in (train_loader, validation_loader, test_loader):
            image = next(iter(loader))["image"]
            self.assertEqual(image.dtype, torch.float32)
            self.assertTrue(torch.isfinite(image).all())
        expected_red = ((10 / 255) - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        validation_red = next(iter(validation_loader))["image"][0, 0, 0, 0].item()
        self.assertAlmostEqual(validation_red, expected_red, places=5)

    def test_rejects_unknown_split_even_when_loading_another_split(self) -> None:
        manifest = self.root / "samples.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
        rows[-1]["split"] = "tset"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "Unsupported split"):
            CuriosityWheelDataset(self.root, "train")

    def test_rejects_duplicate_image_id(self) -> None:
        manifest = self.root / "samples.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
        rows[-1]["image_id"] = rows[0]["image_id"]
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "Duplicate image_id"):
            CuriosityWheelDataset(self.root, "validation")

    def test_rejects_incomplete_pair(self) -> None:
        manifest = self.root / "samples.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
        rows[-1]["pair_id"] = "different_pair"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "exactly one clean and one hole"):
            CuriosityWheelDataset(self.root, "train")

    def test_checks_all_split_artifacts_when_dataset_is_built(self) -> None:
        missing = self.root / "images" / "validation" / "hole" / "validation_hole.png"
        missing.unlink()

        with self.assertRaisesRegex(FileNotFoundError, "validation_hole.png"):
            CuriosityWheelDataset(self.root, "validation")

    def test_rejects_hole_without_anomaly_mask(self) -> None:
        manifest = self.root / "samples.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
        rows[-1]["anomaly_mask_path"] = ""
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "Hole sample has no anomaly mask"):
            CuriosityWheelDataset(self.root, "test")

    def test_rejects_clean_with_anomaly_mask(self) -> None:
        manifest = self.root / "samples.csv"
        rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
        rows[0]["anomaly_mask_path"] = "masks/anomaly/train/hole/unexpected.png"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "Clean sample unexpectedly has an anomaly mask"):
            CuriosityWheelDataset(self.root, "train")

    def test_kaggle_notebook_is_self_contained(self) -> None:
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
        self.assertNotIn("PROJECT_ROOT", source)
        self.assertIn("class CuriosityWheelDataset", source)
        self.assertIn("def build_dataloaders", source)
        self.assertIn("class PreprocessingConfig", source)
        self.assertIn("def audit_preprocessing", source)
        self.assertIn("self._validate_manifest_rows(rows)", source)
        self.assertIn("Duplicate image_id in samples.csv", source)
        self.assertIn("EXPECTED_SPLIT_CONDITION_COUNTS", source)
        self.assertIn("evaluation_preprocessing=EVALUATION_PREPROCESSING", source)
        self.assertIn("# Configuration", source)
        self.assertIn("IMAGE_SIZE = (256, 256)", source)
        self.assertIn("TRAIN_AUGMENTATIONS_ENABLED = False", source)
        self.assertIn("images must be finite float32 tensors", source)


if __name__ == "__main__":
    unittest.main()
