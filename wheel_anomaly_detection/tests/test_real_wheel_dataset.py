import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.anomaly_detection.data.real_wheel_dataset import (
    RealWheelCropDataset,
)


SAMPLE_COLUMNS = [
    "image_id",
    "sol",
    "instrument",
    "acquisition_date_utc",
    "wheel_sequence",
    "rotation_index",
    "condition",
    "local_path",
]
CROP_COLUMNS = [
    "image_id",
    "crop_top",
    "crop_left",
    "crop_height",
    "crop_width",
    "crop_policy",
    "crop_version",
]


class RealWheelDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "nested" / "pilot"
        (self.root / "images" / "no_visible_hole").mkdir(parents=True)
        (self.root / "images" / "hole").mkdir(parents=True)
        (self.root / "masks" / "hole").mkdir(parents=True)

        samples = [
            self._sample("clean", "no_visible_hole"),
            self._sample("damaged", "hole"),
        ]
        self._write_csv(self.root / "real_samples.csv", SAMPLE_COLUMNS, samples)
        crops = [self._crop("clean"), self._crop("damaged")]
        self._write_csv(self.root / "real_crops.csv", CROP_COLUMNS, crops)
        self._write_csv(
            self.root / "hole_masks.csv",
            ["image_id", "mask_path"],
            [{"image_id": "damaged", "mask_path": "masks/hole/damaged.png"}],
        )

        Image.new("RGB", (12, 10), color=(10, 20, 30)).save(
            self.root / "images" / "no_visible_hole" / "clean.png"
        )
        damaged = Image.new("RGB", (12, 10), color=(10, 20, 30))
        damaged.putpixel((5, 4), (255, 0, 0))
        damaged.save(self.root / "images" / "hole" / "damaged.png")
        mask = Image.new("L", (12, 10), color=0)
        for x in (4, 5):
            for y in (3, 4):
                mask.putpixel((x, y), 255)
        mask.save(self.root / "masks" / "hole" / "damaged.png")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _sample(image_id: str, condition: str) -> dict[str, str]:
        return {
            "image_id": image_id,
            "sol": "1",
            "instrument": "MAHLI",
            "acquisition_date_utc": "2026-01-01T00:00:00",
            "wheel_sequence": "1",
            "rotation_index": "1",
            "condition": condition,
            "local_path": f"images/{condition}/{image_id}.png",
        }

    @staticmethod
    def _crop(image_id: str) -> dict[str, str]:
        return {
            "image_id": image_id,
            "crop_top": "1",
            "crop_left": "2",
            "crop_height": "8",
            "crop_width": "8",
            "crop_policy": "manual_wheel_geometry",
            "crop_version": "test_v1",
        }

    def _dataset(self) -> RealWheelCropDataset:
        return RealWheelCropDataset(
            self.root,
            output_size=(4, 4),
            normalize_mean=None,
            normalize_std=None,
        )

    def test_loads_aligned_crops_and_zero_mask_for_negative(self) -> None:
        dataset = self._dataset()
        clean = dataset[0]
        damaged = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(clean["image"].shape, (3, 4, 4))
        self.assertEqual(clean["anomaly_mask"].shape, (1, 4, 4))
        self.assertEqual(clean["image"].dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(clean["anomaly_mask"]).item(), 0)
        self.assertFalse(clean["has_anomaly_mask"])
        self.assertEqual(damaged["label"].item(), 1)
        self.assertTrue(damaged["has_anomaly_mask"])
        self.assertGreater(torch.count_nonzero(damaged["anomaly_mask"]).item(), 0)
        self.assertEqual(damaged["metadata"]["crop_version"], "test_v1")

    def test_real_kaggle_notebook_is_self_contained(self) -> None:
        notebook_path = (
            Path(__file__).parents[1]
            / "notebooks"
            / "kaggle_real_wheel_zero_shot.ipynb"
        )
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertNotIn("src.anomaly_detection", source)
        self.assertIn("class RealWheelCropDataset", source)
        self.assertNotIn("def discover_real_dataset_root", source)
        self.assertIn("real_crops.csv", source)
        self.assertIn("InterpolationMode.NEAREST", source)
        self.assertIn(
            'DATASET_ROOT = Path("/kaggle/input/datasets/dalphan01/real-mars-rover-wheel-ad/real_wheel_pilot_v2")',
            source,
        )
        self.assertNotIn("audit_real_crop_contract", source)


if __name__ == "__main__":
    unittest.main()
