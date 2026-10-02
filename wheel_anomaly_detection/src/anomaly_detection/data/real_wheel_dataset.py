from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import pil_to_tensor

from .preprocessing import IMAGENET_MEAN, IMAGENET_STD, RGBTriplet


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_real_wheel_records(root: str | Path) -> list[dict[str, Any]]:
    """Join the three fixed manifests of Real Mars Rover Wheel AD."""
    root = Path(root)
    samples = _read_csv(root / "real_samples.csv")
    crops = {
        row["image_id"]: row for row in _read_csv(root / "real_crops.csv")
    }
    masks = {
        row["image_id"]: row for row in _read_csv(root / "hole_masks.csv")
    }

    records: list[dict[str, Any]] = []
    for sample_row in samples:
        image_id = sample_row["image_id"]
        crop_row = crops[image_id]
        records.append(
            {
                **sample_row,
                "mask_path": masks.get(image_id, {}).get("mask_path", ""),
                "crop_top": int(crop_row["crop_top"]),
                "crop_left": int(crop_row["crop_left"]),
                "crop_height": int(crop_row["crop_height"]),
                "crop_width": int(crop_row["crop_width"]),
                "crop_policy": crop_row["crop_policy"],
                "crop_version": crop_row["crop_version"],
            }
        )
    return records


class RealWheelCropDataset(Dataset[dict[str, Any]]):
    """Load the held-out MAHLI pilot with aligned per-image wheel crops."""

    def __init__(
        self,
        root: str | Path,
        *,
        output_size: tuple[int, int] = (384, 384),
        normalize_mean: RGBTriplet | None = IMAGENET_MEAN,
        normalize_std: RGBTriplet | None = IMAGENET_STD,
    ) -> None:
        self.root = Path(root)
        self.output_size = output_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.records = load_real_wheel_records(self.root)

    def _open_image(self, relative_path: str, mode: str) -> Image.Image:
        with Image.open(self.root / relative_path) as opened:
            return opened.convert(mode)

    @staticmethod
    def _crop_box(record: dict[str, Any]) -> tuple[int, int, int, int]:
        return (
            int(record["crop_top"]),
            int(record["crop_left"]),
            int(record["crop_height"]),
            int(record["crop_width"]),
        )

    def __len__(self) -> int:
        return len(self.records)

    def load_native_crop(
        self, index: int
    ) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
        """Return unresized RGB/mask crops for visual QA."""
        record = self.records[index]
        image = self._open_image(record["local_path"], "RGB")
        if record["mask_path"]:
            mask = self._open_image(record["mask_path"], "L")
        else:
            mask = Image.new("L", image.size, color=0)
        top, left, height, width = self._crop_box(record)
        box = (left, top, left + width, top + height)
        return image.crop(box), mask.crop(box), dict(record)

    def load_original(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
        """Return the original RGB frame and metadata for crop-box audits."""
        record = self.records[index]
        return self._open_image(record["local_path"], "RGB"), dict(record)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, anomaly_mask, record = self.load_native_crop(index)
        image_tensor = pil_to_tensor(image)
        mask_tensor = pil_to_tensor(anomaly_mask)

        image_tensor = TF.resize(
            image_tensor,
            self.output_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask_tensor = TF.resize(
            mask_tensor,
            self.output_size,
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.convert_image_dtype(image_tensor, torch.float32)
        if self.normalize_mean is not None:
            image_tensor = TF.normalize(
                image_tensor,
                mean=self.normalize_mean,
                std=self.normalize_std,
            )

        return {
            "image": image_tensor,
            "anomaly_mask": mask_tensor,
            "label": torch.tensor(record["condition"] == "hole", dtype=torch.long),
            "has_anomaly_mask": bool(record["mask_path"]),
            "metadata": {
                key: value
                for key, value in record.items()
                if key not in {"local_path", "mask_path"}
            },
        }

    def image_for_display(self, image: torch.Tensor) -> torch.Tensor:
        result = image.detach().clone()
        if self.normalize_mean is not None:
            mean = result.new_tensor(self.normalize_mean).view(3, 1, 1)
            std = result.new_tensor(self.normalize_std).view(3, 1, 1)
            result = result * std + mean
        return result.clamp(0.0, 1.0)
