from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from .preprocessing import WheelPreprocessor


REQUIRED_COLUMNS = {
    "image_id",
    "split",
    "condition",
    "image_path",
    "target_mask_path",
    "anomaly_mask_path",
    "pair_id",
}
VALID_SPLITS = ("train", "validation", "test")
VALID_CONDITIONS = {"clean", "hole"}
ARTIFACT_PATH_FIELDS = ("image_path", "target_mask_path", "anomaly_mask_path")


class CuriosityWheelDataset(Dataset[dict[str, Any]]):
    """Load one split of the extracted Curiosity wheel Kaggle dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        preprocessing: WheelPreprocessor | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.preprocessing = preprocessing or WheelPreprocessor()

        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split {split!r}; expected one of {VALID_SPLITS}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.root}")

        manifest_path = self.root / "samples.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")

        with manifest_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"samples.csv is missing required columns: {sorted(missing)}")
            rows = list(reader)

        self._validate_manifest_rows(rows)
        self.rows = [row for row in rows if row["split"] == split]

        if not self.rows:
            raise ValueError(f"No samples found for split {split!r}")

        for row in self.rows:
            for field in ARTIFACT_PATH_FIELDS:
                relative_path = row[field]
                if relative_path and not (self.root / relative_path).is_file():
                    raise FileNotFoundError(
                        f"Dataset artifact is missing: {self.root / relative_path}"
                    )

    @classmethod
    def _validate_manifest_rows(cls, rows: list[dict[str, str]]) -> None:
        """Validate the complete manifest before selecting one split."""
        if not rows:
            raise ValueError("samples.csv contains no samples")

        image_ids: set[str] = set()
        pairs: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            image_id = row["image_id"]
            if not image_id:
                raise ValueError("samples.csv contains an empty image_id")
            if image_id in image_ids:
                raise ValueError(f"Duplicate image_id in samples.csv: {image_id}")
            image_ids.add(image_id)

            row_split = row["split"]
            if row_split not in VALID_SPLITS:
                raise ValueError(
                    f"Unsupported split {row_split!r} in sample {image_id}; "
                    f"expected one of {VALID_SPLITS}"
                )

            condition = row["condition"]
            if condition not in VALID_CONDITIONS:
                raise ValueError(f"Unsupported condition {condition!r} in sample {image_id}")
            if not row["image_path"] or not row["target_mask_path"]:
                raise ValueError(f"Sample has incomplete artifact paths: {image_id}")
            if condition == "hole" and not row["anomaly_mask_path"]:
                raise ValueError(f"Hole sample has no anomaly mask: {image_id}")
            if condition == "clean" and row["anomaly_mask_path"]:
                raise ValueError(f"Clean sample unexpectedly has an anomaly mask: {image_id}")
            for field in ARTIFACT_PATH_FIELDS:
                if row[field]:
                    cls._validate_relative_path(row[field])

            pair_id = row["pair_id"]
            if condition == "hole" and not pair_id:
                raise ValueError(f"Hole sample has no pair_id: {image_id}")
            if pair_id:
                pairs.setdefault(pair_id, []).append(row)

        for pair_id, pair_rows in pairs.items():
            conditions = {row["condition"] for row in pair_rows}
            splits = {row["split"] for row in pair_rows}
            if len(pair_rows) != 2 or conditions != VALID_CONDITIONS:
                raise ValueError(
                    f"Pair {pair_id!r} must contain exactly one clean and one hole sample"
                )
            if len(splits) != 1:
                raise ValueError(f"Pair {pair_id!r} crosses dataset splits: {sorted(splits)}")

    @staticmethod
    def _validate_relative_path(value: str) -> None:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid dataset-relative path: {value!r}")

    def _load_image(self, relative_path: str, mode: str) -> Image.Image:
        path = self.root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Dataset artifact is missing: {path}")
        with Image.open(path) as image:
            return image.convert(mode)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = pil_to_tensor(self._load_image(row["image_path"], "RGB"))
        target_mask = pil_to_tensor(self._load_image(row["target_mask_path"], "L"))

        if image.shape[1:] != target_mask.shape[1:]:
            raise ValueError(f"Image/target mask size mismatch for {row['image_id']}")

        anomaly_path = row["anomaly_mask_path"]
        if anomaly_path:
            anomaly_mask = pil_to_tensor(self._load_image(anomaly_path, "L"))
            if anomaly_mask.shape != target_mask.shape:
                raise ValueError(f"Target/anomaly mask size mismatch for {row['image_id']}")
        else:
            # Clean samples use an empty mask so every batch has one stable schema.
            anomaly_mask = torch.zeros_like(target_mask)

        image, target_mask, anomaly_mask = self.preprocessing(
            image,
            target_mask,
            anomaly_mask,
        )

        return {
            "image": image,
            "target_mask": target_mask,
            "anomaly_mask": anomaly_mask,
            "label": torch.tensor(row["condition"] == "hole", dtype=torch.long),
            "has_anomaly_mask": bool(anomaly_path),
            "metadata": {
                key: value
                for key, value in row.items()
                if key not in {"image_path", "target_mask_path", "anomaly_mask_path"}
            },
        }
