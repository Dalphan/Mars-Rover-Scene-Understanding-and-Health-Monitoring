from __future__ import annotations

import io
import logging
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image


class MarsBenchHFDataset(torch.utils.data.Dataset):
    """RAM-aware Hugging Face dataset used by the training notebook.

    Image and mask columns remain encoded in the Arrow table.  Worker
    processes decode one sample at a time and return compact NumPy ``uint8``
    arrays; conversion, normalization, and the device copy happen in the main
    process.  This avoids accumulating PyTorch shared-memory storages.
    """

    def __init__(
        self,
        repo_id: str,
        split: str,
        num_classes: int,
        token: str | None = None,
        cache_dir: str | None = None,
        transform: Callable[[Image.Image, Image.Image], tuple[np.ndarray, np.ndarray]] | None = None,
        include_class_labels: bool = False,
        dataset_name: str | None = None,
        expected_split_sizes: Mapping[str, int] | None = None,
        image_column: str = "image",
        mask_column: str = "mask",
        width_column: str = "width",
        height_column: str = "height",
        class_labels_column: str = "class_labels",
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.repo_id = str(repo_id)
        self.split = str(split)
        self.num_classes = int(num_classes)
        self.dataset_name = dataset_name or self.repo_id
        self.transform = transform
        self.image_column = image_column
        self.mask_column = mask_column
        self.class_labels_column = class_labels_column
        expected_split_sizes = dict(expected_split_sizes or {})
        token = None if token in (None, "", "HF_TOKEN_PLACEHOLDER") else token

        self.logger.info("Loading %s split=%s", self.repo_id, self.split)
        loaded = load_dataset(
            self.repo_id,
            split=self.split,
            token=token,
            cache_dir=cache_dir,
        )
        required = {
            image_column,
            mask_column,
            width_column,
            height_column,
            class_labels_column,
        }
        missing = required - set(loaded.column_names)
        if missing:
            raise ValueError(
                f"Dataset split {self.split!r} is missing required columns: "
                f"{sorted(missing)}; available={sorted(loaded.column_names)}"
            )

        # Sampling metadata is deliberately retained only in the parent.
        self.class_labels = (
            tuple(loaded[class_labels_column]) if include_class_labels else None
        )
        self.dataset = (
            loaded.select_columns([image_column, mask_column])
            .cast_column(image_column, HFImage(decode=False))
            .cast_column(mask_column, HFImage(decode=False))
        )

        expected_size = expected_split_sizes.get(self.split)
        if expected_size is not None and len(self.dataset) != int(expected_size):
            self.logger.warning(
                "Dataset size drift: dataset=%s split=%s expected=%d actual=%d",
                self.dataset_name,
                self.split,
                int(expected_size),
                len(self.dataset),
            )
        self.logger.info(
            "Loaded dataset=%s repo=%s split=%s samples=%d worker_columns=%s",
            self.dataset_name,
            self.repo_id,
            self.split,
            len(self.dataset),
            self.dataset.column_names,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def release_class_labels(self) -> None:
        self.class_labels = None

    @staticmethod
    def _open_encoded_image(record: Any) -> Image.Image:
        if isinstance(record, Image.Image):
            return record.copy()
        if not isinstance(record, dict):
            raise TypeError(
                "Expected an encoded Hugging Face image record, got "
                f"{type(record).__name__}"
            )
        payload = record.get("bytes")
        image_path = record.get("path")
        if payload is not None:
            return Image.open(io.BytesIO(payload))
        if image_path:
            return Image.open(image_path)
        raise ValueError("Encoded image record has neither bytes nor path")

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if self.transform is None:
            raise RuntimeError("MarsBenchHFDataset requires a synchronized transform")

        sample = self.dataset[index]
        with self._open_encoded_image(sample[self.image_column]) as raw_image:
            image = raw_image.convert("RGB")
            with self._open_encoded_image(sample[self.mask_column]) as raw_mask:
                mask = raw_mask.convert("L")
                image_array, mask_array = self.transform(image, mask)

        invalid_ids = np.setdiff1d(np.unique(mask_array), np.arange(self.num_classes))
        if invalid_ids.size:
            raise ValueError(
                f"{self.dataset_name} split={self.split} sample={index} contains "
                f"invalid mask IDs {invalid_ids.tolist()}; expected "
                f"0..{self.num_classes - 1}"
            )
        return {"image": image_array, "mask": mask_array}
