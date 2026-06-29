from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image


class S5MarsHFDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        split: str,
        token: str | None,
        cache_dir: str | None,
        image_column: str,
        mask_column: str,
        class_labels_column: str,
        image_mode: str,
        transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]] | None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.repo_id = repo_id
        self.split = split
        self.image_column = image_column
        self.mask_column = mask_column
        self.class_labels_column = class_labels_column
        self.image_mode = image_mode
        self.transform = transform

        self.logger.info("Loading Hugging Face dataset repo=%s split=%s", repo_id, split)
        self.dataset = load_dataset(repo_id, split=split, token=token, cache_dir=cache_dir)
        self._validate_columns()
        self.logger.info(
            "Loaded dataset: repo=%s split=%s samples=%d columns=%s",
            repo_id,
            split,
            len(self.dataset),
            self.dataset.column_names,
        )

    def _validate_columns(self) -> None:
        required = {
            self.image_column,
            self.mask_column,
            "width",
            "height",
            self.class_labels_column,
        }
        available = set(self.dataset.column_names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"Dataset split '{self.split}' is missing required columns: {missing}. "
                f"Available columns: {sorted(available)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = self._as_pil(sample[self.image_column], self.image_column).convert(self.image_mode)
        mask = self._as_pil(sample[self.mask_column], self.mask_column).convert("L")

        if self.transform is None:
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous().float()
            mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64)).long()
        else:
            image_tensor, mask_tensor = self.transform(image, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "class_labels": list(sample[self.class_labels_column]),
            "width": int(sample["width"]),
            "height": int(sample["height"]),
            "index": int(index),
        }

    @staticmethod
    def _as_pil(value: Any, column_name: str) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        raise TypeError(f"Column '{column_name}' must contain PIL images, got {type(value)!r}")
