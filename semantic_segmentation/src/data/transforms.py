from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image


class SegmentationTransform:
    """Legacy tensor transform used by the dataset-analysis entrypoint."""
    def __init__(
        self,
        resize_enabled: bool,
        size: tuple[int, int],
        normalize_enabled: bool,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.resize_enabled = resize_enabled
        self.size = size
        self.normalize_enabled = normalize_enabled
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resize_enabled:
            width, height = self.size[1], self.size[0]
            image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
            mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)

        # Raw RGB image: [512, 512, 3].
        # Tensor image after transform: [3, 512, 512].
        # Batched image: [B, 3, 512, 512].
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

        if self.normalize_enabled:
            image_tensor = (image_tensor - self.mean) / self.std

        # Raw mask: [512, 512].
        # Mask values are Mars-Bench semantic class IDs.
        # Batched mask: [B, 512, 512].
        mask_array = np.asarray(mask, dtype=np.int64)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor.float(), mask_tensor


class NumpySegmentationTransform:
    """Notebook-synchronized resize and train-only geometric augmentation."""

    def __init__(
        self,
        size: tuple[int, int],
        horizontal_flip_prob: float = 0.0,
        vertical_flip_prob: float = 0.0,
        random_rotate90_prob: float = 0.0,
    ) -> None:
        self.size = tuple(int(value) for value in size)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.random_rotate90_prob = float(random_rotate90_prob)
        probabilities = {
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "vertical_flip_prob": self.vertical_flip_prob,
            "random_rotate90_prob": self.random_rotate90_prob,
        }
        invalid = {
            name: value
            for name, value in probabilities.items()
            if not 0.0 <= value <= 1.0
        }
        if invalid:
            raise ValueError(
                f"Augmentation probabilities must be in [0, 1], got {invalid}"
            )

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        image = image.resize(
            (self.size[1], self.size[0]), resample=Image.Resampling.BILINEAR
        )
        mask = mask.resize(
            (self.size[1], self.size[0]), resample=Image.Resampling.NEAREST
        )
        image_array = np.asarray(image, dtype=np.uint8).copy()
        mask_array = np.asarray(mask, dtype=np.uint8).copy()
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]

        # One random decision is shared by image and mask for each transform.
        if random.random() < self.horizontal_flip_prob:
            image_array = np.flip(image_array, axis=1)
            mask_array = np.flip(mask_array, axis=1)
        if random.random() < self.vertical_flip_prob:
            image_array = np.flip(image_array, axis=0)
            mask_array = np.flip(mask_array, axis=0)
        if random.random() < self.random_rotate90_prob:
            quarter_turns = random.randrange(4)
            if quarter_turns:
                image_array = np.rot90(image_array, k=quarter_turns, axes=(0, 1))
                mask_array = np.rot90(mask_array, k=quarter_turns, axes=(0, 1))

        return np.ascontiguousarray(image_array), np.ascontiguousarray(mask_array)


def prepare_batch(
    batch: dict[str, np.ndarray],
    device: torch.device | str,
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert the compact worker payload and normalize it on the target device."""

    images = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2).contiguous()
    masks = torch.from_numpy(batch["mask"])
    images = images.to(device, non_blocking=True).float().div_(255.0)
    masks = masks.to(device, non_blocking=True).long()
    mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
    std_tensor = images.new_tensor(std).view(1, 3, 1, 1)
    images.sub_(mean_tensor).div_(std_tensor)
    return images, masks
