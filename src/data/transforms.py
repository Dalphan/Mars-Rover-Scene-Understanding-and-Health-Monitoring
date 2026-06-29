from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image


class SegmentationTransform:
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

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if image_array.ndim == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()

        if self.normalize_enabled:
            image_tensor = (image_tensor - self.mean) / self.std

        mask_array = np.asarray(mask, dtype=np.int64)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        mask_tensor = torch.from_numpy(mask_array).long()
        return image_tensor.float(), mask_tensor
