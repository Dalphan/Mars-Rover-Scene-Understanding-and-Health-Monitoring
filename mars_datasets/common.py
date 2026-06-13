from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def image_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.array(value.convert("RGB"))
    array = np.asarray(value)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    return array


def mask_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        array = np.array(value)
    else:
        array = np.asarray(value)

    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        elif np.all(array[:, :, 0] == array[:, :, 1]) and np.all(array[:, :, 0] == array[:, :, 2]):
            array = array[:, :, 0]
        else:
            array = rgb_mask_to_ids(array)
    return array.astype(np.int64)


def rgb_mask_to_ids(mask: np.ndarray) -> np.ndarray:
    rgb = mask[:, :, :3].astype(np.int64)
    if rgb.max() <= 8:
        return rgb[:, :, 0]

    known_colors = {
        (0, 0, 0): 0,
        (1, 1, 1): 1,
        (2, 2, 2): 2,
        (3, 3, 3): 3,
        (255, 255, 255): 255,
    }
    out = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for color, class_id in known_colors.items():
        out[np.all(rgb == color, axis=2)] = class_id
    return out


def to_tensor_sample(image: Any, mask: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    elif isinstance(image, torch.Tensor):
        image = image.float()

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(mask, torch.Tensor):
        mask = mask.long()

    return image, mask


def collect_image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "-mask", "_mxy", "-mxy", "_label", "-label", "_labels", "-labels"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem
