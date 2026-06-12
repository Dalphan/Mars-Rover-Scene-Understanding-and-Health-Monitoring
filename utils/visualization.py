from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


def build_palette(num_classes: int) -> List[tuple]:
    base = [
        (220, 20, 60),
        (119, 11, 32),
        (0, 128, 0),
        (255, 215, 0),
        (70, 130, 180),
        (138, 43, 226),
        (255, 140, 0),
        (0, 206, 209),
    ]
    if num_classes <= len(base):
        return base[:num_classes]

    rng = np.random.RandomState(0)
    palette = list(base)
    for _ in range(num_classes - len(base)):
        palette.append(tuple(rng.randint(0, 255, size=3).tolist()))
    return palette


def _colorize_mask(mask: np.ndarray, palette: List[tuple], ignore_index: Optional[int]):
    height, width = mask.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)
    for idx, color_val in enumerate(palette):
        color[mask == idx] = color_val
    if ignore_index is not None:
        color[mask == ignore_index] = (0, 0, 0)
    return color


def _denormalize(image: np.ndarray, mean: Optional[List[float]], std: Optional[List[float]]):
    if mean is None or std is None:
        return image
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)
    return image * std_arr + mean_arr


def save_prediction_grid(
    images,
    masks,
    preds,
    output_dir: str,
    epoch: int,
    max_samples: int,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = images[:max_samples].detach().cpu()
    masks = masks[:max_samples].detach().cpu()
    preds = preds[:max_samples].detach().cpu()

    if num_classes is None:
        num_classes = int(max(masks.max().item(), preds.max().item())) + 1

    palette = build_palette(num_classes)
    rows = []

    for idx in range(images.shape[0]):
        image = images[idx].permute(1, 2, 0).numpy()
        image = _denormalize(image, mean, std)
        image = (image * 255.0).clip(0, 255).astype(np.uint8)

        mask = masks[idx].numpy()
        pred = preds[idx].numpy()

        mask_color = _colorize_mask(mask, palette, ignore_index)
        pred_color = _colorize_mask(pred, palette, ignore_index)

        img_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask_color)
        pred_pil = Image.fromarray(pred_color)

        width, height = img_pil.size
        row = Image.new("RGB", (width * 3, height))
        row.paste(img_pil, (0, 0))
        row.paste(mask_pil, (width, 0))
        row.paste(pred_pil, (width * 2, 0))
        rows.append(row)

    grid = Image.new("RGB", (rows[0].size[0], rows[0].size[1] * len(rows)))
    for idx, row in enumerate(rows):
        grid.paste(row, (0, idx * row.size[1]))

    grid.save(output_path / f"val_preds_epoch_{epoch:03d}.png")
