from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch


def make_palette(num_classes: int) -> dict[int, tuple[int, int, int]]:
    base = [
        (30, 30, 30),
        (166, 118, 29),
        (0, 109, 119),
        (131, 56, 236),
        (230, 57, 70),
        (255, 183, 3),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
    ]
    return {class_id: base[class_id % len(base)] for class_id in range(num_classes)}


def _to_numpy_image(image) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    return np.asarray(image)


def _to_numpy_mask(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask)


def _class_name(class_names: dict, class_id: int) -> str:
    return str(class_names.get(class_id, class_names.get(str(class_id), f"class_{class_id}")))


def _legend_handles(class_names: dict, palette: dict[int, tuple[int, int, int]]) -> list[Patch]:
    handles = []
    for class_id, color in sorted(palette.items()):
        rgb = tuple(channel / 255.0 for channel in color)
        handles.append(Patch(facecolor=rgb, edgecolor="black", label=f"{class_id}: {_class_name(class_names, class_id)}"))
    return handles


def _add_class_legend(fig, class_names: dict, palette: dict[int, tuple[int, int, int]]) -> None:
    fig.legend(
        handles=_legend_handles(class_names, palette),
        loc="center right",
        title="Classes",
        frameon=True,
        borderaxespad=0.8,
    )


def colorize_mask(mask: np.ndarray, palette: dict[int, tuple[int, int, int]], ignore_index=0) -> np.ndarray:
    mask = _to_numpy_mask(mask)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in palette.items():
        if class_id == int(ignore_index):
            color_mask[mask == class_id] = (20, 20, 20)
        else:
            color_mask[mask == class_id] = color
    return color_mask


def overlay_mask(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = _to_numpy_image(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)
    overlay = ((1.0 - alpha) * image + alpha * color_mask).astype(np.uint8)
    background = np.all(color_mask == (20, 20, 20), axis=-1)
    overlay[background] = image[background]
    return overlay


def denormalize_image(image_tensor: torch.Tensor, mean, std) -> np.ndarray:
    tensor = image_tensor.detach().cpu().float()
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.as_tensor(std, dtype=torch.float32).view(3, 1, 1)
    tensor = tensor * std_tensor + mean_tensor
    tensor = tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return tensor


def plot_sample(image, mask, class_names, palette, alpha, save_path=None, title=None):
    image_np = _to_numpy_image(image)
    mask_np = _to_numpy_mask(mask)
    color_mask = colorize_mask(mask_np, palette)
    overlay = overlay_mask(image_np, color_mask, alpha)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    if title:
        fig.suptitle(title)
    axes[0].imshow(np.clip(image_np, 0, 1) if image_np.dtype != np.uint8 else image_np)
    axes[0].set_title("Image")
    axes[1].imshow(color_mask)
    axes[1].set_title("Mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    _add_class_legend(fig, class_names, palette)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_sample_grid(dataset, indices, class_names, palette, alpha, save_path, logger: logging.Logger | None = None):
    logger = logger or logging.getLogger(__name__)
    indices = list(indices)
    logger.info("Saving sample grid with %d samples to %s", len(indices), save_path)
    rows = len(indices)
    fig, axes = plt.subplots(rows, 3, figsize=(14, max(4, 3 * rows)))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(indices):
        sample = dataset[index]
        image = sample["image"]
        if getattr(dataset, "transform", None) is not None and dataset.transform.normalize_enabled:
            image_np = denormalize_image(image, dataset.transform.mean.flatten(), dataset.transform.std.flatten())
        else:
            image_np = _to_numpy_image(image)
        mask_np = _to_numpy_mask(sample["mask"])
        color_mask = colorize_mask(mask_np, palette)
        overlay = overlay_mask(image_np, color_mask, alpha)

        axes[row, 0].imshow(np.clip(image_np, 0, 1))
        axes[row, 0].set_title(f"Image #{index}")
        axes[row, 1].imshow(color_mask)
        axes[row, 1].set_title("Mask")
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        for axis in axes[row]:
            axis.axis("off")

    _add_class_legend(fig, class_names, palette)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
