from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.common import collect_image_files, mask_to_numpy, normalized_stem, to_tensor_sample
from taxonomies import CORE_VALID_CLASSES, IGNORE_INDEX, mapping_for_dataset, remap_mask


@dataclass(frozen=True)
class AI4MARSSample:
    image_path: Path
    mask_path: Optional[Path]
    range_mask_path: Optional[Path] = None
    rover_mask_path: Optional[Path] = None


def discover_ai4mars_samples(root: str | Path, split: Optional[str] = None) -> list[AI4MARSSample]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"AI4MARS root not found: {root_path}")

    search_root = root_path / split if split and (root_path / split).exists() else root_path
    files = collect_image_files(search_root)

    image_files = [path for path in files if _looks_like_image(path)]
    mask_files = [path for path in files if _looks_like_label_mask(path)]
    range_files = [path for path in files if _looks_like_range_mask(path)]
    rover_files = [path for path in files if _looks_like_rover_mask(path)]

    masks_by_stem = _index_by_stem(mask_files)
    range_by_stem = _index_by_stem(range_files)
    rover_by_stem = _index_by_stem(rover_files)

    return [
        AI4MARSSample(
            image_path=image_path,
            mask_path=masks_by_stem.get(normalized_stem(image_path)),
            range_mask_path=range_by_stem.get(normalized_stem(image_path)),
            rover_mask_path=rover_by_stem.get(normalized_stem(image_path)),
        )
        for image_path in image_files
    ]


def audit_ai4mars(root: str | Path, split: Optional[str] = None, taxonomy: str = "core") -> dict:
    samples = discover_ai4mars_samples(root=root, split=split)
    mapping = mapping_for_dataset("ai4mars", taxonomy=taxonomy)

    original_values = []
    mapped_values = []
    for sample in samples:
        if sample.mask_path is None:
            continue
        mask = mask_to_numpy(Image.open(sample.mask_path))
        original_values.append(mask)
        mapped_values.append(_apply_ai4mars_masks(mask, sample, mapping))

    return {
        "dataset": "ai4mars",
        "split": split,
        "images": len(samples),
        "masks": sum(sample.mask_path is not None for sample in samples),
        "missing_masks": sum(sample.mask_path is None for sample in samples),
        "original_label_values": _sorted_unique(original_values),
        "mapped_label_values": _sorted_unique(mapped_values),
    }


class AI4MARSFolderDataset(Dataset):
    def __init__(
        self,
        root: Optional[str] = None,
        split: Optional[str] = None,
        taxonomy: str = "core",
        transform=None,
        use_rover_mask_as_label: bool = False,
        split_file: Optional[str] = None,
        images_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
    ):
        self.root = Path(root) if root is not None else None
        self.split = split
        self.taxonomy = taxonomy
        self.transform = transform
        self.use_rover_mask_as_label = use_rover_mask_as_label
        self.mapping = mapping_for_dataset("ai4mars", taxonomy=taxonomy)
        self.items = self._load_items(split_file, images_dir, masks_dir)

        if not self.items:
            location = self.root if self.root is not None else images_dir
            raise FileNotFoundError(f"No AI4MARS images discovered under: {location}")

    def _load_items(
        self,
        split_file: Optional[str],
        images_dir: Optional[str],
        masks_dir: Optional[str],
    ) -> list[AI4MARSSample]:
        if split_file or images_dir or masks_dir:
            if not split_file or not images_dir or not masks_dir:
                raise ValueError("split_file, images_dir, and masks_dir must be provided together.")
            return self._load_legacy_split(split_file, images_dir, masks_dir)

        if self.root is None:
            raise ValueError("AI4MARS root is required when legacy split paths are not provided.")
        return discover_ai4mars_samples(self.root, split=self.split)

    def _load_legacy_split(self, split_file: str, images_dir: str, masks_dir: str) -> list[AI4MARSSample]:
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        items = []
        with split_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [part for part in line.replace(",", " ").split() if part]
                image_rel = parts[0]
                mask_rel = parts[1] if len(parts) > 1 else parts[0]
                items.append(AI4MARSSample(Path(images_dir) / image_rel, Path(masks_dir) / mask_rel))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample = self.items[idx]
        if sample.mask_path is None:
            raise FileNotFoundError(f"Missing AI4MARS mask for image: {sample.image_path}")

        image_np = np.array(Image.open(sample.image_path).convert("RGB"))
        native_mask = mask_to_numpy(Image.open(sample.mask_path))
        mask_np = _apply_ai4mars_masks(native_mask, sample, self.mapping)

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed["image"]
            mask_np = transformed["mask"]

        image_tensor, mask_tensor = to_tensor_sample(image_np, mask_np)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": "ai4mars",
            "valid_classes": CORE_VALID_CLASSES,
            "image_id": sample.image_path.stem,
        }


def _apply_ai4mars_masks(mask: np.ndarray, sample: AI4MARSSample, mapping: dict[int, int]) -> np.ndarray:
    mapped = remap_mask(mask, mapping)

    if sample.range_mask_path is not None:
        range_mask = mask_to_numpy(Image.open(sample.range_mask_path))
        mapped[range_mask == 1] = IGNORE_INDEX

    if sample.rover_mask_path is not None:
        rover_mask = mask_to_numpy(Image.open(sample.rover_mask_path))
        mapped[rover_mask == 1] = IGNORE_INDEX

    return mapped


def _index_by_stem(paths: list[Path]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        indexed.setdefault(normalized_stem(path), path)
    return indexed


def _looks_like_image(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if any(part in {"masks", "mask", "labels", "label", "mxy", "rng", "range", "rover"} for part in parts):
        return False
    if any(token in name for token in ("mask", "label", "mxy", "rng", "range", "rover")):
        return False
    return "images" in parts or "image" in parts or "edr" in parts


def _looks_like_label_mask(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if _looks_like_range_mask(path) or _looks_like_rover_mask(path):
        return False
    return any(part in {"masks", "mask", "labels", "label", "mxy"} for part in parts) or any(
        token in name for token in ("mask", "label", "mxy")
    )


def _looks_like_range_mask(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "rng" in parts or "range" in parts or "rng" in name or "range" in name


def _looks_like_rover_mask(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    return "rover" in parts or "rover" in name


def _sorted_unique(values: list[np.ndarray]) -> list[int]:
    if not values:
        return []
    return sorted({int(value) for array in values for value in np.unique(array)})
