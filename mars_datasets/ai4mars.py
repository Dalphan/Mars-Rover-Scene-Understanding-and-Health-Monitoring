from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from mars_datasets.common import mask_to_numpy, to_tensor_sample
from taxonomies import CORE_VALID_CLASSES, IGNORE_INDEX, mapping_for_dataset, remap_mask


@dataclass(frozen=True)
class AI4MARSSample:
    image_path: Path
    mask_path: Optional[Path]
    range_mask_path: Optional[Path] = None
    rover_mask_path: Optional[Path] = None


def discover_ai4mars_samples(root: str | Path, split: Optional[str] = None) -> list[AI4MARSSample]:
    msl_root = _resolve_msl_root(root)
    print(f"[AI4MARS discover] MSL root resolved to: {msl_root}")
    split_name = split or "train"

    if split_name == "train":
        return [*_mcam_samples(msl_root), *_ncam_samples(msl_root, "train")]
    if split_name in {"val", "validation", "test"}:
        return _ncam_samples(msl_root, "test")
    raise ValueError(f"Unsupported AI4MARS split: {split_name}")


def audit_ai4mars(root: str | Path, split: Optional[str] = None, taxonomy: str = "core") -> dict:
    split_name = split or "train"
    print(f"[AI4MARS audit] start split={split_name} root={root}")
    samples = discover_ai4mars_samples(root=root, split=split)
    mapping = mapping_for_dataset("ai4mars", taxonomy=taxonomy)
    print(f"[AI4MARS audit] discovered {len(samples)} image/label pairs")

    original_values: set[int] = set()
    mapped_values: set[int] = set()
    missing_masks = 0
    progress_step = max(1, len(samples) // 10) if samples else 1

    for index, sample in enumerate(samples, start=1):
        if sample.mask_path is None:
            missing_masks += 1
            continue

        try:
            mask = mask_to_numpy(Image.open(sample.mask_path))
        except FileNotFoundError:
            missing_masks += 1
            print(f"[AI4MARS audit] missing label skipped: {sample.mask_path}")
            continue

        mask_values = {int(value) for value in np.unique(mask)}
        original_values.update(mask_values)
        mapped_values.update(mapping.get(value, IGNORE_INDEX) for value in mask_values)

        if _mask_contains_positive(sample.range_mask_path):
            mapped_values.add(IGNORE_INDEX)
        if _mask_contains_positive(sample.rover_mask_path):
            mapped_values.add(IGNORE_INDEX)

        if index == 1 or index == len(samples) or index % progress_step == 0:
            print(f"[AI4MARS audit] processed {index}/{len(samples)} samples")

    print(
        "[AI4MARS audit] done "
        f"images={len(samples)} masks={len(samples) - missing_masks} missing_masks={missing_masks}"
    )

    return {
        "dataset": "ai4mars",
        "split": split,
        "images": len(samples),
        "masks": len(samples) - missing_masks,
        "missing_masks": missing_masks,
        "original_label_values": sorted(original_values),
        "mapped_label_values": sorted(mapped_values),
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

    def _load_legacy_split(
        self,
        split_file: str,
        images_dir: str,
        masks_dir: str,
    ) -> list[AI4MARSSample]:
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


def _resolve_msl_root(root: str | Path) -> Path:
    root_path = Path(root)
    msl_root = root_path / "AI4MARSv0-6" / "ai4mars-dataset-merged-0.6" / "msl"
    if (msl_root / "mcam").exists() and (msl_root / "ncam").exists():
        return msl_root
    raise FileNotFoundError(f"AI4MARS MSL root not found: {msl_root}")


def _mcam_samples(msl_root: Path) -> list[AI4MARSSample]:
    images_dir = msl_root / "mcam" / "images"
    labels_dir = msl_root / "mcam" / "labels" / "train"
    samples = []
    skipped = 0
    for image_path in _jpgs(images_dir):
        sample = _sample_with_required_label(
            image_path=image_path,
            label_path=labels_dir / f"{image_path.stem}_15033_merged.png",
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
    print(f"[AI4MARS discover] mcam train: {len(samples)} samples, {skipped} missing labels")
    return samples


def _ncam_samples(msl_root: Path, split: str) -> list[AI4MARSSample]:
    images_dir = msl_root / "ncam" / "images" / "edr"
    rover_dir = msl_root / "ncam" / "images" / "mxy"
    range_dir = msl_root / "ncam" / "images" / "rng-30m"
    labels_dir = msl_root / "ncam" / "labels" / "train"
    if split == "test":
        labels_dir = msl_root / "ncam" / "labels" / "test" / "masked-gold-min3-100agree"

    samples = []
    skipped = 0
    range_masks = _png_lookup(range_dir)
    rover_masks = _png_lookup(rover_dir)

    for image_path in _jpgs(images_dir):
        sample = _sample_with_required_label(
            image_path=image_path,
            label_path=labels_dir / f"{image_path.stem}.png",
            range_mask_path=_optional_mask_from_lookup(range_masks, image_path.stem),
            rover_mask_path=_optional_mask_from_lookup(rover_masks, image_path.stem),
        )
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
    print(f"[AI4MARS discover] ncam {split}: {len(samples)} samples, {skipped} missing labels")
    return samples


def _sample_with_required_label(
    image_path: Path,
    label_path: Path,
    range_mask_path: Optional[Path] = None,
    rover_mask_path: Optional[Path] = None,
) -> Optional[AI4MARSSample]:
    try:
        label_path.stat()
    except FileNotFoundError:
        return None
    return AI4MARSSample(
        image_path=image_path,
        mask_path=label_path,
        range_mask_path=range_mask_path,
        rover_mask_path=rover_mask_path,
    )


def _jpgs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg"}
    )


def _png_lookup(path: Path) -> dict[str, Path]:
    if not path.exists():
        return {}
    return {
        file_path.name: file_path
        for file_path in path.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".png"
    }


def _optional_mask_from_lookup(files: dict[str, Path], stem: str) -> Optional[Path]:
    return files.get(f"{stem}.png") or files.get(f"{stem}_merged.png")


def _mask_contains_positive(path: Optional[Path]) -> bool:
    if path is None:
        return False
    try:
        return bool(np.any(mask_to_numpy(Image.open(path)) == 1))
    except FileNotFoundError:
        return False
