import importlib
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from datasets.common import image_to_numpy, mask_to_numpy, to_tensor_sample
from taxonomies import CORE_VALID_CLASSES, mapping_for_dataset, remap_mask


SUPPORTED_MARS_BENCH_DATASETS = {
    "Mirali33/mb-mars_seg_mer",
    "Mirali33/mb-mars_seg_msl",
    "Mirali33/mb-s5mars",
}


class MarsBenchHFDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        taxonomy: str = "core",
        image_key: str = "image",
        mask_key: str = "mask",
        config_name: Optional[str] = None,
        transform=None,
    ):
        if dataset_name not in SUPPORTED_MARS_BENCH_DATASETS:
            raise ValueError(
                f"Unsupported Mars-Bench dataset: {dataset_name}. "
                f"Supported: {sorted(SUPPORTED_MARS_BENCH_DATASETS)}"
            )

        self.dataset = _load_hf_dataset(dataset_name, split=split, config_name=config_name)
        self.dataset_name = dataset_name
        self.short_name = dataset_name.rsplit("/", 1)[-1]
        self.split = split
        self.taxonomy = taxonomy
        self.image_key = image_key
        self.mask_key = mask_key
        self.transform = transform
        self.mapping = mapping_for_dataset(dataset_name, taxonomy=taxonomy)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        record = self.dataset[idx]
        image_np = image_to_numpy(record[self.image_key])
        native_mask = mask_to_numpy(record[self.mask_key])
        mask_np = remap_mask(native_mask, self.mapping)

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask_np)
            image_np = transformed["image"]
            mask_np = transformed["mask"]

        image_tensor, mask_tensor = to_tensor_sample(image_np, mask_np)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": self.short_name,
            "valid_classes": CORE_VALID_CLASSES,
            "image_id": self._image_id(record, idx),
        }

    def audit(self, max_samples: Optional[int] = None) -> dict:
        original_values = []
        mapped_values = []
        limit = len(self.dataset) if max_samples is None else min(max_samples, len(self.dataset))
        for idx in range(limit):
            native_mask = mask_to_numpy(self.dataset[idx][self.mask_key])
            original_values.append(native_mask)
            mapped_values.append(remap_mask(native_mask, self.mapping))

        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "images": len(self.dataset),
            "masks": len(self.dataset),
            "missing_masks": 0,
            "original_label_values": _sorted_unique(original_values),
            "mapped_label_values": _sorted_unique(mapped_values),
            "audited_samples": limit,
        }

    def _image_id(self, record: dict, idx: int) -> str:
        for key in ("image_id", "id", "filename", "file_name", "path"):
            if key in record and record[key] is not None:
                return str(record[key])
        return f"{self.short_name}-{idx}"


def audit_mars_bench(
    dataset_name: str,
    split: str,
    taxonomy: str = "core",
    image_key: str = "image",
    mask_key: str = "mask",
    config_name: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> dict:
    dataset = MarsBenchHFDataset(
        dataset_name=dataset_name,
        split=split,
        taxonomy=taxonomy,
        image_key=image_key,
        mask_key=mask_key,
        config_name=config_name,
    )
    return dataset.audit(max_samples=max_samples)


def _sorted_unique(values: list[np.ndarray]) -> list[int]:
    if not values:
        return []
    return sorted({int(value) for array in values for value in np.unique(array)})


def _load_hf_dataset(dataset_name: str, split: str, config_name: Optional[str] = None):
    repo_root = Path(__file__).resolve().parents[1]
    saved_path = list(sys.path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "datasets" or name.startswith("datasets.")
    }

    try:
        sys.path = [
            path
            for path in sys.path
            if Path(path or ".").resolve() != repo_root
        ]
        for name in list(sys.modules):
            if name == "datasets" or name.startswith("datasets."):
                del sys.modules[name]

        module = importlib.import_module("datasets")
        module_file = Path(getattr(module, "__file__", "")).resolve()
        if repo_root in module_file.parents:
            raise ImportError("Imported the local datasets package instead of Hugging Face datasets.")
        if config_name:
            return module.load_dataset(dataset_name, config_name, split=split)
        return module.load_dataset(dataset_name, split=split)
    finally:
        for name in list(sys.modules):
            if name == "datasets" or name.startswith("datasets."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path = saved_path
