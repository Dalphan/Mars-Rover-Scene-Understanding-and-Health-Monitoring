from __future__ import annotations

import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger(__name__)

class QuantizationTransform:
    """Exact resize and ImageNet normalization used by PTQ/QAT notebooks."""

    def __init__(self, size, mean, std) -> None:
        self.size = tuple(int(value) for value in size)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __call__(self, image, mask):
        width, height = self.size[1], self.size[0]
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).contiguous()
        image_tensor = (image_tensor - self.mean) / self.std
        mask_array = np.asarray(mask, dtype=np.int64)
        if mask_array.ndim == 3:
            mask_array = mask_array[..., 0]
        return image_tensor.float(), torch.from_numpy(mask_array).long()


def select_dataset_subset(dataset, max_samples, strategy="random", seed=42):
    if max_samples is None or int(max_samples) >= len(dataset):
        return dataset
    sample_count = int(max_samples)
    if sample_count <= 0:
        raise ValueError("max_samples must be positive or null")
    rng = np.random.default_rng(int(seed))
    shuffled_indices = rng.permutation(len(dataset)).tolist()
    if strategy == "random":
        return dataset.select(shuffled_indices[:sample_count])
    if strategy != "class_aware":
        raise ValueError("selection must be random or class_aware")

    labels_by_index = [set(labels) for labels in dataset["class_labels"]]
    all_labels = sorted(set().union(*labels_by_index))
    pools = {
        label: [index for index in shuffled_indices if label in labels_by_index[index]]
        for label in all_labels
    }
    positions = {label: 0 for label in all_labels}
    presence_counts = {label: 0 for label in all_labels}
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < sample_count:
        available = []
        for label in all_labels:
            pool = pools[label]
            while positions[label] < len(pool) and pool[positions[label]] in selected_set:
                positions[label] += 1
            if positions[label] < len(pool):
                available.append(label)
        if not available:
            break
        target = min(available, key=lambda label: (presence_counts[label], label))
        index = pools[target][positions[target]]
        positions[target] += 1
        selected.append(index)
        selected_set.add(index)
        for label in labels_by_index[index]:
            presence_counts[label] += 1
    if len(selected) < sample_count:
        selected.extend(index for index in shuffled_indices if index not in selected_set)
    LOGGER.info("Class-aware subset presence: %s", presence_counts)
    return dataset.select(selected[:sample_count])


class SegmentationQuantizationDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        split: str,
        transform,
        token: str | None = None,
        max_samples: int | None = None,
        selection: str = "random",
        seed: int = 42,
    ) -> None:
        dataset = load_dataset(repo_id, split=split, token=token)
        required = {"image", "mask", "class_labels"}
        missing = required - set(dataset.column_names)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        self.source_fingerprint = dataset._fingerprint
        self.source_sample_count = len(dataset)
        dataset = dataset.add_column("_source_index", list(range(len(dataset))))
        self.dataset = select_dataset_subset(dataset, max_samples, selection, seed)
        self.subset_fingerprint = self.dataset._fingerprint
        self.selected_source_indices = [
            int(value) for value in self.dataset["_source_index"]
        ]
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image, mask = self.transform(
            sample["image"].convert("RGB"), sample["mask"].convert("L")
        )
        return {
            "image": image,
            "mask": mask,
            "index": int(index),
            "source_index": int(sample["_source_index"]),
        }


def segmentation_collate_fn(batch):
    return {
        "image": torch.stack([sample["image"] for sample in batch]),
        "mask": torch.stack([sample["mask"] for sample in batch]),
        "index": torch.tensor([sample["index"] for sample in batch], dtype=torch.long),
        "source_index": torch.tensor(
            [sample["source_index"] for sample in batch], dtype=torch.long
        ),
    }


def build_dataset(dataset_cfg, split, max_samples=None, selection="random", seed=42):
    token = os.environ.get("HF_TOKEN") or None
    transform = QuantizationTransform(
        dataset_cfg.image_size, dataset_cfg.image_mean, dataset_cfg.image_std
    )
    return SegmentationQuantizationDataset(
        repo_id=str(dataset_cfg.repo_id),
        split=str(split),
        token=token,
        transform=transform,
        max_samples=max_samples,
        selection=str(selection),
        seed=int(seed),
    )


def build_dataloader(
    dataset, batch_size: int, num_workers: int, shuffle=False, seed: int = 42
):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=segmentation_collate_fn,
        generator=torch.Generator().manual_seed(int(seed)),
    )
