from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from mars_datasets.ai4mars import AI4MARSFolderDataset
from mars_datasets.mars_bench import MarsBenchHFDataset
from utils.augmentations import build_transforms


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.loaded_stages = set()

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self._setup_fit()
        if stage in (None, "test"):
            self._setup_test()

    def _setup_fit(self):
        if "fit" in self.loaded_stages and self.train_dataset is not None and self.val_dataset is not None:
            print("[DataModule] fit datasets already loaded; skipping setup.")
            return

        self.train_dataset = self._build_dataset("train", is_train=True)
        self.val_dataset = self._build_dataset("val", is_train=False)
        self.loaded_stages.add("fit")

    def _setup_test(self):
        if "test" in self.loaded_stages and self.test_dataset is not None:
            print("[DataModule] test dataset already loaded; skipping setup.")
            return

        self.test_dataset = self._build_dataset("test", is_train=False)
        self.loaded_stages.add("test")

    def _build_dataset(self, split: str, is_train: bool):
        transform = build_transforms(self.cfg.data, is_train=is_train)
        name = self.cfg.data.name

        if name == "ai4mars":
            split_cfg = getattr(self.cfg.data, split, None)
            return AI4MARSFolderDataset(
                root=_cfg_get(self.cfg.data, "root", None),
                split=_cfg_get(split_cfg, "name", split),
                taxonomy=_cfg_get(self.cfg.data, "taxonomy", "core"),
                use_rover_mask_as_label=_cfg_get(self.cfg.data, "use_rover_mask_as_label", False),
                split_file=_cfg_get(split_cfg, "split_file", None),
                images_dir=_cfg_get(split_cfg, "images_dir", None),
                masks_dir=_cfg_get(split_cfg, "masks_dir", None),
                transform=transform,
            )

        if name == "mars_bench":
            split_name = self.cfg.data.splits[split]
            return MarsBenchHFDataset(
                dataset_name=self.cfg.data.hf_dataset,
                config_name=_cfg_get(self.cfg.data, "hf_config", None),
                split=split_name,
                taxonomy=_cfg_get(self.cfg.data, "taxonomy", "core"),
                image_key=_cfg_get(self.cfg.data, "image_key", "image"),
                mask_key=_cfg_get(self.cfg.data, "mask_key", "mask"),
                transform=transform,
            )

        raise ValueError(f"Unknown dataset name: {name}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.loader.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.loader.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.data.loader.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
        )


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    return getattr(cfg, key, default)
