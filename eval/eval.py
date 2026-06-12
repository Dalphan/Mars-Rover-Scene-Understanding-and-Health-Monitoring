import pytorch_lightning as pl
import torch
from hydra import main
from omegaconf import OmegaConf

from datasets.datamodule import SegmentationDataModule
from models.factory import build_model
from train.segmentation_module import SegmentationModule
from utils.seed import set_seed


@main(version_base="1.3", config_path="../configs", config_name="config")
def run(cfg):
    if cfg.ckpt_path is None:
        raise ValueError("Set ckpt_path to a checkpoint file before running evaluation.")

    set_seed(cfg.seed)

    data_module = SegmentationDataModule(cfg)
    model = build_model(cfg)
    module = SegmentationModule(cfg, model)

    checkpoint = torch.load(cfg.ckpt_path, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"], strict=True)

    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer.test(module, datamodule=data_module)


if __name__ == "__main__":
    run()
