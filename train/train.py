import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from hydra import main
from omegaconf import OmegaConf

from mars_datasets.datamodule import SegmentationDataModule
from models.factory import build_model
from train.segmentation_module import SegmentationModule
from utils.runtime import configure_runtime
from utils.seed import set_seed


@main(version_base="1.3", config_path="../configs", config_name="config")
def run(cfg):
    configure_runtime(cfg)
    set_seed(cfg.seed)

    data_module = SegmentationDataModule(cfg)
    model = build_model(cfg)
    module = SegmentationModule(cfg, model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.output_dir,
        filename="epoch{epoch:03d}-miou{val_miou:.4f}",
        monitor="val_miou",
        mode="max",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks=[checkpoint_callback, lr_monitor],
    )
    trainer.fit(module, datamodule=data_module)


if __name__ == "__main__":
    run()
