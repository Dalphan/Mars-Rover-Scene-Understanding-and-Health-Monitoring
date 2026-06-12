import pytorch_lightning as pl
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from utils.losses import build_loss
from utils.visualization import save_prediction_grid


class SegmentationModule(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = build_loss(cfg, num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

        self.train_iou = MulticlassJaccardIndex(
            num_classes=cfg.num_classes, ignore_index=cfg.ignore_index
        )
        self.val_iou = MulticlassJaccardIndex(
            num_classes=cfg.num_classes, ignore_index=cfg.ignore_index
        )
        self.val_iou_per_class = MulticlassJaccardIndex(
            num_classes=cfg.num_classes, ignore_index=cfg.ignore_index, average=None
        )
        self.val_acc = MulticlassAccuracy(
            num_classes=cfg.num_classes, ignore_index=cfg.ignore_index
        )
        self.test_iou = MulticlassJaccardIndex(
            num_classes=cfg.num_classes, ignore_index=cfg.ignore_index
        )
        self.test_acc = MulticlassAccuracy(
            num_classes=cfg.num_classes, ignore_index=cfg.ignore_index
        )

        self._val_example = None

    def forward(self, images):
        return self.model(images)

    def _shared_step(self, batch):
        images = batch["image"]
        masks = batch["mask"]
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks, images

    def training_step(self, batch, batch_idx):
        loss, preds, masks, _ = self._shared_step(batch)
        self.train_iou(preds, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_miou", self.train_iou, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks, images = self._shared_step(batch)
        self.val_iou(preds, masks)
        self.val_iou_per_class(preds, masks)
        self.val_acc(preds, masks)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_miou", self.val_iou, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._val_example = (
                images.detach(),
                masks.detach(),
                preds.detach(),
            )

    def on_validation_epoch_end(self):
        per_class = self.val_iou_per_class.compute()
        for idx, value in enumerate(per_class):
            self.log(f"val_iou_class_{idx}", value, on_epoch=True)
        self.val_iou_per_class.reset()

        if self.cfg.logging.save_visuals and self._val_example is not None:
            images, masks, preds = self._val_example
            save_prediction_grid(
                images,
                masks,
                preds,
                output_dir=self.cfg.paths.output_dir,
                epoch=self.current_epoch,
                max_samples=self.cfg.logging.visuals_max_samples,
                mean=self.cfg.data.normalize.mean,
                std=self.cfg.data.normalize.std,
                ignore_index=self.cfg.ignore_index,
                num_classes=self.cfg.num_classes,
            )

    def test_step(self, batch, batch_idx):
        loss, preds, masks, _ = self._shared_step(batch)
        self.test_iou(preds, masks)
        self.test_acc(preds, masks)
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_miou", self.test_iou, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optimizer
