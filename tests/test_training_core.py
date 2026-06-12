import torch
from omegaconf import OmegaConf

from train.segmentation_module import SegmentationModule


class TinySegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, images):
        return self.conv(images)


def test_segmentation_module_training_step_accepts_sample_dict():
    cfg = OmegaConf.create(
        {
            "num_classes": 3,
            "ignore_index": 255,
            "model": {"loss": {"type": "ce", "dice_weight": 0.0}},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
            "logging": {"save_visuals": False},
            "paths": {"output_dir": "."},
            "data": {"normalize": {"mean": [0, 0, 0], "std": [1, 1, 1]}},
        }
    )
    module = SegmentationModule(cfg, TinySegmentationModel())
    batch = {
        "image": torch.rand(2, 3, 4, 4),
        "mask": torch.tensor(
            [
                [[0, 1, 2, 255], [0, 1, 2, 255], [0, 1, 2, 255], [0, 1, 2, 255]],
                [[2, 1, 0, 255], [2, 1, 0, 255], [2, 1, 0, 255], [2, 1, 0, 255]],
            ],
            dtype=torch.long,
        ),
        "dataset": ["fixture", "fixture"],
        "valid_classes": [[0, 1, 2], [0, 1, 2]],
        "image_id": ["a", "b"],
    }

    loss = module.training_step(batch, 0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
