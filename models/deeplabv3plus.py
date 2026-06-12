from torch import nn
import segmentation_models_pytorch as smp


class DeepLabV3PlusModel(nn.Module):
    def __init__(self, encoder_name: str, num_classes: int, pretrained: bool):
        super().__init__()
        encoder_weights = "imagenet" if pretrained else None
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, images):
        return self.model(images)
