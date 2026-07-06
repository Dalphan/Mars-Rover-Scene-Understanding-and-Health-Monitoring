from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetResNet34ForMars(nn.Module):
    """
    U-Net with ResNet34 encoder for S5Mars semantic segmentation.

    Input:
        images: FloatTensor [B, 3, 512, 512]

    Output:
        logits: FloatTensor [B, 9, 512, 512]

    S5Mars target:
        masks: LongTensor [B, 512, 512]
        values: 0..8
        ignore_index: 0
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_classes: int = 9,
        log_shapes: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.log_shapes = log_shapes

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Shape trace:

        images:
            [B, 3, 512, 512]

        U-Net encoder:
            progressively downsamples spatial resolution while increasing channels.

        U-Net decoder:
            progressively upsamples and combines skip connections.

        logits:
            [B, 9, 512, 512]
        """
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, 3, H, W], got {images.shape}")

        if images.shape[1] != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {images.shape}")

        logits = self.model(images)

        if logits.shape[-2:] != images.shape[-2:]:
            raise ValueError(
                f"Expected logits spatial shape {images.shape[-2:]}, "
                f"got {logits.shape[-2:]}"
            )

        if self.log_shapes:
            print(f"logits: shape={tuple(logits.shape)}, dtype={logits.dtype}")

        return logits

    def apply_freeze(self, freeze: str) -> None:
        """
        Apply freeze strategy.

        freeze:
            none:
                train the full U-Net.

            encoder:
                freeze ResNet34 encoder.
                train U-Net decoder + segmentation head.

            classifier:
                freeze everything except the final segmentation head.
        """
        freeze = freeze.lower()

        for param in self.parameters():
            param.requires_grad = True

        if freeze == "none":
            return

        if freeze == "encoder":
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            for param in self.model.decoder.parameters():
                param.requires_grad = True

            for param in self.model.segmentation_head.parameters():
                param.requires_grad = True

            return

        if freeze == "classifier":
            for param in self.parameters():
                param.requires_grad = False

            for param in self.model.segmentation_head.parameters():
                param.requires_grad = True

            return

        raise ValueError(
            f"Unknown freeze mode '{freeze}'. "
            "Expected one of: none, encoder, classifier."
        )
