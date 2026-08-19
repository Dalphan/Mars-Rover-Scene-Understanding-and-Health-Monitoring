from __future__ import annotations

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SMPModelForMars(nn.Module):
    """
    Generic segmentation_models_pytorch wrapper for S5Mars.

    Supported architectures:
        - unet
        - deeplabv3
        - deeplabv3plus

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
        architecture: str,
        encoder_name: str,
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        num_classes: int = 9,
        log_shapes: bool = False,
    ) -> None:
        super().__init__()

        self.architecture = architecture.lower()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.log_shapes = log_shapes

        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        """
        Build SMP model with raw logits output.
        """
        common_kwargs = dict(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=None,
        )

        if self.architecture == "unet":
            return smp.Unet(**common_kwargs)

        if self.architecture == "deeplabv3":
            return smp.DeepLabV3(**common_kwargs)

        if self.architecture == "deeplabv3plus":
            return smp.DeepLabV3Plus(**common_kwargs)

        raise ValueError(
            f"Unknown SMP architecture '{self.architecture}'. "
            "Expected one of: unet, deeplabv3, deeplabv3plus."
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Shape trace:

        images:
            [B, 3, 512, 512]

        SMP model output:
            logits: [B, 9, H_out, W_out]

        Final output:
            logits: [B, 9, 512, 512]
        """
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, 3, H, W], got {images.shape}")

        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {images.shape[1]}"
            )

        logits = self.model(images)

        if logits.ndim != 4:
            raise ValueError(f"Expected logits [B, C, H, W], got {logits.shape}")

        if logits.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} output channels, got {logits.shape[1]}"
            )

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
                train the full model.

            encoder:
                freeze the pretrained encoder/backbone.
                train decoder if present + segmentation head.

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

            if hasattr(self.model, "decoder"):
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
