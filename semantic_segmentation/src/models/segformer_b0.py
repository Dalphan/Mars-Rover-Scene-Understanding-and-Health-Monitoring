from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerB0ForMars(nn.Module):
    """
    SegFormer-B0 wrapper for S5Mars / Mars-Bench semantic segmentation.

    Expected input:
        images: FloatTensor [B, 3, 512, 512]

    Expected output:
        logits: FloatTensor [B, 9, 512, 512]

    S5Mars label convention:
        target mask: LongTensor [B, 512, 512]
        values: 0..8
        ignore_index: 0
    """

    def __init__(
        self,
        pretrained_name: str,
        num_classes: int = 9,
        ignore_index: int = 0,
        log_shapes: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.log_shapes = log_shapes

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            semantic_loss_ignore_index=ignore_index,
            ignore_mismatched_sizes=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Shape trace for 512x512 input:

        images:
            [B, 3, 512, 512]

        SegFormer-B0 encoder stages internally:
            stage 1: [B,  32, 128, 128]   # stride 4
            stage 2: [B,  64,  64,  64]   # stride 8
            stage 3: [B, 160,  32,  32]   # stride 16
            stage 4: [B, 256,  16,  16]   # stride 32

        Hugging Face decode head output:
            raw_logits: [B, 9, 128, 128]

        Final output:
            logits: [B, 9, 512, 512]
        """
        if images.ndim != 4:
            raise ValueError(f"Expected images [B, 3, H, W], got {images.shape}")

        if images.shape[1] != 3:
            raise ValueError(f"Expected RGB images with 3 channels, got {images.shape}")

        # Input images: [B, 3, 512, 512].
        input_hw = images.shape[-2:]

        outputs = self.model(pixel_values=images)

        # SegFormer-B0 encoder stages for 512x512 input:
        # stage 1: [B, 32, 128, 128]
        # stage 2: [B, 64, 64, 64]
        # stage 3: [B, 160, 32, 32]
        # stage 4: [B, 256, 16, 16]
        # Decode head raw logits: [B, 9, 128, 128].
        raw_logits = outputs.logits

        # Upsampled logits: [B, 9, 512, 512].
        logits = F.interpolate(
            raw_logits,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )

        if self.log_shapes:
            print(f"raw_logits: shape={tuple(raw_logits.shape)}, dtype={raw_logits.dtype}")
            print(f"logits: shape={tuple(logits.shape)}, dtype={logits.dtype}")

        return logits

    def apply_freeze(self, freeze: str) -> None:
        """
        Apply freeze strategy.

        freeze:
            none:
                train full SegFormer.

            encoder:
                freeze MiT-B0 encoder/backbone.
                train decode head + classifier.

            classifier:
                freeze everything except final classifier.
        """
        freeze = freeze.lower()

        for param in self.parameters():
            param.requires_grad = True

        if freeze == "none":
            return

        if freeze == "encoder":
            for param in self.model.segformer.parameters():
                param.requires_grad = False

            for param in self.model.decode_head.parameters():
                param.requires_grad = True

            return

        if freeze == "classifier":
            for param in self.parameters():
                param.requires_grad = False

            for param in self.model.decode_head.classifier.parameters():
                param.requires_grad = True

            return

        raise ValueError(
            f"Unknown freeze mode '{freeze}'. "
            "Expected one of: none, encoder, classifier."
        )
