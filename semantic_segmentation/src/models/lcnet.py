"""Project-native LCNet reimplementation for semantic segmentation.

Based on Shi et al., "Lightweight Context-Aware Network Using Partial-Channel
Transformation for Real-Time Semantic Segmentation",
https://doi.org/10.1109/TITS.2023.3348631. The authors' reference repository is
https://github.com/lztjy/LCNet. This is an independent, project-compatible
reimplementation and does not include pretrained weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANT_DILATIONS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "lcnet3_7": ((2, 2, 2), (4, 4, 8, 8, 16, 16, 32)),
    "lcnet3_11": ((2, 2, 2), (4, 4, 8, 8, 16, 16, 32, 32, 32, 32, 32)),
}


class ConvBNSelu(nn.Module):
    """Convolution optionally followed by BatchNorm(eps=1e-3) and SELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        norm_activation: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
        ]
        if norm_activation:
            layers.extend(
                [nn.BatchNorm2d(out_channels, eps=1e-3), nn.SELU(inplace=True)]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels <= in_channels:
            raise ValueError("LCNet downsampling requires out_channels > in_channels")
        self.convolution = ConvBNSelu(
            in_channels, out_channels - in_channels, 3, stride=2, padding=1
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.norm_activation = nn.Sequential(
            nn.BatchNorm2d(out_channels, eps=1e-3), nn.SELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_activation(torch.cat((self.convolution(x), self.pool(x)), dim=1))


class ThreeBranchContextAggregation(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.base = ConvBNSelu(channels, channels, 3, padding=1, norm_activation=True)
        self.local_depthwise = ConvBNSelu(
            channels, channels, 3, padding=1, groups=channels, norm_activation=True
        )
        self.dilated_depthwise = ConvBNSelu(
            channels,
            channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            norm_activation=True,
        )
        self.norm_activation = nn.Sequential(
            nn.BatchNorm2d(channels, eps=1e-3), nn.SELU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        combined = base + self.local_depthwise(base) + self.dilated_depthwise(base)
        return self.norm_activation(combined)


class PartialChannelTransformation(nn.Module):
    def __init__(self, channels: int, dilation: int, partial_rate: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < partial_rate < 1.0:
            raise ValueError(f"partial_rate must be in (0, 1), got {partial_rate}")
        self.identity_channels = round(channels * (1.0 - partial_rate))
        transformed_channels = channels - self.identity_channels
        self.tca = ThreeBranchContextAggregation(transformed_channels, dilation)
        self.pointwise = ConvBNSelu(
            channels, channels, 1, norm_activation=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity, transformed = torch.split(
            x, (self.identity_channels, x.shape[1] - self.identity_channels), dim=1
        )
        transformed = self.tca(transformed)
        return self.pointwise(torch.cat((identity, transformed), dim=1))


class DualAttentionGuidedDecoder(nn.Module):
    def __init__(self, deep_channels: int, shallow_channels: int, num_classes: int) -> None:
        super().__init__()
        self.deep_projection = ConvBNSelu(
            deep_channels, shallow_channels, 1, norm_activation=True
        )
        self.negative_projection = ConvBNSelu(
            shallow_channels, shallow_channels, 1, norm_activation=True
        )
        self.depthwise = ConvBNSelu(
            shallow_channels,
            shallow_channels,
            3,
            padding=1,
            groups=shallow_channels,
            norm_activation=True,
        )
        self.classifier = ConvBNSelu(
            shallow_channels, num_classes, 1, norm_activation=True
        )

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        spatial_map = torch.sigmoid(shallow)
        projected = self.deep_projection(deep)
        projected_map = torch.sigmoid(projected)
        negative_map = self.negative_projection(-projected_map)
        reverse_guided = negative_map * projected_map + projected
        reverse_guided = F.interpolate(
            reverse_guided,
            size=spatial_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.depthwise(spatial_map * reverse_guided)
        return self.classifier(fused)


class LCNet(nn.Module):
    """LCNet3_7 or LCNet3_11 returning full-resolution raw logits."""

    def __init__(
        self,
        num_classes: int,
        variant: str = "lcnet3_7",
        in_channels: int = 3,
        base_channels: int = 32,
        partial_rate: float = 0.5,
        stage1_blocks: int | None = None,
        stage2_blocks: int | None = None,
        pretrained: bool = False,
        log_shapes: bool = False,
    ) -> None:
        super().__init__()
        variant = variant.lower()
        if variant not in VARIANT_DILATIONS:
            choices = ", ".join(VARIANT_DILATIONS)
            raise ValueError(f"Unsupported LCNet variant '{variant}'. Expected one of: {choices}.")
        if in_channels != 3:
            raise ValueError(f"LCNet requires RGB input (in_channels=3), got {in_channels}")
        if pretrained:
            raise ValueError("Pretrained weights are not available for this LCNet reimplementation")

        stage1_dilations, stage2_dilations = VARIANT_DILATIONS[variant]
        expected_blocks = (len(stage1_dilations), len(stage2_dilations))
        configured_blocks = (
            expected_blocks[0] if stage1_blocks is None else stage1_blocks,
            expected_blocks[1] if stage2_blocks is None else stage2_blocks,
        )
        if configured_blocks != expected_blocks:
            raise ValueError(
                f"Variant {variant} requires stage blocks {expected_blocks}, got {configured_blocks}"
            )

        self.variant = variant
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.log_shapes = log_shapes
        self.initialization = nn.Sequential(
            ConvBNSelu(in_channels, base_channels, 3, stride=2, padding=1, norm_activation=True),
            ConvBNSelu(base_channels, base_channels, 3, padding=1, norm_activation=True),
            ConvBNSelu(base_channels, base_channels, 3, padding=1, norm_activation=True),
        )
        self.stage1 = self._make_stage(
            base_channels, base_channels * 2, stage1_dilations, partial_rate
        )
        self.stage2 = self._make_stage(
            base_channels * 2, base_channels * 4, stage2_dilations, partial_rate
        )
        self.decoder = DualAttentionGuidedDecoder(
            base_channels * 4, base_channels * 2, num_classes
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        dilations: Sequence[int],
        partial_rate: float,
    ) -> nn.Sequential:
        return nn.Sequential(
            DownsamplingBlock(in_channels, out_channels),
            *(PartialChannelTransformation(out_channels, d, partial_rate) for d in dilations),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected images [B, {self.in_channels}, H, W], got {tuple(images.shape)}"
            )
        input_size = images.shape[-2:]
        shallow = self.stage1(self.initialization(images))
        deep = self.stage2(shallow)
        scores = self.decoder(shallow, deep)
        logits = F.interpolate(scores, size=input_size, mode="bilinear", align_corners=False)
        if self.log_shapes:
            print(f"logits: shape={tuple(logits.shape)}, dtype={logits.dtype}")
        return logits

    def apply_freeze(self, freeze: str) -> None:
        freeze = freeze.lower()
        for parameter in self.parameters():
            parameter.requires_grad = True
        if freeze == "none":
            return
        if freeze == "encoder":
            for module in (self.initialization, self.stage1, self.stage2):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            return
        if freeze == "classifier":
            for parameter in self.parameters():
                parameter.requires_grad = False
            for parameter in self.decoder.classifier.parameters():
                parameter.requires_grad = True
            return
        raise ValueError(f"Unknown freeze mode '{freeze}'. Expected one of: none, encoder, classifier.")
