import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation


def _tiny_segformer_config(num_classes: int) -> SegformerConfig:
    return SegformerConfig(
        num_labels=num_classes,
        num_channels=3,
        depths=[2, 2, 2, 2],
        hidden_sizes=[32, 64, 160, 256],
        decoder_hidden_size=128,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        drop_path_rate=0.0,
    )


class SegFormerModel(nn.Module):
    def __init__(self, checkpoint: str, num_classes: int, pretrained: bool):
        super().__init__()
        if pretrained and checkpoint:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    checkpoint,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
                return
            except Exception:
                # Kaggle notebooks often run offline, so fall back to a tiny local config.
                pass

        if checkpoint:
            try:
                config = SegformerConfig.from_pretrained(checkpoint)
                config.num_labels = num_classes
            except Exception:
                config = _tiny_segformer_config(num_classes)
        else:
            config = _tiny_segformer_config(num_classes)
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images):
        outputs = self.model(pixel_values=images)
        logits = outputs.logits
        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(
                logits, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits
