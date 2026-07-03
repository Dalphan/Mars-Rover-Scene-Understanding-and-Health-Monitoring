from __future__ import annotations

from src.models.segformer_b0 import SegFormerB0ForMars


def build_model(cfg):
    """
    Build the segmentation model from Hydra config.
    """
    if cfg.model.name == "segformer_b0":
        return SegFormerB0ForMars(
            pretrained_name=cfg.model.pretrained_name,
            num_classes=cfg.model.num_classes,
            ignore_index=cfg.model.ignore_index,
            log_shapes=cfg.model.log_shapes,
        )

    raise ValueError(f"Unknown model name: {cfg.model.name}")
