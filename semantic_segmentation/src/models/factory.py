from __future__ import annotations


def build_model(cfg):
    """
    Build segmentation model from Hydra config.

    Supported models:
        - segformer_b0
        - smp
    """
    if cfg.model.name == "segformer_b0":
        from src.models.segformer_b0 import SegFormerB0ForMars

        return SegFormerB0ForMars(
            pretrained_name=cfg.model.pretrained_name,
            num_classes=cfg.model.num_classes,
            ignore_index=cfg.model.ignore_index,
            log_shapes=cfg.model.log_shapes,
        )

    if cfg.model.name == "smp":
        from src.models.smp_model import SMPModelForMars

        return SMPModelForMars(
            architecture=cfg.model.architecture,
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            num_classes=cfg.model.num_classes,
            log_shapes=cfg.model.log_shapes,
        )

    if cfg.model.name == "lcnet":
        from src.models.lcnet import LCNet

        return LCNet(
            variant=cfg.model.variant,
            in_channels=cfg.model.in_channels,
            base_channels=cfg.model.base_channels,
            partial_rate=cfg.model.partial_rate,
            stage1_blocks=cfg.model.stage1_blocks,
            stage2_blocks=cfg.model.stage2_blocks,
            num_classes=cfg.model.num_classes,
            pretrained=cfg.model.pretrained,
            log_shapes=cfg.model.log_shapes,
        )

    raise ValueError(
        f"Unknown model name '{cfg.model.name}'. "
        "Expected one of: segformer_b0, smp, lcnet."
    )
