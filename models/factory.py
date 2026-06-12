from models.deeplabv3plus import DeepLabV3PlusModel
from models.segformer import SegFormerModel


def build_model(cfg):
    name = cfg.model.name
    if name == "segformer_b0":
        return SegFormerModel(
            checkpoint=cfg.model.checkpoint,
            num_classes=cfg.num_classes,
            pretrained=cfg.model.pretrained,
        )
    if name == "deeplabv3plus_mnv3":
        return DeepLabV3PlusModel(
            encoder_name=cfg.model.encoder_name,
            num_classes=cfg.num_classes,
            pretrained=cfg.model.pretrained,
        )
    raise ValueError(f"Unknown model name: {name}")
