from types import SimpleNamespace

from src.models.factory import build_model


def _cfg(model):
    return SimpleNamespace(model=SimpleNamespace(**model))


def test_build_segformer_b0_from_config(monkeypatch):
    import src.models.segformer_b0 as segformer_module

    calls = {}

    class DummySegFormer:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(segformer_module, "SegFormerB0ForMars", DummySegFormer)

    model = build_model(
        _cfg(
            {
                "name": "segformer_b0",
                "pretrained_name": "nvidia/segformer-b0-finetuned-ade-512-512",
                "num_classes": 9,
                "ignore_index": 0,
                "log_shapes": False,
            }
        )
    )

    assert isinstance(model, DummySegFormer)
    assert calls == {
        "pretrained_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "num_classes": 9,
        "ignore_index": 0,
        "log_shapes": False,
    }


def test_build_default_smp_config(monkeypatch):
    import src.models.smp_model as smp_module

    calls = {}

    class DummySMP:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(smp_module, "SMPModelForMars", DummySMP)

    model = build_model(
        _cfg(
            {
                "name": "smp",
                "architecture": "unet",
                "encoder_name": "resnet34",
                "encoder_weights": "imagenet",
                "in_channels": 3,
                "num_classes": 9,
                "log_shapes": False,
            }
        )
    )

    assert isinstance(model, DummySMP)
    assert calls == {
        "architecture": "unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "num_classes": 9,
        "log_shapes": False,
    }
