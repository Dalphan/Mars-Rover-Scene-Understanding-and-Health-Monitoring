import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from src.models.smp_model import SMPModelForMars


SMP_CASES = [
    ("unet", "resnet34"),
    ("unet", "mobilenet_v2"),
    ("deeplabv3", "resnet34"),
    ("deeplabv3plus", "mobilenet_v2"),
]


@pytest.mark.parametrize("architecture,encoder_name", SMP_CASES)
def test_smp_model_forward_shape(architecture, encoder_name):
    model = SMPModelForMars(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        num_classes=9,
        log_shapes=False,
    )

    model.eval()

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 9, 512, 512)


@pytest.mark.parametrize("architecture,encoder_name", SMP_CASES)
def test_smp_freeze_encoder(architecture, encoder_name):
    model = SMPModelForMars(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        num_classes=9,
    )

    model.apply_freeze("encoder")

    assert all(not p.requires_grad for p in model.model.encoder.parameters())
    assert any(p.requires_grad for p in model.model.segmentation_head.parameters())

    if hasattr(model.model, "decoder"):
        assert any(p.requires_grad for p in model.model.decoder.parameters())


@pytest.mark.parametrize("architecture,encoder_name", SMP_CASES)
def test_smp_freeze_classifier(architecture, encoder_name):
    model = SMPModelForMars(
        architecture=architecture,
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        num_classes=9,
    )

    model.apply_freeze("classifier")

    assert all(not p.requires_grad for p in model.model.encoder.parameters())
    assert any(p.requires_grad for p in model.model.segmentation_head.parameters())

    if hasattr(model.model, "decoder"):
        assert all(not p.requires_grad for p in model.model.decoder.parameters())
