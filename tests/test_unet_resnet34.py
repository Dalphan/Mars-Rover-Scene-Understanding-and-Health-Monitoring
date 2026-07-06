import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from src.models.unet_resnet34 import UNetResNet34ForMars


def test_unet_resnet34_forward_shape():
    model = UNetResNet34ForMars(
        encoder_name="resnet34",
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


def test_unet_resnet34_freeze_encoder():
    model = UNetResNet34ForMars(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=9,
    )

    model.apply_freeze("encoder")

    assert all(not p.requires_grad for p in model.model.encoder.parameters())
    assert any(p.requires_grad for p in model.model.decoder.parameters())
    assert any(p.requires_grad for p in model.model.segmentation_head.parameters())


def test_unet_resnet34_freeze_classifier():
    model = UNetResNet34ForMars(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=9,
    )

    model.apply_freeze("classifier")

    assert all(not p.requires_grad for p in model.model.encoder.parameters())
    assert all(not p.requires_grad for p in model.model.decoder.parameters())
    assert any(p.requires_grad for p in model.model.segmentation_head.parameters())
