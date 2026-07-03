import torch

from src.models.segformer_b0 import SegFormerB0ForMars


def test_segformer_b0_forward_shape():
    model = SegFormerB0ForMars(
        pretrained_name="nvidia/segformer-b0-finetuned-ade-512-512",
        num_classes=9,
        ignore_index=0,
        log_shapes=False,
    )

    model.eval()

    x = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 9, 512, 512)
