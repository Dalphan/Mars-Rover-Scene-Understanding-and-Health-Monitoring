from __future__ import annotations


def apply_freeze_mode(model, freeze: str) -> None:
    """
    Apply freeze strategy to SegFormer.

    Args:
        model:
            SegFormerB0ForMars wrapper.

        freeze:
            - "none": train all parameters
            - "encoder": freeze SegFormer encoder/backbone
            - "classifier": train only final classifier

    Expected wrapper structure:
        model.model.segformer
        model.model.decode_head
        model.model.decode_head.classifier
    """
    freeze = freeze.lower()

    for param in model.parameters():
        param.requires_grad = True

    if freeze == "none":
        return

    if freeze == "encoder":
        for param in model.model.segformer.parameters():
            param.requires_grad = False

        for param in model.model.decode_head.parameters():
            param.requires_grad = True

        return

    if freeze == "classifier":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.model.decode_head.classifier.parameters():
            param.requires_grad = True

        return

    raise ValueError(
        f"Unknown freeze mode: {freeze}. "
        "Expected one of: none, encoder, classifier."
    )


def count_trainable_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
