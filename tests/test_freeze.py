import torch.nn as nn

from src.utils.model_stats import count_trainable_parameters


class FakeHFSegFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.segformer = nn.Linear(4, 4)
        self.decode_head = nn.Module()
        self.decode_head.proj = nn.Linear(4, 4)
        self.decode_head.classifier = nn.Linear(4, 2)


class FakeWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeHFSegFormer()

    def apply_freeze(self, freeze: str) -> None:
        freeze = freeze.lower()

        for param in self.parameters():
            param.requires_grad = True

        if freeze == "none":
            return

        if freeze == "encoder":
            for param in self.model.segformer.parameters():
                param.requires_grad = False
            return

        if freeze == "classifier":
            for param in self.parameters():
                param.requires_grad = False
            for param in self.model.decode_head.classifier.parameters():
                param.requires_grad = True
            return

        raise ValueError(f"Unknown freeze mode: {freeze}")


def _all_trainable(module):
    return all(param.requires_grad for param in module.parameters())


def _all_frozen(module):
    return all(not param.requires_grad for param in module.parameters())


def test_freeze_none_trains_everything():
    model = FakeWrapper()

    model.apply_freeze("none")

    assert _all_trainable(model)


def test_freeze_encoder_keeps_decode_head_trainable():
    model = FakeWrapper()

    model.apply_freeze("encoder")

    assert _all_frozen(model.model.segformer)
    assert _all_trainable(model.model.decode_head)


def test_freeze_classifier_trains_only_classifier():
    model = FakeWrapper()

    model.apply_freeze("classifier")

    assert _all_frozen(model.model.segformer)
    assert _all_frozen(model.model.decode_head.proj)
    assert _all_trainable(model.model.decode_head.classifier)


def test_count_trainable_parameters_counts_frozen_params():
    model = FakeWrapper()
    model.apply_freeze("classifier")

    stats = count_trainable_parameters(model)

    assert stats["total"] == stats["trainable"] + stats["frozen"]
    assert stats["trainable"] > 0
    assert stats["frozen"] > 0
