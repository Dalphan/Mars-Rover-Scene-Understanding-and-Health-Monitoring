import torch.nn as nn

from src.utils.freeze import apply_freeze_mode, count_trainable_parameters


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


def _all_trainable(module):
    return all(param.requires_grad for param in module.parameters())


def _all_frozen(module):
    return all(not param.requires_grad for param in module.parameters())


def test_freeze_none_trains_everything():
    model = FakeWrapper()

    apply_freeze_mode(model, "none")

    assert _all_trainable(model)


def test_freeze_encoder_keeps_decode_head_trainable():
    model = FakeWrapper()

    apply_freeze_mode(model, "encoder")

    assert _all_frozen(model.model.segformer)
    assert _all_trainable(model.model.decode_head)


def test_freeze_classifier_trains_only_classifier():
    model = FakeWrapper()

    apply_freeze_mode(model, "classifier")

    assert _all_frozen(model.model.segformer)
    assert _all_frozen(model.model.decode_head.proj)
    assert _all_trainable(model.model.decode_head.classifier)


def test_count_trainable_parameters_counts_frozen_params():
    model = FakeWrapper()
    apply_freeze_mode(model, "classifier")

    stats = count_trainable_parameters(model)

    assert stats["total"] == stats["trainable"] + stats["frozen"]
    assert stats["trainable"] > 0
    assert stats["frozen"] > 0
