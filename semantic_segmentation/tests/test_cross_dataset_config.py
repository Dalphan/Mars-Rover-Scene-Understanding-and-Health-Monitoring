from types import SimpleNamespace

import pytest

from src.data.dataloaders import validate_cross_dataset_configuration


MARSSEG_CLASSES = {0: "Background", 1: "Rock"}


def _cfg(dataset_name, enabled):
    return SimpleNamespace(
        dataset_name=dataset_name,
        run_cross_dataset_evaluation=enabled,
        datasets={
            "s5mars": SimpleNamespace(
                family="s5mars",
                cross_dataset_name=None,
                class_names={0: "Background", 1: "Rover"},
                num_classes=2,
            ),
            "marsseg_msl": SimpleNamespace(
                family="marsseg",
                cross_dataset_name="marsseg_mer",
                class_names=MARSSEG_CLASSES,
                num_classes=2,
            ),
            "marsseg_mer": SimpleNamespace(
                family="marsseg",
                cross_dataset_name="marsseg_msl",
                class_names=MARSSEG_CLASSES,
                num_classes=2,
            ),
        },
    )


def test_msl_cross_evaluation_resolves_mer():
    target_name, _ = validate_cross_dataset_configuration(
        _cfg("marsseg_msl", True)
    )
    assert target_name == "marsseg_mer"


def test_s5mars_cross_evaluation_is_rejected():
    with pytest.raises(ValueError, match="only for marsseg"):
        validate_cross_dataset_configuration(_cfg("s5mars", True))


def test_disabled_cross_evaluation_has_no_target():
    assert validate_cross_dataset_configuration(_cfg("s5mars", False)) is None
