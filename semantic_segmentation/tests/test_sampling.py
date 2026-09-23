import numpy as np

from src.data.sampling import compute_image_level_sampling_weights


CLASS_NAMES = {0: "Background", 1: "Rock", 2: "Rover"}


def test_inverse_rarity_weights_rare_images_more_without_changing_mean():
    audit = compute_image_level_sampling_weights(
        class_labels=[
            ["Background", "Rock"],
            [0, 1],
            ["Background", "Rock"],
            ["Background", "Rover"],
        ],
        class_names=CLASS_NAMES,
        num_classes=3,
        power=0.5,
        max_weight=3.0,
    )

    assert np.isclose(audit.normalized_sample_weights.mean(), 1.0)
    assert audit.normalized_sample_weights[3] > audit.normalized_sample_weights[0]
    assert audit.class_image_counts.tolist() == [4, 3, 1]


def test_excluding_background_uses_only_eligible_classes():
    audit = compute_image_level_sampling_weights(
        class_labels=[[0, 1], [0, 1], [0, 2]],
        class_names=CLASS_NAMES,
        num_classes=3,
        power=1.0,
        max_weight=2.0,
        excluded_class_ids=[0],
    )

    assert audit.class_image_counts.tolist() == [0, 2, 1]
    assert audit.normalized_sample_weights[2] > audit.normalized_sample_weights[0]
