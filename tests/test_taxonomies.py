import numpy as np

from taxonomies import AI4MARS_TO_CORE, MARS_SEG_TO_CORE, S5MARS_TO_CORE, remap_mask


def test_ai4mars_core_mapping():
    mask = np.array([[0, 1, 2, 3, 255]], dtype=np.uint8)

    mapped = remap_mask(mask, AI4MARS_TO_CORE)

    assert mapped.tolist() == [[1, 0, 1, 2, 255]]


def test_mars_seg_core_mapping_ignores_non_core_scene_labels():
    mask = np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uint8)

    mapped = remap_mask(mask, MARS_SEG_TO_CORE)

    assert mapped.tolist() == [[255, 0, 1, 2, 255, 255, 255]]


def test_s5mars_core_mapping_ignores_rover_hazard_and_scene_labels():
    mask = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint8)

    mapped = remap_mask(mask, S5MARS_TO_CORE)

    assert mapped.tolist() == [[255, 0, 255, 255, 2, 255, 1, 255, 255]]
