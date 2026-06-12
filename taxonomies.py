from typing import Dict, Iterable

import numpy as np


IGNORE_INDEX = 255

CORE_CLASSES = {
    0: "bedrock",
    1: "loose_regolith",
    2: "rock",
}

CORE_VALID_CLASSES = list(CORE_CLASSES.keys())

AI4MARS_TO_CORE = {
    0: 1,
    1: 0,
    2: 1,
    3: 2,
    255: IGNORE_INDEX,
}

MARS_SEG_TO_CORE = {
    0: IGNORE_INDEX,
    1: 0,
    2: 1,
    3: 2,
    4: IGNORE_INDEX,
    5: IGNORE_INDEX,
    6: IGNORE_INDEX,
}

S5MARS_TO_CORE = {
    0: IGNORE_INDEX,
    1: 0,
    2: IGNORE_INDEX,
    3: IGNORE_INDEX,
    4: 2,
    5: IGNORE_INDEX,
    6: 1,
    7: IGNORE_INDEX,
    8: IGNORE_INDEX,
}


def mapping_for_dataset(dataset_name: str, taxonomy: str = "core") -> Dict[int, int]:
    if taxonomy != "core":
        raise ValueError(f"Only the core taxonomy is implemented in this tranche: {taxonomy}")

    normalized = dataset_name.lower()
    if normalized == "ai4mars":
        return AI4MARS_TO_CORE
    if "s5mars" in normalized:
        return S5MARS_TO_CORE
    if "mars_seg_mer" in normalized or "mars_seg_msl" in normalized:
        return MARS_SEG_TO_CORE
    raise ValueError(f"Unsupported dataset for core taxonomy mapping: {dataset_name}")


def remap_mask(mask: np.ndarray, mapping: Dict[int, int], ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    mask_np = np.asarray(mask)
    mapped = np.full(mask_np.shape, ignore_index, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        mapped[mask_np == source_id] = target_id
    return mapped


def unique_values(values: Iterable[np.ndarray]) -> list[int]:
    uniques: set[int] = set()
    for value in values:
        uniques.update(int(v) for v in np.unique(value))
    return sorted(uniques)
