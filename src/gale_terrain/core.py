"""Pure geometry and validation helpers for the Gale terrain milestone."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CropBounds:
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, ...]) -> "CropBounds":
        if len(values) != 4:
            raise ValueError("Projected crop bounds must contain four values")
        bounds = cls(*(float(value) for value in values))
        if bounds.right <= bounds.left or bounds.top <= bounds.bottom:
            raise ValueError("Projected crop bounds must have positive width and height")
        return bounds

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.bottom + self.top) / 2.0)


def expected_resolution(bounds: CropBounds, shape: tuple[int, int] | list[int]) -> tuple[float, float]:
    if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
        raise ValueError("Raster shape must be [height, width] with positive values")
    height, width = int(shape[0]), int(shape[1])
    return (bounds.width / width, bounds.height / height)


def normalized_uv(x: float, y: float, bounds: CropBounds) -> tuple[float, float]:
    return ((x - bounds.left) / bounds.width, (y - bounds.bottom) / bounds.height)


def crs_parameters_equivalent(first: dict, second: dict, tolerance: float = 1e-9) -> bool:
    """Compare projection semantics while ignoring non-semantic WKT names."""

    text_keys = ("proj", "units")
    numeric_keys = ("lat_ts", "lat_0", "lon_0", "x_0", "y_0", "R")
    if any(first.get(key) != second.get(key) for key in text_keys):
        return False
    try:
        return all(abs(float(first[key]) - float(second[key])) <= tolerance for key in numeric_keys)
    except (KeyError, TypeError, ValueError):
        return False


def validate_grid_contract(config: dict) -> dict:
    crop = config["crop"]
    bounds = CropBounds.from_sequence(crop["projected_bounds_m"])
    dtm_shape = tuple(int(value) for value in crop["dtm_shape"])
    ortho_shape = tuple(int(value) for value in crop["ortho_shape"])
    heightfield_shape = tuple(int(value) for value in crop["heightfield_shape"])
    if dtm_shape[0] != dtm_shape[1] or ortho_shape[0] != ortho_shape[1]:
        raise ValueError("The baseline expects square DTM and ortho crops")
    if ortho_shape[0] % dtm_shape[0] != 0:
        raise ValueError("Ortho dimensions must be an integer multiple of DTM dimensions")
    if heightfield_shape != (dtm_shape[0] + 1, dtm_shape[1] + 1):
        raise ValueError("Heightfield point grid must have one more sample per axis than the DTM raster")
    vertical_exaggeration = float(crop.get("maximum_vertical_exaggeration", 1.0))
    if vertical_exaggeration != 1.0:
        raise ValueError("Vertical exaggeration is forbidden for the baseline")
    return {
        "bounds": bounds,
        "dtm_resolution_m": expected_resolution(bounds, dtm_shape),
        "ortho_resolution_m": expected_resolution(bounds, ortho_shape),
        "resolution_ratio": ortho_shape[0] // dtm_shape[0],
    }
