"""Pure geometry and library definitions for irregular wheel perforations.

The module deliberately contains no Blender imports.  Blender consumes the
JSON-compatible profile returned by :func:`build_perforation_library` and is
responsible only for extrusion, Boolean operations and rendering.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping, Sequence


ORIENTATIONS = ("axial", "circumferential")
FAMILIES = ("jagged_slit", "branched_tear", "peeled_window")
SIZES = ("small", "medium")

# Long/short full dimensions as fractions of wheel diameter.  The intervals
# are intentionally broad enough that the seed changes silhouette as well as
# texture, while keeping the pilot comparable across variants.
SIZE_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "small": {
        "long": (0.055, 0.075),
        "short": (0.018, 0.030),
    },
    "medium": {
        "long": (0.085, 0.120),
        "short": (0.025, 0.045),
    },
}


def _cross(left: Sequence[float], right: Sequence[float]) -> float:
    return float(left[0]) * float(right[1]) - float(left[1]) * float(right[0])


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            float(points[index][0]) * float(points[(index + 1) % len(points)][1])
            - float(points[(index + 1) % len(points)][0]) * float(points[index][1])
            for index in range(len(points))
        )
    ) * 0.5


def _orientation(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return _cross((float(b[0]) - float(a[0]), float(b[1]) - float(a[1])),
                  (float(c[0]) - float(a[0]), float(c[1]) - float(a[1])))


def _on_segment(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> bool:
    return (
        min(float(a[0]), float(b[0])) - 1e-9 <= float(p[0]) <= max(float(a[0]), float(b[0])) + 1e-9
        and min(float(a[1]), float(b[1])) - 1e-9 <= float(p[1]) <= max(float(a[1]), float(b[1])) + 1e-9
    )


def _segments_intersect(
    a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]
) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    eps = 1e-9
    if abs(ab_c) <= eps and _on_segment(a, b, c):
        return True
    if abs(ab_d) <= eps and _on_segment(a, b, d):
        return True
    if abs(cd_a) <= eps and _on_segment(c, d, a):
        return True
    if abs(cd_b) <= eps and _on_segment(c, d, b):
        return True
    return (ab_c > eps) != (ab_d > eps) and (cd_a > eps) != (cd_b > eps)


def polygon_self_intersects(points: Sequence[Sequence[float]]) -> bool:
    count = len(points)
    if count < 4:
        return False
    for left in range(count):
        left_next = (left + 1) % count
        for right in range(left + 1, count):
            right_next = (right + 1) % count
            if left == right or left_next == right or right_next == left:
                continue
            if left == 0 and right_next == count - 1:
                continue
            if _segments_intersect(points[left], points[left_next], points[right], points[right_next]):
                return True
    return False


def _convex_hull(points: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return unique
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _orientation(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _orientation(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def polygon_quality(points: Sequence[Sequence[float]]) -> dict[str, float | bool]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    long_axis = max(width, height)
    short_axis = min(width, height)
    hull = _convex_hull(points)
    hull_area = polygon_area(hull)
    area = polygon_area(points)
    perimeter = sum(
        math.hypot(
            float(points[(index + 1) % len(points)][0]) - float(points[index][0]),
            float(points[(index + 1) % len(points)][1]) - float(points[index][1]),
        )
        for index in range(len(points))
    )
    return {
        "width": width,
        "height": height,
        "aspect_ratio": long_axis / max(short_axis, 1e-12),
        "area": area,
        "convexity": area / max(hull_area, 1e-12),
        "circularity": 4.0 * math.pi * area / max(perimeter * perimeter, 1e-12),
        "self_intersects": polygon_self_intersects(points),
    }


def _size_value(rng: random.Random, size: str, diameter: float) -> tuple[float, float]:
    try:
        bands = SIZE_RANGES[size]
    except KeyError as exc:
        raise ValueError(f"Unknown perforation size: {size!r}") from exc
    return (
        rng.uniform(*bands["long"]) * diameter,
        rng.uniform(*bands["short"]) * diameter,
    )


def _profile_points(
    family: str,
    orientation: str,
    size: str,
    diameter: float,
    seed: int,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown perforation family: {family!r}")
    if orientation not in ORIENTATIONS:
        raise ValueError(f"Unknown perforation orientation: {orientation!r}")
    count = {"jagged_slit": 16, "branched_tear": 20, "peeled_window": 18}[family]
    for attempt in range(64):
        rng = random.Random(int(seed) + attempt * 7919)
        long_size, short_size = _size_value(rng, size, diameter)
        points: list[tuple[float, float]] = []
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            jitter = rng.uniform(-0.10, 0.10)
            angle += jitter * (2.0 * math.pi / count)
            if family == "jagged_slit":
                factor = rng.uniform(0.72, 1.20)
                # A few asymmetric teeth make the contour visibly non-periodic.
                if index in {2, 7, 12}:
                    factor *= rng.uniform(1.16, 1.32)
            elif family == "branched_tear":
                factor = rng.uniform(0.68, 1.18)
                lobe = math.exp(-((angle - 0.48 * math.pi) / 0.28) ** 2)
                factor *= 1.0 + 0.72 * lobe
                if index in {4, 5, 6}:
                    factor *= rng.uniform(0.72, 0.88)
            else:
                factor = rng.uniform(0.70, 1.16)
                flap = math.exp(-((angle - 1.15 * math.pi) / 0.22) ** 2)
                factor *= 1.0 + 0.85 * flap
            x = 0.5 * long_size * factor * math.cos(angle)
            y = 0.5 * short_size * factor * math.sin(angle)
            points.append((x, y) if orientation == "axial" else (y, x))

        quality = polygon_quality(points)
        if (
            not bool(quality["self_intersects"])
            and 2.0 <= float(quality["aspect_ratio"]) <= 4.5
            and 0.60 <= float(quality["convexity"]) <= 0.90
            and float(quality["circularity"]) <= 0.78
        ):
            return points, quality
    raise ValueError(f"Could not generate a valid {family}/{orientation}/{size} profile")


def generate_perforation_profile(
    *,
    variant_id: str,
    family: str,
    orientation: str,
    size: str,
    diameter: float,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic, JSON-compatible irregular local profile."""

    if float(diameter) <= 0.0:
        raise ValueError("Wheel diameter must be positive")
    points, quality = _profile_points(family, orientation, size, float(diameter), int(seed))
    flap_count = {"jagged_slit": 1, "branched_tear": 2, "peeled_window": 3}[family]
    rng = random.Random(int(seed) ^ 0x5EED)
    folds = [
        {
            "edge_index": int(rng.randrange(len(points))),
            "angle_degrees": round(rng.uniform(12.0, 40.0), 4),
            "direction": "outward" if rng.random() >= 0.35 else "inward",
            "length_ratio": round(rng.uniform(0.12, 0.28), 4),
        }
        for _ in range(flap_count)
    ]
    return {
        "variant_id": str(variant_id),
        "family": family,
        "orientation": orientation,
        "size": size,
        "seed": int(seed),
        "diameter": float(diameter),
        "points_axial_tangent": [[round(float(x), 8), round(float(y), 8)] for x, y in points],
        "folds": folds,
        "open_area_fraction_before_folds": 1.0,
        "minimum_open_area_fraction_after_folds": 0.60,
        "quality": {key: (bool(value) if isinstance(value, bool) else float(value)) for key, value in quality.items()},
    }


def build_perforation_library(
    *, diameter: float, base_seed: int = 41000
) -> list[dict[str, Any]]:
    """Build the fixed 12-variant pilot in stable order."""

    variants: list[dict[str, Any]] = []
    counter = 0
    for orientation in ORIENTATIONS:
        for family in FAMILIES:
            for size in SIZES:
                counter += 1
                variants.append(
                    generate_perforation_profile(
                        variant_id=f"perforation_{counter:02d}_{orientation}_{family}_{size}",
                        family=family,
                        orientation=orientation,
                        size=size,
                        diameter=diameter,
                        seed=int(base_seed) + counter,
                    )
                )
    return variants


def validate_perforation_profile(profile: Mapping[str, Any]) -> list[str]:
    """Validate the pure profile contract before Blender is invoked."""

    errors: list[str] = []
    if profile.get("family") not in FAMILIES:
        errors.append("Unknown perforation family")
    if profile.get("orientation") not in ORIENTATIONS:
        errors.append("Unknown perforation orientation")
    if profile.get("size") not in SIZES:
        errors.append("Unknown perforation size")
    points = profile.get("points_axial_tangent")
    if not isinstance(points, Sequence) or len(points) < 8:
        errors.append("Perforation contour must contain at least eight points")
        return errors
    quality = polygon_quality(points)
    if bool(quality["self_intersects"]):
        errors.append("Perforation contour self-intersects")
    if not 2.0 <= float(quality["aspect_ratio"]) <= 4.5:
        errors.append("Perforation aspect ratio outside [2.0, 4.5]")
    if not 0.60 <= float(quality["convexity"]) <= 0.90:
        errors.append("Perforation convexity outside [0.60, 0.90]")
    if float(quality["circularity"]) > 0.78:
        errors.append("Perforation is too circular")
    if float(profile.get("minimum_open_area_fraction_after_folds", 0.0)) < 0.60:
        errors.append("Folded profile leaves less than 60% open area")
    return errors
