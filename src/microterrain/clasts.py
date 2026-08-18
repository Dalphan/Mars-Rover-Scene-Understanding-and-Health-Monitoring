"""Pure deterministic Level-2 clast scattering."""

from __future__ import annotations

import hashlib
import math

from .core import derive_seed, validate_level1_config


FAMILIES = ("fine_grains", "fragments", "coarse_clasts")


def _burial_range(settings: dict, clasts: dict) -> tuple[float, float]:
    values = settings.get("buried_fraction")
    if values is None:
        values = [clasts["buried_fraction_min"], clasts["buried_fraction_max"]]
    low, high = map(float, values)
    if not 0 <= low < high <= 0.8:
        raise ValueError("Burial range must be ordered and remain at most 80 percent")
    return low, high


def _variant_counts(clasts: dict) -> dict[str, int]:
    configured = clasts.get("library_variants")
    if configured is None:
        value = int(clasts.get("library_variants_per_family", 0))
        configured = {family: value for family in FAMILIES}
    counts = {family: int(configured.get(family, 0)) for family in FAMILIES}
    if any(value < 3 for value in counts.values()):
        raise ValueError("At least three source variants per clast family are required")
    return counts


def _allocate_class_counts(total: int, classes: list[dict]) -> list[int]:
    raw = [total * float(entry["fraction"]) for entry in classes]
    counts = [int(math.floor(value)) for value in raw]
    order = sorted(range(len(classes)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def validate_level2_config(config: dict) -> dict:
    level1 = validate_level1_config(config)
    micro = config["microterrain"]
    clasts = micro.get("clasts", {})
    if int(micro.get("level", 0)) < 2 or not clasts.get("enabled"):
        raise ValueError("Microterrain Level 2 clasts must be enabled")
    variants = _variant_counts(clasts)
    area = float(micro["patch_size_m"]) ** 2
    counts = {}
    burial_ranges = {}
    for family in FAMILIES:
        settings = clasts[family]
        density = float(settings["density_m2"])
        minimum = float(settings["size_min_m"])
        maximum = float(settings["size_max_m"])
        tilt = float(settings["maximum_tilt_deg"])
        if density < 0 or not 0 < minimum < maximum or not 0 <= tilt <= 90:
            raise ValueError(f"Invalid Level-2 settings for {family}")
        counts[family] = int(round(area * density))
        if family != "coarse_clasts" or not settings.get("size_classes"):
            burial_ranges[family] = list(_burial_range(settings, clasts))
    total = sum(counts.values())
    if total <= 0 or total > int(clasts["maximum_instances"]):
        raise ValueError("Configured clast instance count is outside the performance gate")

    coarse = clasts["coarse_clasts"]
    coarse_classes = coarse.get("size_classes", [])
    class_counts = {}
    if coarse_classes:
        if len(coarse_classes) < 2:
            raise ValueError("Coarse clasts require at least two size classes")
        if abs(sum(float(entry["fraction"]) for entry in coarse_classes) - 1.0) > 1e-6:
            raise ValueError("Coarse clast size-class fractions must sum to one")
        names = [str(entry["name"]) for entry in coarse_classes]
        if len(names) != len(set(names)):
            raise ValueError("Coarse clast size-class names must be unique")
        allocated = _allocate_class_counts(counts["coarse_clasts"], coarse_classes)
        for entry, allocated_count in zip(coarse_classes, allocated):
            minimum = float(entry["size_min_m"])
            maximum = float(entry["size_max_m"])
            if not float(coarse["size_min_m"]) <= minimum < maximum <= float(coarse["size_max_m"]):
                raise ValueError(f"Invalid coarse size class {entry['name']}")
            burial_ranges[f"coarse_clasts:{entry['name']}"] = list(_burial_range(entry, clasts))
            class_counts[str(entry["name"])] = allocated_count
        spacing_factor = float(coarse.get("minimum_center_spacing_factor", 0.0))
        if not 0 <= spacing_factor <= 2:
            raise ValueError("Coarse minimum-center-spacing factor is invalid")
    else:
        burial_ranges["coarse_clasts"] = list(_burial_range(coarse, clasts))

    for key in ("clustering_strength", "microrelief_correlation"):
        if not 0 <= float(clasts[key]) <= 1:
            raise ValueError(f"{key} must lie in [0, 1]")
    margin = float(clasts["edge_margin_m"])
    if not 0 <= margin < float(micro["patch_size_m"]) / 4:
        raise ValueError("Clast edge margin is invalid")
    return {
        **level1,
        "area_m2": area,
        "counts": counts,
        "total_instances": total,
        "variants_per_family": variants,
        "burial_ranges": burial_ranges,
        "coarse_class_counts": class_counts,
    }


def _normalise_01(values, np):
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-12:
        return np.full_like(values, 0.5, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def _cluster_field(grid_x, grid_y, wavelength: float, seed: int, np):
    rng = np.random.default_rng(seed)
    field = np.zeros_like(grid_x, dtype=np.float32)
    for _ in range(7):
        angle = rng.uniform(0.0, math.pi)
        local_wavelength = wavelength * rng.uniform(0.68, 1.55)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        projection = math.cos(angle) * grid_x + math.sin(angle) * grid_y
        field += rng.uniform(0.55, 1.0) * np.cos(2.0 * math.pi * projection / local_wavelength + phase)
    return _normalise_01(field, np)


def _bilinear(axis_x, axis_y, values, x, y, np):
    columns = np.interp(x, axis_x, np.arange(len(axis_x)))
    rows = np.interp(y, axis_y, np.arange(len(axis_y)))
    c0 = np.floor(columns).astype(np.int32)
    r0 = np.floor(rows).astype(np.int32)
    c1 = np.minimum(c0 + 1, len(axis_x) - 1)
    r1 = np.minimum(r0 + 1, len(axis_y) - 1)
    wc, wr = columns - c0, rows - r0
    return values[r0, c0] * (1 - wc) * (1 - wr) + values[r0, c1] * wc * (1 - wr) + values[r1, c0] * (1 - wc) * wr + values[r1, c1] * wc * wr


def _sizes_and_classes(family: str, count: int, settings: dict, clasts: dict, rng, np):
    classes = settings.get("size_classes", []) if family == "coarse_clasts" else []
    if not classes:
        size = np.exp(rng.uniform(math.log(float(settings["size_min_m"])), math.log(float(settings["size_max_m"])), count))
        burial_low, burial_high = _burial_range(settings, clasts)
        burial = rng.uniform(burial_low, burial_high, count)
        return size, burial, np.full(count, -1, dtype=np.int32)

    allocated = _allocate_class_counts(count, classes)
    size_parts, burial_parts, class_parts = [], [], []
    for class_index, (entry, class_count) in enumerate(zip(classes, allocated)):
        # Rare classes cannot rely on unconstrained random draws: with only a
        # few instances they may never represent the upper half of the range.
        # Jittered logarithmic strata preserve variability while guaranteeing
        # coverage of the configured class interval.
        quantiles = (np.arange(class_count, dtype=np.float64) + rng.random(class_count)) / class_count
        rng.shuffle(quantiles)
        logarithmic_size = math.log(float(entry["size_min_m"])) + quantiles * (
            math.log(float(entry["size_max_m"])) - math.log(float(entry["size_min_m"]))
        )
        size_parts.append(np.exp(logarithmic_size))
        burial_low, burial_high = _burial_range(entry, clasts)
        burial_parts.append(rng.uniform(burial_low, burial_high, class_count))
        class_parts.append(np.full(class_count, class_index, dtype=np.int32))
    size = np.concatenate(size_parts)
    burial = np.concatenate(burial_parts)
    class_index = np.concatenate(class_parts)
    permutation = rng.permutation(count)
    return size[permutation], burial[permutation], class_index[permutation]


def _candidate_batch(rng, number, probability, valid_indices, x_axis, y_axis, np):
    selected = rng.choice(valid_indices, size=number, replace=True, p=probability)
    rows, columns = np.divmod(selected, len(x_axis))
    spacing_x = float(x_axis[1] - x_axis[0])
    spacing_y = float(y_axis[1] - y_axis[0])
    x = x_axis[columns].astype(np.float64) + rng.uniform(-0.45 * spacing_x, 0.45 * spacing_x, number)
    y = y_axis[rows].astype(np.float64) + rng.uniform(-0.45 * spacing_y, 0.45 * spacing_y, number)
    return x, y


def _position_mask(x, y, radius, x_axis, y_axis, edge_margin, exclusion_zones, np):
    mask = (
        (x >= float(x_axis[0]) + edge_margin + radius)
        & (x <= float(x_axis[-1]) - edge_margin - radius)
        & (y >= float(y_axis[0]) + edge_margin + radius)
        & (y <= float(y_axis[-1]) - edge_margin - radius)
    )
    for center_x, center_y, radius_x, radius_y in exclusion_zones:
        mask &= ((x - float(center_x)) / (float(radius_x) + radius)) ** 2 + ((y - float(center_y)) / (float(radius_y) + radius)) ** 2 >= 1.0
    return mask


def _sample_positions(family: str, size, probability, valid_indices, x_axis, y_axis, edge_margin, exclusion_zones, settings, rng, np):
    count = len(size)
    radius = 0.5 * size
    x = np.empty(count, dtype=np.float64)
    y = np.empty(count, dtype=np.float64)
    spacing_factor = float(settings.get("minimum_center_spacing_factor", 0.0)) if family == "coarse_clasts" else 0.0
    if spacing_factor > 0:
        accepted = []
        for index in np.argsort(-size):
            found = False
            for _ in range(32):
                candidate_x, candidate_y = _candidate_batch(rng, 128, probability, valid_indices, x_axis, y_axis, np)
                candidate_radius = np.full(128, radius[index], dtype=np.float64)
                valid = _position_mask(candidate_x, candidate_y, candidate_radius, x_axis, y_axis, edge_margin, exclusion_zones, np)
                if accepted:
                    accepted_indices = np.asarray(accepted, dtype=np.int32)
                    dx = candidate_x[:, None] - x[accepted_indices][None, :]
                    dy = candidate_y[:, None] - y[accepted_indices][None, :]
                    required = spacing_factor * (size[index] + size[accepted_indices])
                    valid &= np.all(dx * dx + dy * dy >= required[None, :] ** 2, axis=1)
                choices = np.flatnonzero(valid)
                if len(choices):
                    chosen = int(choices[0])
                    x[index], y[index] = candidate_x[chosen], candidate_y[chosen]
                    accepted.append(int(index))
                    found = True
                    break
            if not found:
                raise ValueError("Unable to place coarse clasts without overlaps, wheel intrusion, or boundary clipping")
        return x, y

    remaining = np.arange(count, dtype=np.int32)
    for _ in range(64):
        if not len(remaining):
            return x, y
        candidate_x, candidate_y = _candidate_batch(rng, len(remaining), probability, valid_indices, x_axis, y_axis, np)
        valid = _position_mask(candidate_x, candidate_y, radius[remaining], x_axis, y_axis, edge_margin, exclusion_zones, np)
        accepted = remaining[valid]
        x[accepted], y[accepted] = candidate_x[valid], candidate_y[valid]
        remaining = remaining[~valid]
    raise ValueError(f"Unable to place all {family} instances inside the safe region")


def _family_scatter(family: str, count: int, probability, valid_indices, x_axis, y_axis, surface_z, gradients, settings: dict, clasts: dict, variants: int, edge_margin: float, exclusion_zones, seed: int, np):
    rng = np.random.default_rng(seed)
    size, burial, class_index = _sizes_and_classes(family, count, settings, clasts, rng, np)
    x, y = _sample_positions(family, size, probability, valid_indices, x_axis, y_axis, edge_margin, exclusion_zones, settings, rng, np)
    z_surface = _bilinear(x_axis, y_axis, surface_z, x, y, np)
    gradient_y, gradient_x = gradients
    sampled_dx = _bilinear(x_axis, y_axis, gradient_x, x, y, np)
    sampled_dy = _bilinear(x_axis, y_axis, gradient_y, x, y, np)
    normal_z = 1.0 / np.sqrt(1.0 + sampled_dx * sampled_dx + sampled_dy * sampled_dy)
    normal_x, normal_y = -sampled_dx * normal_z, -sampled_dy * normal_z

    if family == "fine_grains":
        raw_aspect = rng.uniform([0.72, 0.72, 0.68], [1.18, 1.18, 1.12], size=(count, 3))
    elif family == "fragments":
        raw_aspect = rng.uniform([0.48, 0.55, 0.40], [1.45, 1.35, 1.10], size=(count, 3))
    else:
        raw_aspect = rng.uniform([0.58, 0.62, 0.48], [1.55, 1.40, 1.18], size=(count, 3))
    scale = size[:, None] * raw_aspect / raw_aspect.max(axis=1)[:, None]

    maximum_tilt = math.radians(float(settings["maximum_tilt_deg"]))
    random_tilt_x = rng.normal(0.0, maximum_tilt / 2.5, count).clip(-maximum_tilt, maximum_tilt)
    random_tilt_y = rng.normal(0.0, maximum_tilt / 2.5, count).clip(-maximum_tilt, maximum_tilt)
    rotations = np.column_stack((np.arctan2(normal_y, normal_z) + random_tilt_x, -np.arctan2(normal_x, normal_z) + random_tilt_y, rng.uniform(0.0, 2.0 * math.pi, count)))

    cx, sx = np.cos(rotations[:, 0]), np.sin(rotations[:, 0])
    cy, sy = np.cos(rotations[:, 1]), np.sin(rotations[:, 1])
    vertical_radius = 0.5 * np.sqrt((sy * scale[:, 0]) ** 2 + (sx * cy * scale[:, 1]) ** 2 + (cx * cy * scale[:, 2]) ** 2)
    z = z_surface + (1.0 - 2.0 * burial) * vertical_radius
    positions = np.column_stack((x, y, z)).astype(np.float32)
    rotations = rotations.astype(np.float32)
    scale = scale.astype(np.float32)
    size = size.astype(np.float32)
    burial = burial.astype(np.float32)
    prototype_index = rng.integers(0, variants, count, dtype=np.int32)
    signature = hashlib.sha256()
    for values in (positions, rotations, scale, size, burial, class_index, prototype_index):
        signature.update(values.tobytes(order="C"))
    return {
        "positions": positions,
        "rotations": rotations,
        "scales": scale,
        "characteristic_size_m": size,
        "burial": burial,
        "size_class_index": class_index,
        "prototype_index": prototype_index,
        "signature_sha256": signature.hexdigest(),
    }


def generate_clast_scatter(x_coordinates, y_coordinates, surface_z, displacement, config: dict, np, exclusion_zones=()):
    contract = validate_level2_config(config)
    micro = config["microterrain"]
    clasts = micro["clasts"]
    x_axis = np.asarray(x_coordinates, dtype=np.float32)
    y_axis = np.asarray(y_coordinates, dtype=np.float32)
    surface_z = np.asarray(surface_z, dtype=np.float32)
    displacement = np.asarray(displacement, dtype=np.float32)
    if surface_z.shape != (len(y_axis), len(x_axis)) or displacement.shape != surface_z.shape:
        raise ValueError("Level-2 surface arrays do not match the configured patch grid")
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    cluster = _cluster_field(grid_x - grid_x.mean(), grid_y - grid_y.mean(), float(clasts["cluster_wavelength_m"]), derive_seed(int(micro["seed"]), "clast_cluster_mask"), np)
    depression_preference = _normalise_01(-displacement, np)
    clustering_strength = float(clasts["clustering_strength"])
    correlation = float(clasts["microrelief_correlation"])
    weights = ((1.0 - clustering_strength) + clustering_strength * (0.15 + 0.85 * cluster)) * ((1.0 - correlation) + correlation * (0.30 + 0.70 * depression_preference))
    for _, _, radius_x, radius_y in exclusion_zones:
        if radius_x <= 0 or radius_y <= 0:
            raise ValueError("Clast exclusion radii must be positive")
    # Dynamic per-instance rejection below handles exact rock radius. This base
    # mask only removes the outermost grid border and keeps weighted sampling compact.
    valid = (grid_x > x_axis[0]) & (grid_x < x_axis[-1]) & (grid_y > y_axis[0]) & (grid_y < y_axis[-1])
    valid_indices = np.flatnonzero(valid)
    probability = weights.ravel()[valid_indices].astype(np.float64)
    probability /= probability.sum()
    gradients = np.gradient(surface_z, y_axis, x_axis)
    families = {}
    metrics = {
        "seed": int(micro["seed"]),
        "total_instances": contract["total_instances"],
        "counts": contract["counts"],
        "size_ranges_m": {},
        "burial_ranges": contract["burial_ranges"],
        "coarse_size_class_counts": contract["coarse_class_counts"],
        "clustering_strength": clustering_strength,
        "microrelief_correlation": correlation,
        "family_signatures_sha256": {},
        "exclusion_zones": [list(map(float, zone)) for zone in exclusion_zones],
        "size_aware_exclusion": True,
        "boundary_radius_guard": True,
        "coarse_overlap_guard": float(clasts["coarse_clasts"].get("minimum_center_spacing_factor", 0.0)),
        "coarse_size_sampling": "jittered_log_stratified_per_class",
    }
    edge_margin = float(clasts["edge_margin_m"])
    for family in FAMILIES:
        settings = clasts[family]
        family_seed = derive_seed(int(micro["seed"]), f"clast_scatter:{family}")
        result = _family_scatter(
            family,
            contract["counts"][family],
            probability,
            valid_indices,
            x_axis,
            y_axis,
            surface_z,
            gradients,
            settings,
            clasts,
            contract["variants_per_family"][family],
            edge_margin,
            exclusion_zones,
            family_seed,
            np,
        )
        families[family] = result
        metrics["size_ranges_m"][family] = [float(settings["size_min_m"]), float(settings["size_max_m"])]
        metrics["family_signatures_sha256"][family] = result["signature_sha256"]
    combined = hashlib.sha256("".join(metrics["family_signatures_sha256"][family] for family in FAMILIES).encode("ascii")).hexdigest()
    metrics["signature_sha256"] = combined
    metrics["cluster_mask_sha256"] = hashlib.sha256(cluster.astype(np.float32).tobytes(order="C")).hexdigest()
    return families, metrics
