"""Pure deterministic generation and validation for Martian microterrain."""

from __future__ import annotations

import hashlib
import math


def derive_seed(master_seed: int, namespace: str) -> int:
    payload = f"microterrain-v1:{int(master_seed)}:{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def validate_level1_config(config: dict) -> dict:
    micro = config.get("microterrain", {})
    meso = micro.get("meso_relief", {})
    if not micro.get("enabled") or int(micro.get("level", 0)) < 1:
        raise ValueError("Microterrain Level 1 must be enabled")
    patch_size = float(micro["patch_size_m"])
    spacing = float(micro["grid_spacing_m"])
    transition = float(micro["edge_transition_m"])
    if patch_size <= 0 or spacing <= 0:
        raise ValueError("Patch size and grid spacing must be positive")
    intervals = round(patch_size / spacing)
    if not math.isclose(intervals * spacing, patch_size, abs_tol=1e-9):
        raise ValueError("Patch size must be an integer multiple of grid spacing")
    if intervals % 2:
        raise ValueError("Patch grid must have an even interval count so the target is a vertex")
    if not 0 < transition < patch_size / 2:
        raise ValueError("Edge transition must lie inside the patch")
    if not meso.get("enabled"):
        raise ValueError("Meso-relief must be enabled for Level 1")
    nyquist = 2.0 * spacing
    for name in ("large_scale", "medium_scale", "fine_scale"):
        layer = meso[name]
        if float(layer["wavelength_m"]) < nyquist:
            raise ValueError(f"{name} wavelength is below the geometric Nyquist limit")
        if float(layer["amplitude_m"]) < 0 or int(layer["components"]) < 2:
            raise ValueError(f"Invalid {name} parameters")
    maximum = float(meso["maximum_abs_displacement_m"])
    if not 0 < maximum <= 0.02:
        raise ValueError("Level 1 displacement gate must be in (0, 0.02] metres")
    return {"intervals": intervals, "samples": intervals + 1, "vertex_count": (intervals + 1) ** 2, "face_count": intervals**2}


def _normalize(field, np):
    centered = field - field.mean()
    standard_deviation = float(centered.std())
    return centered / standard_deviation if standard_deviation > 1e-12 else centered


def _spectral_field(grid_x, grid_y, layer: dict, anisotropy: float, seed: int, np):
    rng = np.random.default_rng(seed)
    field = np.zeros_like(grid_x, dtype=np.float32)
    wavelength = float(layer["wavelength_m"])
    count = int(layer["components"])
    preferred_direction = rng.uniform(0.0, math.pi)
    for _ in range(count):
        if rng.random() < anisotropy:
            angle = preferred_direction + rng.normal(0.0, 0.22)
        else:
            angle = rng.uniform(0.0, math.pi)
        local_wavelength = wavelength * rng.uniform(0.72, 1.35)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        weight = rng.uniform(0.65, 1.0)
        projection = math.cos(angle) * grid_x + math.sin(angle) * grid_y
        field += weight * np.cos((2.0 * math.pi / local_wavelength) * projection + phase)
    return _normalize(field, np) * float(layer["amplitude_m"])


def generate_meso_relief(x_coordinates, y_coordinates, config: dict, np):
    contract = validate_level1_config(config)
    micro = config["microterrain"]
    meso = micro["meso_relief"]
    master_seed = int(micro["seed"])
    grid_x, grid_y = np.meshgrid(np.asarray(x_coordinates, dtype=np.float32), np.asarray(y_coordinates, dtype=np.float32))
    local_x = grid_x - float((x_coordinates[0] + x_coordinates[-1]) / 2.0)
    local_y = grid_y - float((y_coordinates[0] + y_coordinates[-1]) / 2.0)
    anisotropy = float(meso["anisotropy"])
    displacement = np.zeros_like(grid_x, dtype=np.float32)
    layer_seeds = {}
    for name in ("large_scale", "medium_scale", "fine_scale"):
        seed = derive_seed(master_seed, name)
        layer_seeds[name] = seed
        displacement += _spectral_field(local_x, local_y, meso[name], anisotropy, seed, np)

    ripple_seed = derive_seed(master_seed, "ripples")
    layer_seeds["ripples"] = ripple_seed
    ripple_rng = np.random.default_rng(ripple_seed)
    angle = math.radians(float(meso["ripple_direction_deg"])) + ripple_rng.normal(0.0, 0.08)
    ripple_coordinate = math.cos(angle) * local_x + math.sin(angle) * local_y
    envelope = _spectral_field(local_x, local_y, {"wavelength_m": 0.35, "amplitude_m": 1.0, "components": 5}, 0.25, derive_seed(master_seed, "ripple_envelope"), np)
    envelope = np.clip(0.5 + 0.22 * envelope, 0.0, 1.0)
    ripple_amplitude = float(meso["large_scale"]["amplitude_m"]) * float(meso["ripple_strength"])
    displacement += ripple_amplitude * envelope * np.sin(2.0 * math.pi * ripple_coordinate / float(meso["ripple_wavelength_m"]) + ripple_rng.uniform(0.0, 2.0 * math.pi))

    area = float(micro["patch_size_m"]) ** 2
    depression_seed = derive_seed(master_seed, "depressions")
    layer_seeds["depressions"] = depression_seed
    depression_rng = np.random.default_rng(depression_seed)
    depression_count = int(round(area * float(meso["depression_density_m2"])))
    depression_radius = list(map(float, meso["depression_radius_m"]))
    depression_depth = list(map(float, meso["depression_depth_m"]))
    half_size = float(micro["patch_size_m"]) / 2.0
    for _ in range(depression_count):
        cx, cy = depression_rng.uniform(-half_size, half_size, size=2)
        radius = depression_rng.uniform(*depression_radius)
        depth = depression_rng.uniform(*depression_depth)
        aspect = depression_rng.uniform(0.65, 1.35)
        theta = depression_rng.uniform(0.0, math.pi)
        dx, dy = local_x - cx, local_y - cy
        xr = math.cos(theta) * dx + math.sin(theta) * dy
        yr = -math.sin(theta) * dx + math.cos(theta) * dy
        displacement -= depth * np.exp(-0.5 * ((xr / radius) ** 2 + (yr / (radius * aspect)) ** 2))

    crust_seed = derive_seed(master_seed, "crust")
    layer_seeds["crust"] = crust_seed
    crust_rng = np.random.default_rng(crust_seed)
    crust_count = int(round(area * float(meso["crust_patch_density_m2"])))
    crust_mask = np.zeros_like(displacement)
    for _ in range(crust_count):
        cx, cy = crust_rng.uniform(-half_size, half_size, size=2)
        radius = crust_rng.uniform(*map(float, meso["crust_radius_m"]))
        aspect = crust_rng.uniform(0.55, 1.5)
        theta = crust_rng.uniform(0.0, math.pi)
        dx, dy = local_x - cx, local_y - cy
        xr = math.cos(theta) * dx + math.sin(theta) * dy
        yr = -math.sin(theta) * dx + math.cos(theta) * dy
        mask = np.exp(-0.5 * ((xr / radius) ** 2 + (yr / (radius * aspect)) ** 2))
        crust_mask = np.maximum(crust_mask, mask)
    displacement = displacement * (1.0 - float(meso["crust_flattening"]) * crust_mask) + float(meso["crust_raise_m"]) * crust_mask

    maximum = float(meso["maximum_abs_displacement_m"])
    clipped_fraction = float((np.abs(displacement) > maximum).mean())
    displacement = np.clip(displacement, -maximum, maximum)
    edge_distance = np.minimum.reduce((grid_x - x_coordinates[0], x_coordinates[-1] - grid_x, grid_y - y_coordinates[0], y_coordinates[-1] - grid_y))
    edge_t = np.clip(edge_distance / float(micro["edge_transition_m"]), 0.0, 1.0)
    edge_weight = edge_t * edge_t * (3.0 - 2.0 * edge_t)
    displacement *= edge_weight
    displacement = displacement.astype(np.float32)
    signature = hashlib.sha256(displacement.tobytes(order="C")).hexdigest()
    metrics = {
        "seed": master_seed,
        "derived_seeds": layer_seeds,
        "signature_sha256": signature,
        "min_displacement_m": float(displacement.min()),
        "max_displacement_m": float(displacement.max()),
        "maximum_absolute_displacement_m": float(np.abs(displacement).max()),
        "mean_abs_displacement_m": float(np.abs(displacement).mean()),
        "rms_displacement_m": float(np.sqrt(np.mean(displacement * displacement))),
        "clipped_fraction": clipped_fraction,
        "edge_max_abs_displacement_m": float(max(np.abs(displacement[0]).max(), np.abs(displacement[-1]).max(), np.abs(displacement[:, 0]).max(), np.abs(displacement[:, -1]).max())),
        "depression_count": depression_count,
        "crust_patch_count": crust_count,
        "patch_vertex_count": contract["vertex_count"],
        "patch_face_count": contract["face_count"],
    }
    return displacement, metrics
