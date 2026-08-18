"""Pure deterministic sampling for Blender wheel-domain randomization."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping


def _require_weights(name: str, weights: Mapping[str, float]) -> None:
    if not weights or any(float(value) <= 0.0 for value in weights.values()):
        raise ValueError(f"{name} weights must all be positive")
    if abs(sum(map(float, weights.values())) - 1.0) > 1e-9:
        raise ValueError(f"{name} weights must sum to one")


def validate_domain_randomization_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported domain-randomization schema")
    distributions = config["distributions"]
    for name in ("lighting", "surface_wear", "camera_pose", "target_wheel"):
        _require_weights(name, distributions[name])
    rolls = list(map(float, distributions["healthy_roll_degrees"]))
    if len(rolls) < 2 or len(set(rolls)) != len(rolls):
        raise ValueError("Healthy roll phases must be unique")
    lighting = config["jitter"]["lighting"]
    camera = config["jitter"]["camera"]
    for name, value in {**lighting, **camera}.items():
        if float(value) < 0.0:
            raise ValueError(f"Jitter bound {name} cannot be negative")
    if float(camera["maximum_position_offset_m"]) > 0.02 + 1e-12:
        raise ValueError("Camera-position jitter exceeds the audited 2 cm design limit")
    if float(camera["maximum_aim_offset_degrees"]) > 1.5 + 1e-12:
        raise ValueError("Camera-aim jitter exceeds the audited 1.5 degree design limit")
    if float(camera["focal_length_fraction"]) > 0.02 + 1e-12:
        raise ValueError("Focal-length jitter exceeds the audited 2 percent design limit")
    pair_lock = config["pair_lock"]
    if not pair_lock.get("same_parameters_for_healthy_and_anomaly") or not pair_lock.get("independent_of_anomaly_label"):
        raise ValueError("Counterfactual pair locking is mandatory")
    attempts = int(config["gates"]["maximum_camera_resample_attempts"])
    if attempts < 1 or attempts > 128:
        raise ValueError("Camera-gate resample attempts must be between 1 and 128")
    alignment = config["gates"]["terrain_alignment"]
    supported_alignment_modes = {
        "translate_rover_target_to_left_counterpart",
        "translate_rover_target_to_anchor_xy",
    }
    if alignment.get("mode") not in supported_alignment_modes or not alignment.get("record_world_translation"):
        raise ValueError("The terrain alignment must be explicit and recorded")
    if alignment.get("mode") == "translate_rover_target_to_anchor_xy" and not alignment.get("anchor_wheel_object"):
        raise ValueError("Anchor-based terrain alignment requires anchor_wheel_object")
    contact = alignment.get("vertical_contact", {})
    if contact.get("enabled") is not True or alignment.get("translation_axes") != ["x", "y", "z"]:
        raise ValueError("Terrain alignment must include deterministic vertical wheel contact")
    outer_fraction = float(contact.get("outer_radius_fraction", 0.0))
    if not 0.8 <= outer_fraction <= 1.0:
        raise ValueError("Vertical-contact outer radius fraction must be between 0.8 and 1.0")
    maximum_samples = int(contact.get("maximum_contact_samples", 0))
    minimum_samples = int(contact.get("minimum_contact_samples", 0))
    if minimum_samples < 8 or maximum_samples < minimum_samples:
        raise ValueError("Vertical-contact sample counts are invalid")
    target_penetration = float(contact.get("target_penetration_m", -1.0))
    maximum_penetration = float(contact.get("maximum_penetration_m", -1.0))
    maximum_gap = float(contact.get("maximum_gap_m", -1.0))
    if not 0.0 <= target_penetration <= maximum_penetration <= 0.01 or not 0.0 <= maximum_gap <= 0.005:
        raise ValueError("Vertical-contact clearance bounds are invalid")
    return {
        "lighting_ids": list(distributions["lighting"]),
        "wear_ids": list(distributions["surface_wear"]),
        "pose_ids": list(distributions["camera_pose"]),
        "wheel_ids": list(distributions["target_wheel"]),
        "roll_phases": rolls,
    }


def _seed(master_seed: int, sample_index: int, attempt: int = 0) -> int:
    payload = f"{int(master_seed)}:{int(sample_index)}:{int(attempt)}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _weighted_choice(rng: random.Random, weights: Mapping[str, float]) -> str:
    needle = rng.random()
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += float(weight)
        if needle <= cumulative:
            return str(key)
    return str(next(reversed(weights)))


def _symmetric(rng: random.Random, magnitude: float) -> float:
    return rng.uniform(-float(magnitude), float(magnitude))


def _disk(rng: random.Random, radius: float) -> tuple[float, float]:
    length = float(radius) * math.sqrt(rng.random())
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return length * math.cos(angle), length * math.sin(angle)


def _sphere(rng: random.Random, radius: float) -> tuple[float, float, float]:
    maximum = float(radius)
    while True:
        values = tuple(rng.uniform(-1.0, 1.0) for _ in range(3))
        squared = sum(value * value for value in values)
        if 1e-12 < squared <= 1.0:
            scale = maximum * (rng.random() ** (1.0 / 3.0)) / math.sqrt(squared)
            return tuple(value * scale for value in values)


def sample_domain_randomization(config: dict, sample_index: int, attempt: int = 0) -> dict:
    validate_domain_randomization_config(config)
    # Semantic choices, illumination and surface state are stable across camera-gate
    # retries.  A rejected framing must not silently bias those distributions.
    sample_seed = _seed(config["master_seed"], sample_index, 0)
    rng = random.Random(sample_seed)
    camera_attempt_seed = _seed(config["master_seed"], sample_index, attempt)
    camera_rng = random.Random(camera_attempt_seed)
    distributions = config["distributions"]
    lighting_bounds = config["jitter"]["lighting"]
    camera_bounds = config["jitter"]["camera"]
    aim_yaw, aim_pitch = _disk(camera_rng, float(camera_bounds["maximum_aim_offset_degrees"]))
    position = _sphere(camera_rng, float(camera_bounds["maximum_position_offset_m"]))
    pair_lock_id = hashlib.sha256(
        f"domain-pair:{sample_seed}:{camera_attempt_seed}".encode("ascii")
    ).hexdigest()[:20]
    return {
        "sample_id": f"dr_{int(sample_index):06d}",
        "sample_index": int(sample_index),
        "attempt": int(attempt),
        "sample_seed": sample_seed,
        "camera_attempt_seed": camera_attempt_seed,
        "pair_lock_id": pair_lock_id,
        "lighting_preset": _weighted_choice(rng, distributions["lighting"]),
        "surface_wear": _weighted_choice(rng, distributions["surface_wear"]),
        "camera_pose": _weighted_choice(rng, distributions["camera_pose"]),
        "target_wheel": _weighted_choice(rng, distributions["target_wheel"]),
        "healthy_roll_degrees": float(rng.choice(distributions["healthy_roll_degrees"])),
        "wear_seed": int(rng.randrange(0, 2**31)),
        "lighting_jitter": {
            "sun_azimuth_degrees": _symmetric(rng, lighting_bounds["sun_azimuth_degrees"]),
            "sun_elevation_degrees": _symmetric(rng, lighting_bounds["sun_elevation_degrees"]),
            "sun_energy_scale": 1.0 + _symmetric(rng, lighting_bounds["sun_energy_fraction"]),
            "world_strength_scale": 1.0 + _symmetric(rng, lighting_bounds["world_strength_fraction"]),
            "exposure_ev": _symmetric(rng, lighting_bounds["exposure_ev"]),
        },
        "camera_jitter": {
            "position_basis_m": list(position),
            "aim_yaw_degrees": aim_yaw,
            "aim_pitch_degrees": aim_pitch,
            "focal_length_scale": 1.0 + _symmetric(camera_rng, camera_bounds["focal_length_fraction"]),
        },
    }


def preview_tokens(sample: dict, categories: list[str]) -> set[tuple[str, str]]:
    field_by_category = {
        "lighting": "lighting_preset",
        "surface_wear": "surface_wear",
        "camera_pose": "camera_pose",
        "target_wheel": "target_wheel",
    }
    return {(category, str(sample[field_by_category[category]])) for category in categories}


def select_stratified_preview(config: dict, candidates: list[dict] | None = None) -> list[dict]:
    contract = validate_domain_randomization_config(config)
    preview = config["preview"]
    categories = list(map(str, preview["stratify_categories"]))
    universe = set()
    values = {
        "lighting": contract["lighting_ids"],
        "surface_wear": contract["wear_ids"],
        "camera_pose": contract["pose_ids"],
        "target_wheel": contract["wheel_ids"],
    }
    for category in categories:
        universe.update((category, value) for value in values[category])
    candidates = list(candidates) if candidates is not None else [
        sample_domain_randomization(config, index) for index in range(int(preview["candidate_pool_size"]))
    ]
    target_counts = preview.get("target_counts")
    if target_counts:
        field_by_category = {
            "lighting": "lighting_preset",
            "surface_wear": "surface_wear",
            "camera_pose": "camera_pose",
            "target_wheel": "target_wheel",
        }
        sample_count = int(preview["sample_count"])
        for category, quotas in target_counts.items():
            if set(quotas) != set(values[category]) or sum(map(int, quotas.values())) != sample_count:
                raise ValueError(f"Invalid preview quotas for {category}")
        search = random.Random(int(config["master_seed"]) ^ 0xD04A1)
        for _trial in range(20000):
            order = candidates.copy()
            search.shuffle(order)
            counts = {category: {key: 0 for key in quotas} for category, quotas in target_counts.items()}
            chosen = []
            for candidate in order:
                if any(
                    counts[category][candidate[field_by_category[category]]] >= int(quotas[candidate[field_by_category[category]]])
                    for category, quotas in target_counts.items()
                ):
                    continue
                chosen.append(candidate)
                for category in target_counts:
                    counts[category][candidate[field_by_category[category]]] += 1
                if len(chosen) == sample_count:
                    return sorted(chosen, key=lambda sample: sample["sample_index"])
        raise ValueError("Could not satisfy deterministic preview quotas from the valid candidate pool")
    selected = []
    selected_indices = set()
    missing = set(universe)
    while missing:
        best = max(
            (candidate for candidate in candidates if candidate["sample_index"] not in selected_indices),
            key=lambda candidate: (len(preview_tokens(candidate, categories) & missing), -candidate["sample_index"]),
        )
        gain = preview_tokens(best, categories) & missing
        if not gain:
            raise ValueError(f"Preview candidate pool cannot cover: {sorted(missing)}")
        selected.append(best)
        selected_indices.add(best["sample_index"])
        missing -= gain
    for candidate in candidates:
        if len(selected) >= int(preview["sample_count"]):
            break
        if candidate["sample_index"] not in selected_indices:
            selected.append(candidate)
            selected_indices.add(candidate["sample_index"])
    if len(selected) != int(preview["sample_count"]):
        raise ValueError("Preview candidate pool is too small")
    return selected
