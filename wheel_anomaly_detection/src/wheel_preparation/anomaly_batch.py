"""Pure host-side contract for deterministic paired clean/hole batches."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping

from .clean_batch import canonical_json_bytes, sha256_bytes, validate_run_id
from .domain_randomization import sample_domain_randomization, validate_domain_randomization_config
from .hole_anomaly import (
    FAMILIES,
    IMAGE_SECTORS,
    SEVERITIES,
    derive_seed,
    sample_hole_descriptor,
    validate_hole_anomaly_config,
    validate_hole_descriptor,
)


SCHEMA_VERSION = 1
SUPPORTED_PASSES = ("rgb", "target_wheel_mask", "anomaly_mask")
WHEEL_ORDER = (
    "wheel_front_left",
    "wheel_front_right",
    "wheel_middle_left",
    "wheel_middle_right",
    "wheel_rear_left",
    "wheel_rear_right",
)
POSE_ORDER = (
    "A_overhead",
    "C_leading_three_quarter",
    "C_trailing_three_quarter",
    "D_upper_detail",
)
PAIR_LOCKED_DOMAIN_FIELDS = (
    "lighting_preset",
    "surface_wear",
    "camera_pose",
    "target_wheel",
    "healthy_roll_degrees",
    "wear_seed",
    "lighting_jitter",
)


def validate_anomaly_batch_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Unsupported anomaly-batch schema")
    source = config.get("source", {})
    if not isinstance(source.get("blend"), str) or not source["blend"]:
        raise ValueError("source.blend is required")
    source_hash = str(source.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source.sha256 must be a lowercase SHA-256 digest")
    if not isinstance(config.get("domain_randomization_config"), str):
        raise ValueError("domain_randomization_config is required")
    if not isinstance(config.get("anomaly_config"), str):
        raise ValueError("anomaly_config is required")

    has_range = "range" in config
    has_indices = "sample_indices" in config
    if has_range == has_indices:
        raise ValueError("Configure exactly one of range or sample_indices")
    if has_indices:
        indices = list(map(int, config["sample_indices"]))
        if not indices or any(index < 0 for index in indices):
            raise ValueError("sample_indices must contain non-negative integers")
        if indices != sorted(set(indices)):
            raise ValueError("sample_indices must be unique and strictly increasing")
    else:
        sample_range = config["range"]
        start = int(sample_range.get("start_index", -1))
        count = int(sample_range.get("count", 0))
        if start < 0 or count < 1:
            raise ValueError("range must have start_index >= 0 and count >= 1")
        indices = list(range(start, start + count))

    render = config.get("render", {})
    if render.get("engine") != "BLENDER_EEVEE":
        raise ValueError("Anomaly batch v1 requires BLENDER_EEVEE")
    if list(map(int, render.get("blender_major_minor", []))) != [5, 2]:
        raise ValueError("Anomaly batch v1 requires Blender 5.2")
    resolution = list(map(int, render.get("resolution", [])))
    if resolution != [1200, 900]:
        raise ValueError("Anomaly batch v1 requires 1200x900")
    benchmark = config.get("benchmark", {})
    benchmark_only = bool(benchmark.get("enabled", False))
    samples = int(render.get("samples", 0))
    if samples != 64 and not (benchmark_only and samples == 32):
        raise ValueError("Anomaly batch v1 requires 64 samples; benchmark-only configs may use 32")
    passes = tuple(map(str, render.get("passes", [])))
    if passes != SUPPORTED_PASSES:
        raise ValueError(f"render.passes must be exactly {list(SUPPORTED_PASSES)}")

    execution = config.get("execution", {})
    allowed_execution = {
        "chunk_size_pairs",
        "max_chunk_retries",
        "geometry_cache",
        "async_postprocess",
        "async_queue_depth",
    }
    if not set(execution) <= allowed_execution:
        raise ValueError("execution contains unknown keys")
    chunk_size = int(execution.get("chunk_size_pairs", 0))
    retries = int(execution.get("max_chunk_retries", -1))
    if not 1 <= chunk_size <= 50:
        raise ValueError("execution.chunk_size_pairs must be between 1 and 50")
    if not 0 <= retries <= 3:
        raise ValueError("execution.max_chunk_retries must be between 0 and 3")
    for key in ("geometry_cache", "async_postprocess"):
        if key in execution and type(execution[key]) is not bool:
            raise ValueError(f"execution.{key} must be boolean")
    async_depth = int(execution.get("async_queue_depth", 4))
    if not 1 <= async_depth <= 16:
        raise ValueError("execution.async_queue_depth must be between 1 and 16")
    output_root = config.get("output", {}).get("root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output.root is required")
    if benchmark_only:
        required = {"enabled", "geometry_cache", "png_compression_level", "scene_png_compression", "reuse_preflight"}
        allowed = required | {
            "async_postprocess",
            "async_queue_depth",
            "detailed_profiling",
        }
        if not required <= set(benchmark) or not set(benchmark) <= allowed:
            raise ValueError("Benchmark config keys mismatch")
        if int(benchmark["png_compression_level"]) not in range(10):
            raise ValueError("Benchmark PNG compression level must be 0..9")
        if int(benchmark["scene_png_compression"]) not in range(101):
            raise ValueError("Benchmark scene PNG compression must be 0..100")
        if not isinstance(benchmark["reuse_preflight"], str) or not benchmark["reuse_preflight"]:
            raise ValueError("Benchmark reuse_preflight is required")
        if bool(benchmark.get("async_postprocess", False)) and not 1 <= int(benchmark.get("async_queue_depth", 4)) <= 16:
            raise ValueError("Benchmark async_queue_depth must be between 1 and 16")

    pilot = config.get("pilot_stratification", {})
    if bool(pilot.get("enabled")):
        if len(indices) != 24:
            raise ValueError("The stratified anomaly pilot requires exactly 24 indices")
        if pilot.get("severity_formula") != "wheel_plus_pose_mod_3":
            raise ValueError("Unsupported pilot severity formula")
        if pilot.get("image_sector_formula") != "two_wheel_plus_pose_mod_3":
            raise ValueError("Unsupported pilot image-sector formula")
        family_counts = pilot.get("profile_family_counts", {})
        if tuple(family_counts) != FAMILIES or sum(map(int, family_counts.values())) != 24:
            raise ValueError("Pilot profile-family counts must cover all 24 pairs")
        shoulder_cells = [tuple(map(str, cell)) for cell in pilot.get("shoulder_cells", [])]
        if len(shoulder_cells) != 5 or len(set(shoulder_cells)) != 5:
            raise ValueError("Pilot must define five unique shoulder cells")
        if any(wheel not in WHEEL_ORDER or pose not in POSE_ORDER for wheel, pose in shoulder_cells):
            raise ValueError("Pilot shoulder cell contains an unknown wheel or pose")
        if int(pilot.get("eligible_flap_count", -1)) != 3:
            raise ValueError("Pilot must contain exactly three flaps")

    semantic_assignments = config.get("semantic_assignments")
    if semantic_assignments is not None:
        if bool(pilot.get("enabled")):
            raise ValueError("Production semantic assignments cannot be combined with pilot stratification")
        if not isinstance(semantic_assignments, list) or len(semantic_assignments) != len(indices):
            raise ValueError("semantic_assignments must contain exactly one row per sample index")
        assignment_indices = [int(row.get("sample_index", -1)) for row in semantic_assignments]
        if assignment_indices != indices:
            raise ValueError("semantic_assignments must follow sample_indices exactly")
        required = {"sample_index", "severity", "surface", "image_sector", "profile_family"}
        for row in semantic_assignments:
            if set(row) != required:
                raise ValueError("semantic assignment keys mismatch")
            if row["severity"] not in SEVERITIES or row["surface"] not in {"tread", "shoulder"}:
                raise ValueError("Unknown severity or surface in semantic assignment")
            if row["severity"] == "large" and row["surface"] != "tread":
                raise ValueError("Large semantic assignments must remain on tread")
            if row["image_sector"] not in IMAGE_SECTORS or row["profile_family"] not in FAMILIES:
                raise ValueError("Unknown sector or profile family in semantic assignment")

    return {
        "sample_indices": indices,
        "pair_count": len(indices),
        "resolution": resolution,
        "chunk_size_pairs": chunk_size,
        "max_chunk_retries": retries,
        "source_sha256": source_hash,
        "passes": list(passes),
    }


def _pilot_assignments(config: dict, samples: list[dict], master_seed: int) -> dict[int, dict]:
    pilot = config["pilot_stratification"]
    cells = {(sample["target_wheel"], sample["camera_pose"]) for sample in samples}
    expected_cells = {(wheel, pose) for wheel in WHEEL_ORDER for pose in POSE_ORDER}
    if cells != expected_cells or len(cells) != len(samples):
        missing = sorted(expected_cells - cells)
        duplicate_count = len(samples) - len(cells)
        raise ValueError(f"Pilot indices do not form the exact 6x4 matrix; missing={missing} duplicates={duplicate_count}")

    assignments: dict[int, dict] = {}
    shoulder_cells = {tuple(cell) for cell in pilot["shoulder_cells"]}
    for sample in samples:
        wheel_index = WHEEL_ORDER.index(sample["target_wheel"])
        pose_index = POSE_ORDER.index(sample["camera_pose"])
        severity = SEVERITIES[(wheel_index + pose_index) % 3]
        sector = IMAGE_SECTORS[(2 * wheel_index + pose_index) % 3]
        surface = "shoulder" if (sample["target_wheel"], sample["camera_pose"]) in shoulder_cells else "tread"
        if severity == "large" and surface == "shoulder":
            raise ValueError("Pilot stratification assigned a large hole to shoulder")
        assignments[int(sample["sample_index"])] = {
            "severity": severity,
            "image_sector": sector,
            "surface": surface,
        }

    ranked = sorted(samples, key=lambda sample: derive_seed(master_seed, int(sample["sample_index"]), "pilot-profile-rank"))
    cursor = 0
    for family in FAMILIES:
        count = int(pilot["profile_family_counts"][family])
        for sample in ranked[cursor : cursor + count]:
            assignments[int(sample["sample_index"])]["profile_family"] = family
        cursor += count

    eligible = [sample for sample in samples if assignments[int(sample["sample_index"])]["severity"] in {"medium", "large"}]
    eligible.sort(key=lambda sample: derive_seed(master_seed, int(sample["sample_index"]), "pilot-flap-rank"))
    flap_indices = {int(sample["sample_index"]) for sample in eligible[: int(pilot["eligible_flap_count"])]}
    for sample in samples:
        assignments[int(sample["sample_index"])]["flap_enabled"] = int(sample["sample_index"]) in flap_indices
    return assignments


def build_anomaly_plan(config: dict, domain_config: dict, anomaly_config: dict) -> list[dict]:
    contract = validate_anomaly_batch_config(config)
    validate_domain_randomization_config(domain_config)
    validate_hole_anomaly_config(anomaly_config)
    samples = [sample_domain_randomization(domain_config, index, attempt=0) for index in contract["sample_indices"]]
    if bool(config.get("pilot_stratification", {}).get("enabled")):
        assignments = _pilot_assignments(config, samples, int(domain_config["master_seed"]))
    else:
        assignments = {
            int(row["sample_index"]): {
                "severity": row["severity"],
                "surface": row["surface"],
                "image_sector": row["image_sector"],
                "profile_family": row["profile_family"],
            }
            for row in config.get("semantic_assignments", [])
        }
    anomaly_hash = sha256_bytes(canonical_json_bytes(anomaly_config))
    rows = []
    for pair_index, sample in enumerate(samples):
        override = assignments.get(int(sample["sample_index"]), {})
        descriptor = sample_hole_descriptor(
            anomaly_config,
            master_seed=int(domain_config["master_seed"]),
            sample_index=int(sample["sample_index"]),
            surface_wear=str(sample["surface_wear"]),
            severity=override.get("severity"),
            surface=override.get("surface"),
            image_sector=override.get("image_sector"),
            profile_family=override.get("profile_family"),
            flap_enabled=override.get("flap_enabled"),
        )
        errors = validate_hole_descriptor(anomaly_config, descriptor)
        if errors:
            raise ValueError(f"Invalid hole descriptor for {sample['sample_id']}: {errors}")
        pair_lock_id = hashlib.sha256(
            f"hole-pair:{int(domain_config['master_seed'])}:{int(sample['sample_index'])}:{anomaly_hash}".encode("ascii")
        ).hexdigest()[:24]
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "condition": "paired_clean_hole",
            "pair_id": sample["sample_id"],
            "pair_index": int(pair_index),
            "sample_index": int(sample["sample_index"]),
            "members": {
                "clean": f"{sample['sample_id']}_clean",
                "anomaly": f"{sample['sample_id']}_hole",
            },
            "pair_lock_id": pair_lock_id,
            "domain_sample": sample,
            "anomaly": descriptor,
        })
    return rows


def assert_pair_retry_preserves_semantics(planned_domain: Mapping, candidate: Mapping) -> None:
    for field in PAIR_LOCKED_DOMAIN_FIELDS:
        if candidate[field] != planned_domain[field]:
            raise ValueError(f"Camera retry changed pair-locked field {field}")


def resolved_pair_lock_sha256(shared: Mapping) -> str:
    return sha256_bytes(canonical_json_bytes(dict(shared)))


def anomaly_run_fingerprint(
    *,
    source_sha256: str,
    batch_config_sha256: str,
    domain_config_sha256: str,
    anomaly_config_sha256: str,
    plan_sha256: str,
    referenced_config_sha256s: Mapping[str, str] | None = None,
    runtime_code_sha256s: Mapping[str, str] | None = None,
) -> str:
    return sha256_bytes(canonical_json_bytes({
        "schema_version": SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "batch_config_sha256": batch_config_sha256,
        "domain_config_sha256": domain_config_sha256,
        "anomaly_config_sha256": anomaly_config_sha256,
        "referenced_config_sha256s": dict(sorted((referenced_config_sha256s or {}).items())),
        "runtime_code_sha256s": dict(sorted((runtime_code_sha256s or {}).items())),
        "plan_sha256": plan_sha256,
    }))


def chunk_pairs(rows: list[dict], completed_ids: set[str], chunk_size_pairs: int) -> list[list[dict]]:
    pending = [row for row in rows if row["pair_id"] not in completed_ids]
    return [pending[index : index + int(chunk_size_pairs)] for index in range(0, len(pending), int(chunk_size_pairs))]


def plan_summary(rows: list[dict]) -> dict:
    return {
        "pairs": len(rows),
        "wheels": dict(Counter(row["domain_sample"]["target_wheel"] for row in rows)),
        "poses": dict(Counter(row["domain_sample"]["camera_pose"] for row in rows)),
        "lighting": dict(Counter(row["domain_sample"]["lighting_preset"] for row in rows)),
        "wear": dict(Counter(row["domain_sample"]["surface_wear"] for row in rows)),
        "healthy_roll": dict(Counter(str(row["domain_sample"]["healthy_roll_degrees"]) for row in rows)),
        "severity": dict(Counter(row["anomaly"]["severity"] for row in rows)),
        "surface": dict(Counter(row["anomaly"]["surface"] for row in rows)),
        "image_sector": dict(Counter(row["anomaly"]["image_sector"] for row in rows)),
        "profile_family": dict(Counter(row["anomaly"]["profile_family"] for row in rows)),
        "flap_enabled": sum(bool(row["anomaly"]["flap"]["enabled"]) for row in rows),
    }


__all__ = [
    "anomaly_run_fingerprint",
    "assert_pair_retry_preserves_semantics",
    "build_anomaly_plan",
    "chunk_pairs",
    "plan_summary",
    "resolved_pair_lock_sha256",
    "validate_anomaly_batch_config",
    "validate_run_id",
]
