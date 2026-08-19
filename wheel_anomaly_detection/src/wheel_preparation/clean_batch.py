"""Pure host-side contract for deterministic clean Blender batches."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping

from .domain_randomization import sample_domain_randomization, validate_domain_randomization_config


SCHEMA_VERSION = 1
SUPPORTED_PASSES = ("rgb", "target_wheel_mask")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SEMANTIC_FIELDS = (
    "lighting_preset",
    "surface_wear",
    "camera_pose",
    "target_wheel",
    "healthy_roll_degrees",
    "wear_seed",
    "lighting_jitter",
)


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) for row in rows)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_run_id(run_id: str) -> str:
    value = str(run_id)
    if not RUN_ID_PATTERN.fullmatch(value):
        raise ValueError("run-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,95}")
    return value


def validate_clean_batch_config(config: dict) -> dict:
    if int(config.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError("Unsupported clean-batch schema")
    source = config.get("source", {})
    if not isinstance(source.get("blend"), str) or not source["blend"]:
        raise ValueError("source.blend is required")
    source_hash = str(source.get("sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("source.sha256 must be a lowercase SHA-256 digest")
    if not isinstance(config.get("domain_randomization_config"), str):
        raise ValueError("domain_randomization_config is required")

    has_range = "range" in config
    has_indices = "sample_indices" in config
    if has_range == has_indices:
        raise ValueError("Exactly one of range or sample_indices is required")
    if has_indices:
        sample_indices = [int(value) for value in config["sample_indices"]]
        if not sample_indices or sample_indices != sorted(set(sample_indices)) or any(value < 0 for value in sample_indices):
            raise ValueError("sample_indices must be non-empty, unique, strictly increasing and non-negative")
    else:
        sample_range = config.get("range", {})
        start = int(sample_range.get("start_index", -1))
        count = int(sample_range.get("count", 0))
        if start < 0 or count < 1:
            raise ValueError("range must have start_index >= 0 and count >= 1")
        sample_indices = list(range(start, start + count))

    render = config.get("render", {})
    if render.get("engine") != "BLENDER_EEVEE":
        raise ValueError("Clean batch v1 requires BLENDER_EEVEE")
    if list(map(int, render.get("blender_major_minor", []))) != [5, 2]:
        raise ValueError("Clean batch v1 requires Blender 5.2")
    resolution = list(map(int, render.get("resolution", [])))
    if len(resolution) != 2 or any(value <= 0 for value in resolution):
        raise ValueError("render.resolution must contain two positive integers")
    if int(render.get("samples", 0)) < 1:
        raise ValueError("render.samples must be positive")
    passes = tuple(map(str, render.get("passes", [])))
    if passes != SUPPORTED_PASSES:
        raise ValueError(f"render.passes must be exactly {list(SUPPORTED_PASSES)}")

    execution = config.get("execution", {})
    chunk_size = int(execution.get("chunk_size", 0))
    retries = int(execution.get("max_chunk_retries", -1))
    async_postprocess = bool(execution.get("async_postprocess", False))
    async_queue_depth = int(execution.get("async_queue_depth", 4))
    if chunk_size < 1 or chunk_size > 1000:
        raise ValueError("execution.chunk_size must be between 1 and 1000")
    if retries < 0 or retries > 3:
        raise ValueError("execution.max_chunk_retries must be between 0 and 3")
    if async_postprocess and not 1 <= async_queue_depth <= 16:
        raise ValueError("execution.async_queue_depth must be between 1 and 16")
    output_root = config.get("output", {}).get("root")
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("output.root is required")
    benchmark = config.get("benchmark", {})
    if bool(benchmark.get("enabled", False)):
        allowed = {
            "enabled",
            "geometry_cache",
            "async_postprocess",
            "png_compression_level",
            "scene_png_compression",
            "reuse_matrix_audit",
        }
        if set(benchmark) != allowed:
            raise ValueError("Clean benchmark config keys mismatch")
        if int(benchmark["png_compression_level"]) not in range(10):
            raise ValueError("Clean benchmark PNG compression level must be 0..9")
        if int(benchmark["scene_png_compression"]) not in range(101):
            raise ValueError("Clean benchmark scene PNG compression must be 0..100")
        if not isinstance(benchmark["reuse_matrix_audit"], str) or not benchmark["reuse_matrix_audit"]:
            raise ValueError("Clean benchmark reuse_matrix_audit is required")
    return {
        "sample_indices": sample_indices,
        "start_index": sample_indices[0],
        "count": len(sample_indices),
        "resolution": resolution,
        "chunk_size": chunk_size,
        "max_chunk_retries": retries,
        "passes": list(passes),
        "source_sha256": source_hash,
    }


def build_plan(config: dict, domain_config: dict) -> list[dict]:
    contract = validate_clean_batch_config(config)
    validate_domain_randomization_config(domain_config)
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "condition": "clean",
            **sample_domain_randomization(domain_config, sample_index, attempt=0),
        }
        for sample_index in contract["sample_indices"]
    ]


def run_fingerprint(
    *,
    source_sha256: str,
    batch_config_sha256: str,
    domain_config_sha256: str,
    plan_sha256: str,
    referenced_config_sha256s: Mapping[str, str] | None = None,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "source_sha256": source_sha256,
                "batch_config_sha256": batch_config_sha256,
                "domain_config_sha256": domain_config_sha256,
                "referenced_config_sha256s": dict(sorted((referenced_config_sha256s or {}).items())),
                "plan_sha256": plan_sha256,
            }
        )
    )


def assert_camera_retry_preserves_semantics(planned: dict, candidate: dict) -> None:
    for field in SEMANTIC_FIELDS:
        if candidate[field] != planned[field]:
            raise ValueError(f"Camera retry changed locked field {field}")


def chunk_rows(rows: list[dict], completed_ids: set[str], chunk_size: int) -> list[list[dict]]:
    pending = [row for row in rows if row["sample_id"] not in completed_ids]
    return [pending[index : index + int(chunk_size)] for index in range(0, len(pending), int(chunk_size))]
