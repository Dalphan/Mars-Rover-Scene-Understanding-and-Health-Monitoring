from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


RUN_MANIFEST_SCHEMA_VERSION = 1
ARTIFACTS_MANIFEST_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def generate_run_id(model_run_name: str, seed: int, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d_%H%M%S")
    safe_model = _SAFE_NAME.sub("-", model_run_name.strip()).strip("-")
    if not safe_model:
        raise ValueError("model_run_name must contain at least one safe character")
    return f"{safe_model}__seed{int(seed)}__{timestamp}"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(data: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def collect_environment() -> dict[str, Any]:
    package_names = [
        "torch",
        "torchvision",
        "transformers",
        "segmentation-models-pytorch",
        "onnx",
        "onnxruntime-gpu",
        "tensorrt",
        "nvidia-modelopt",
        "google-api-python-client",
    ]
    packages: dict[str, str | None] = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    cuda_available = torch.cuda.is_available()
    gpus = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [
                        properties.major,
                        properties.minor,
                    ],
                }
            )
    return {
        "captured_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "packages": packages,
    }


def build_run_manifest(
    *,
    run_id: str,
    run_type: str,
    model_run_name: str,
    seed: int,
    config: dict[str, Any],
    source_run_id: str | None = None,
    source_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    if run_type not in {"fp32_training", "ptq", "qat_int8"}:
        raise ValueError("unsupported run_type")
    if run_type != "fp32_training" and not source_run_id:
        raise ValueError(f"source_run_id is required for {run_type}")
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "run_type": run_type,
        "created_at_utc": utc_now_iso(),
        "model_run_name": model_run_name,
        "seed": int(seed),
        "source_run_id": source_run_id,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "config": config,
    }


def validate_source_manifest(
    manifest: dict[str, Any],
    *,
    expected_model_run_name: str | None = None,
) -> None:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported run manifest schema_version "
            f"{manifest.get('schema_version')!r}"
        )
    required = {"run_id", "run_type", "model_run_name", "config"}
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"Run manifest is missing required fields: {missing}")
    if expected_model_run_name and manifest["model_run_name"] != expected_model_run_name:
        raise ValueError(
            "Model mismatch: expected "
            f"{expected_model_run_name!r}, got {manifest['model_run_name']!r}"
        )


def build_artifacts_manifest(
    root: str | Path,
    *,
    exclude_names: set[str] | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"Artifact root does not exist or is not a directory: {root_path}")
    excluded = set(exclude_names or ())
    excluded.add("artifacts_manifest.json")
    artifacts = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        artifacts.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": ARTIFACTS_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "root_name": root_path.name,
        "artifacts": artifacts,
    }


def verify_artifacts_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    root_path = Path(root).resolve()
    if manifest.get("schema_version") != ARTIFACTS_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported artifacts manifest schema_version")
    for artifact in manifest.get("artifacts", []):
        relative = Path(artifact["path"])
        path = (root_path / relative).resolve()
        if root_path not in path.parents:
            raise ValueError(f"Artifact path escapes root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"Artifact size mismatch: {relative}")
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"Artifact SHA-256 mismatch: {relative}")
