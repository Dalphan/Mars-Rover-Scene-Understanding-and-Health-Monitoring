from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.utils.artifacts import (
    build_artifacts_manifest,
    build_run_manifest,
    generate_run_id,
    save_json,
    sha256_file,
    validate_source_manifest,
    verify_artifacts_manifest,
)


def test_run_id_and_manifest_validation():
    run_id = generate_run_id(
        "smp/unet resnet34",
        42,
        now=datetime(2026, 7, 10, 12, 30, tzinfo=UTC),
    )
    assert run_id == "smp-unet-resnet34__seed42__20260710_123000"
    manifest = build_run_manifest(
        run_id=run_id,
        run_type="fp32_training",
        model_run_name="smp_unet_resnet34",
        seed=42,
        config={"model": {"name": "smp"}},
    )
    validate_source_manifest(manifest, expected_model_run_name="smp_unet_resnet34")
    with pytest.raises(ValueError, match="Model mismatch"):
        validate_source_manifest(manifest, expected_model_run_name="segformer_b0")


def test_artifacts_manifest_round_trip_and_tamper_detection(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested" / "b.bin").write_bytes(b"\x00\x01")
    manifest = build_artifacts_manifest(tmp_path)
    save_json(manifest, tmp_path / "artifacts_manifest.json")
    assert {item["path"] for item in manifest["artifacts"]} == {
        "a.txt",
        "nested/b.bin",
    }
    assert sha256_file(tmp_path / "a.txt") == manifest["artifacts"][0]["sha256"]
    verify_artifacts_manifest(tmp_path, manifest)
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_artifacts_manifest(tmp_path, manifest)
