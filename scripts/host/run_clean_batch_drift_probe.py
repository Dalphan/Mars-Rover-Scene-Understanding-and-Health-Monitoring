"""Render A-B-A in one Blender process and prove that sample state is restored."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.host.run_clean_batch import (
    _cleanup_wear_masks,
    _prepare_chunk_plan,
    _resolve_blender,
    _resolve_repo_path,
    _sha256,
)
from src.wheel_preparation.clean_batch import build_plan, validate_clean_batch_config


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "blender" / "clean_batch.json")
    parser.add_argument("--blender-executable", type=Path)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "anomaly_detection_2" / "clean_batch" / "drift_probe_v1",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_clean_batch_config(config)
    domain_path = _resolve_repo_path(config["domain_randomization_config"])
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    source = _resolve_repo_path(config["source"]["blend"])
    source_before = _sha256(source)
    if source_before != config["source"]["sha256"]:
        raise RuntimeError("Source hash mismatch before A-B-A probe")
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    (run_dir / "rgb").mkdir(parents=True)
    (run_dir / "target_wheel_mask").mkdir()
    (run_dir / "_staging").mkdir()
    manifest = run_dir / "manifest.jsonl"
    manifest.write_bytes(b"")

    base_rows = build_plan(config, domain)
    by_index = {int(row["sample_index"]): row for row in base_rows}
    a = by_index[78]
    b = by_index[79]
    probes = []
    for artifact_id, source_row in (("aba_a1", a), ("aba_b", b), ("aba_a2", a)):
        row = copy.deepcopy(source_row)
        row["artifact_sample_id"] = artifact_id
        probes.append(row)
    wear_path = _resolve_repo_path(domain["wear_config"])
    wear = json.loads(wear_path.read_text(encoding="utf-8"))
    chunk_plan, _generated = _prepare_chunk_plan(probes, wear, run_dir / "_staging", 0)
    report_path = chunk_plan.parent / "chunk_report.json"
    blender = _resolve_blender(args.blender_executable)
    command = [
        str(blender), "--background", str(source), "--python-exit-code", "1",
        "--python", str(REPO_ROOT / "scripts" / "blender" / "render_clean_batch.py"), "--",
        "--batch-config", str(config_path),
        "--domain-config", str(domain_path),
        "--chunk-plan", str(chunk_plan),
        "--lighting-config", str(_resolve_repo_path(domain["lighting_config"])),
        "--wear-config", str(wear_path),
        "--pose-config", str(_resolve_repo_path(domain["pose_config"])),
        "--sampling-config", str(_resolve_repo_path(domain["sampling_config"])),
        "--run-dir", str(run_dir),
        "--manifest", str(manifest),
        "--chunk-report", str(report_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    _cleanup_wear_masks(run_dir / "_staging")
    rows = _read_jsonl(manifest)
    if [row["sample_id"] for row in rows] != ["aba_a1", "aba_b", "aba_a2"]:
        raise RuntimeError("A-B-A manifest order mismatch")
    first, middle, last = rows
    comparable_keys = ("seeds", "sampling", "resolved", "geometry", "gates")
    comparisons = {key: first[key] == last[key] for key in comparable_keys}
    rgb_equal = first["artifacts"]["rgb"]["sha256"] == last["artifacts"]["rgb"]["sha256"]
    mask_equal = first["artifacts"]["target_wheel_mask"]["sha256"] == last["artifacts"]["target_wheel_mask"]["sha256"]
    b_is_distinct = first["artifacts"]["rgb"]["sha256"] != middle["artifacts"]["rgb"]["sha256"]
    source_after = _sha256(source)
    report = {
        "schema_version": 1,
        "ok": all(comparisons.values()) and rgb_equal and mask_equal and b_is_distinct and source_after == source_before,
        "sample_indices": [78, 79, 78],
        "metadata_equal": comparisons,
        "rgb_hash_equal": rgb_equal,
        "mask_hash_equal": mask_equal,
        "middle_rgb_distinct": b_is_distinct,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "chunk_report": json.loads(report_path.read_text(encoding="utf-8")),
    }
    (run_dir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise RuntimeError("Clean-batch A-B-A drift probe failed")


if __name__ == "__main__":
    main()
