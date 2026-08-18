"""Run a deterministic A-B-A state-drift probe for the wheel-hole renderer."""

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

from scripts.host.run_anomaly_batch import _prepare_chunk
from scripts.host.run_clean_batch import _atomic_json, _resolve_blender, _resolve_repo_path, _sha256
from src.wheel_preparation.anomaly_batch import build_anomaly_plan


def _clone_pair(row: dict, pair_id: str, pair_index: int) -> dict:
    cloned = copy.deepcopy(row)
    cloned["pair_id"] = pair_id
    cloned["pair_index"] = int(pair_index)
    cloned["members"] = {"clean": f"{pair_id}_clean", "anomaly": f"{pair_id}_hole"}
    return cloned


def _state_without_identity(row: dict) -> dict:
    resolved = copy.deepcopy(row["resolved"])
    resolved.pop("pair_id", None)
    return {
        "condition": row["condition"],
        "sample_index": row["sample_index"],
        "seeds": row["seeds"],
        "sampling": row["sampling"],
        "anomaly": row["anomaly"],
        "geometry": row["geometry"],
        "gates": row["gates"],
        "resolved": resolved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/anomaly_detection_2/anomaly_batch/drift_probe_v1",
    )
    parser.add_argument("--blender-executable", type=Path)
    args = parser.parse_args()

    run_dir = args.output_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Drift-probe output already exists: {run_dir}")
    output_root = (REPO_ROOT / "outputs").resolve()
    if not run_dir.is_relative_to(output_root):
        raise ValueError("Drift-probe output must stay below outputs/")
    for relative in ("rgb/clean", "rgb/anomaly", "target_wheel_mask", "anomaly_mask", "_staging"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)

    batch_path = REPO_ROOT / "configs/blender/anomaly_batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    domain_path = _resolve_repo_path(batch["domain_randomization_config"])
    anomaly_path = _resolve_repo_path(batch["anomaly_config"])
    domain = json.loads(domain_path.read_text(encoding="utf-8"))
    anomaly = json.loads(anomaly_path.read_text(encoding="utf-8"))
    plan = build_anomaly_plan(batch, domain, anomaly)
    a, b = plan[0], plan[1]
    rows = [
        _clone_pair(a, "drift_a1", 0),
        _clone_pair(b, "drift_b", 1),
        _clone_pair(a, "drift_a2", 2),
    ]

    lighting_path = _resolve_repo_path(domain["lighting_config"])
    wear_path = _resolve_repo_path(domain["wear_config"])
    pose_path = _resolve_repo_path(domain["pose_config"])
    sampling_path = _resolve_repo_path(domain["sampling_config"])
    wear_config = json.loads(wear_path.read_text(encoding="utf-8"))
    chunk_path = _prepare_chunk(rows, wear_config, run_dir / "_staging", "aba")
    manifest_path = run_dir / "manifest.jsonl"
    manifest_path.write_bytes(b"")
    report_path = run_dir / "_staging/aba_report.json"
    source = _resolve_repo_path(batch["source"]["blend"])
    source_sha_before = _sha256(source)
    if source_sha_before != batch["source"]["sha256"]:
        raise RuntimeError("Immutable source SHA-256 mismatch before drift probe")

    command = [
        str(_resolve_blender(args.blender_executable)),
        "--background",
        str(source),
        "--python-exit-code",
        "1",
        "--python",
        str(REPO_ROOT / "scripts/blender/render_anomaly_batch.py"),
        "--",
        "--batch-config",
        str(batch_path),
        "--domain-config",
        str(domain_path),
        "--anomaly-config",
        str(anomaly_path),
        "--chunk-plan",
        str(chunk_path),
        "--lighting-config",
        str(lighting_path),
        "--wear-config",
        str(wear_path),
        "--pose-config",
        str(pose_path),
        "--sampling-config",
        str(sampling_path),
        "--run-dir",
        str(run_dir),
        "--manifest",
        str(manifest_path),
        "--chunk-report",
        str(report_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    if [row["pair_id"] for row in manifest] != ["drift_a1", "drift_b", "drift_a2"]:
        raise RuntimeError("Drift-probe manifest order is invalid")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("ok") is not True
        or int(report.get("source_open_count", -1)) != 1
        or int(report.get("render_call_count", -1)) != 6
    ):
        raise RuntimeError("Drift-probe runtime counters are invalid")

    first, last = manifest[0], manifest[2]
    artifact_hashes_first = {key: value["sha256"] for key, value in first["artifacts"].items()}
    artifact_hashes_last = {key: value["sha256"] for key, value in last["artifacts"].items()}
    artifact_match = artifact_hashes_first == artifact_hashes_last
    state_match = _state_without_identity(first) == _state_without_identity(last)
    source_sha_after = _sha256(source)
    result = {
        "schema_version": 1,
        "ok": artifact_match and state_match and source_sha_before == source_sha_after,
        "sequence": ["drift_a1", "drift_b", "drift_a2"],
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "source_open_count": report["source_open_count"],
        "render_call_count": report["render_call_count"],
        "artifact_hashes_a1": artifact_hashes_first,
        "artifact_hashes_a2": artifact_hashes_last,
        "artifact_hashes_match": artifact_match,
        "resolved_state_match": state_match,
        "runtime": report["runtime"],
        "setup_datablocks": report["setup_datablocks"],
    }
    _atomic_json(run_dir / "drift_report.json", result)
    if result["ok"] is not True:
        raise RuntimeError(f"A-B-A drift probe failed: {result}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
