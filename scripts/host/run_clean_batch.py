"""Orchestrate a deterministic, resumable Blender clean-image batch."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.host.run_wheel_surface_wear_pilot import _coverage, _draw_scratch
from scripts.host.validate_clean_batch import (
    validate_artifact_row,
    validate_clean_batch_run,
    validate_manifest_matches_plan,
)
from src.wheel_preparation.clean_batch import (
    build_plan,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    chunk_rows,
    run_fingerprint,
    sha256_bytes,
    validate_clean_batch_config,
    validate_run_id,
)
from src.wheel_preparation.domain_randomization import validate_domain_randomization_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _read_manifest_for_resume(path: Path, recovery_dir: Path) -> list[dict]:
    if not path.is_file():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        newline = raw.rfind(b"\n")
        valid = raw[: newline + 1] if newline >= 0 else b""
        tail = raw[newline + 1 :]
        recovery_dir.mkdir(parents=True, exist_ok=True)
        (recovery_dir / "manifest_truncated_tail.bin").write_bytes(tail)
        temporary = path.with_suffix(".jsonl.recovered")
        temporary.write_bytes(valid)
        os.replace(temporary, path)
        raw = valid
    rows = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Malformed committed manifest line {number}") from error
    return rows


def _generate_selected_wear_mask(wear_config: dict, variant_id: str, seed: int, output_path: Path) -> dict:
    settings = wear_config["mask_generation"]
    size = int(settings["resolution"])
    threshold = int(settings["coverage_threshold"])
    rng = random.Random(int(seed))
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    variants = sorted(
        (variant for variant in wear_config["variants"] if bool(variant["wear_enabled"])),
        key=lambda variant: float(variant["target_mask_coverage"]),
    )
    selected = next((variant for variant in variants if variant["id"] == variant_id), None)
    if selected is None:
        raise ValueError(f"Unknown enabled wear variant: {variant_id}")
    scratches = 0
    for variant in variants:
        target = float(variant["target_mask_coverage"])
        while _coverage(image, threshold) < target:
            for _ in range(24):
                _draw_scratch(draw, rng, settings, size)
                scratches += 1
        if variant["id"] == variant_id:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, compress_level=4)
    return {
        "path": str(output_path.resolve()),
        "sha256": _sha256(output_path),
        "coverage": _coverage(image, threshold),
        "target_coverage": float(selected["target_mask_coverage"]),
        "scratch_count_prefix": scratches,
    }


def _prepare_chunk_plan(rows: list[dict], wear_config: dict, staging: Path, chunk_key: str | int) -> tuple[Path, list[Path]]:
    work_dir = staging / f"chunk_{chunk_key}"
    work_dir.mkdir(parents=True, exist_ok=True)
    placeholder = work_dir / "wear_placeholder.png"
    Image.new("L", (4, 4), 0).save(placeholder)
    generated = [placeholder]
    prepared = []
    for row in rows:
        sample = copy.deepcopy(row)
        if sample["surface_wear"] == "surface_current":
            sample.update({"wear_mask": None, "wear_mask_sha256": None, "wear_mask_coverage": 0.0})
        else:
            mask = _generate_selected_wear_mask(
                wear_config,
                sample["surface_wear"],
                int(sample["wear_seed"]),
                work_dir / f"{sample['sample_id']}_{sample['surface_wear']}.png",
            )
            sample.update(
                {
                    "wear_mask": mask["path"],
                    "wear_mask_sha256": mask["sha256"],
                    "wear_mask_coverage": mask["coverage"],
                }
            )
            generated.append(Path(mask["path"]))
        prepared.append(sample)
    path = work_dir / "chunk_plan.json"
    _atomic_json(path, {"schema_version": 1, "placeholder_wear_mask": str(placeholder.resolve()), "samples": prepared})
    generated.append(path)
    return path, generated


def _cleanup_wear_masks(staging: Path) -> None:
    for pattern in ("chunk_*/wear_placeholder.png", "chunk_*/*_wear_light.png", "chunk_*/*_wear_evident.png"):
        for path in staging.glob(pattern):
            path.unlink(missing_ok=True)


def _resolve_blender(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
    else:
        located = shutil.which("blender")
        candidate = Path(located).resolve() if located else Path(r"D:\Programmi\Blender Foundation\Blender 5.2\blender.exe")
    if not candidate.is_file():
        raise FileNotFoundError("Blender executable not found; pass --blender-executable")
    return candidate


def _run_matrix_audit(blender: Path, source: Path, domain_path: Path, domain: dict, resolution: list[int], run_dir: Path) -> None:
    output = run_dir / "matrix_audit.json"
    script = REPO_ROOT / "scripts" / "blender" / "audit_clean_batch_matrix.py"
    command = [
        str(blender), "--background", str(source), "--python-exit-code", "1", "--python", str(script), "--",
        "--domain-config", str(domain_path),
        "--pose-config", str(_resolve_repo_path(domain["pose_config"])),
        "--sampling-config", str(_resolve_repo_path(domain["sampling_config"])),
        "--resolution", str(resolution[0]), str(resolution[1]),
        "--output", str(output),
    ]
    print("Running 6x4 matrix audit:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("ok") is not True:
        raise RuntimeError("Clean-batch matrix audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "blender" / "clean_batch.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--blender-executable", type=Path)
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)
    config_path = args.config.resolve()
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw.decode("utf-8"))
    contract = validate_clean_batch_config(config)
    domain_path = _resolve_repo_path(config["domain_randomization_config"])
    domain_raw = domain_path.read_bytes()
    domain = json.loads(domain_raw.decode("utf-8"))
    validate_domain_randomization_config(domain)
    referenced_paths = {
        "lighting": _resolve_repo_path(domain["lighting_config"]),
        "surface_wear": _resolve_repo_path(domain["wear_config"]),
        "camera_poses": _resolve_repo_path(domain["pose_config"]),
        "pose_sampling": _resolve_repo_path(domain["sampling_config"]),
    }
    referenced_raw = {name: path.read_bytes() for name, path in referenced_paths.items()}
    referenced_hashes = {name: sha256_bytes(payload) for name, payload in referenced_raw.items()}
    source = _resolve_repo_path(config["source"]["blend"])
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = _sha256(source)
    if source_sha256 != contract["source_sha256"]:
        raise RuntimeError(f"Source Blend SHA-256 mismatch: {source_sha256}")
    output_root = _resolve_repo_path(config["output"]["root"])
    outputs_root = (REPO_ROOT / "outputs").resolve()
    if not output_root.is_relative_to(outputs_root):
        raise ValueError("Clean-batch output root must stay below outputs/")
    run_dir = output_root / run_id
    blender = _resolve_blender(args.blender_executable)

    plan_rows = build_plan(config, domain)
    plan_bytes = canonical_jsonl_bytes(plan_rows)
    plan_sha256 = sha256_bytes(plan_bytes)
    config_sha256 = sha256_bytes(config_raw)
    domain_sha256 = sha256_bytes(domain_raw)
    fingerprint = run_fingerprint(
        source_sha256=source_sha256,
        batch_config_sha256=config_sha256,
        domain_config_sha256=domain_sha256,
        referenced_config_sha256s=referenced_hashes,
        plan_sha256=plan_sha256,
    )

    run_path = run_dir / "run.json"
    plan_path = run_dir / "plan.jsonl"
    manifest_path = run_dir / "manifest.jsonl"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists; use --resume only for the same fingerprint: {run_dir}")
    if args.resume:
        if not run_path.is_file() or not plan_path.is_file():
            raise RuntimeError("Cannot resume without run.json and plan.jsonl")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("fingerprint") != fingerprint:
            raise RuntimeError("Resume fingerprint mismatch")
        if plan_path.read_bytes() != plan_bytes:
            raise RuntimeError("Resume plan mismatch")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "rgb").mkdir()
        (run_dir / "target_wheel_mask").mkdir()
        (run_dir / "_staging").mkdir()
        plan_path.write_bytes(plan_bytes)
        manifest_path.write_bytes(b"")
        run = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "fingerprint": fingerprint,
            "started_at": _utc_now(),
            "source": {"path": _repo_relative(source), "sha256": source_sha256},
            "configs": {
                "batch": {"path": _repo_relative(config_path), "sha256": config_sha256},
                "domain_randomization": {"path": _repo_relative(domain_path), "sha256": domain_sha256},
                **{
                    name: {"path": _repo_relative(referenced_paths[name]), "sha256": digest}
                    for name, digest in sorted(referenced_hashes.items())
                },
            },
            "plan": {"path": "plan.jsonl", "sha256": plan_sha256},
            "selection": {key: config[key] for key in ("range", "sample_indices") if key in config},
            "render": config["render"],
            "execution": config["execution"],
            "counts": {"planned": len(plan_rows), "completed": 0},
        }
        _atomic_json(run_path, run)

    started = time.perf_counter()
    try:
        manifest_rows = _read_manifest_for_resume(manifest_path, run_dir / "_staging" / "recovery")
        expected_by_id = {row["sample_id"]: row for row in plan_rows}
        completed_ids = set()
        last_index = -1
        for row in manifest_rows:
            sample_id = row.get("sample_id")
            if sample_id not in expected_by_id or sample_id in completed_ids:
                raise RuntimeError(f"Invalid or duplicate committed sample: {sample_id}")
            index = int(row["sample_index"])
            if index <= last_index:
                raise RuntimeError("Committed manifest rows are not strictly ordered")
            validate_artifact_row(run_dir, row, tuple(contract["resolution"]))
            validate_manifest_matches_plan(row, expected_by_id[sample_id])
            completed_ids.add(sample_id)
            last_index = index

        matrix_path = run_dir / "matrix_audit.json"
        if not matrix_path.is_file() or json.loads(matrix_path.read_text(encoding="utf-8")).get("ok") is not True:
            benchmark = config.get("benchmark", {})
            reusable = benchmark.get("reuse_matrix_audit") if bool(benchmark.get("enabled", False)) else None
            if reusable:
                reusable_path = _resolve_repo_path(reusable)
                audit = json.loads(reusable_path.read_text(encoding="utf-8"))
                if (
                    audit.get("ok") is not True
                    or int(audit.get("combination_count", -1)) != 24
                    or int(audit.get("accepted_count", -1)) != 24
                ):
                    raise RuntimeError("Reusable clean matrix audit is not a validated 24/24 audit")
                shutil.copyfile(reusable_path, matrix_path)
            else:
                _run_matrix_audit(blender, source, domain_path, domain, contract["resolution"], run_dir)

        wear_path = referenced_paths["surface_wear"]
        wear_config = json.loads(referenced_raw["surface_wear"].decode("utf-8"))
        chunks = chunk_rows(plan_rows, completed_ids, contract["chunk_size"])
        if args.resume and run.get("status") == "complete" and not chunks:
            _cleanup_wear_masks(run_dir / "_staging")
            stale_failure = "error" in run or "failed_at" in run
            run.pop("error", None)
            run.pop("failed_at", None)
            if stale_failure:
                _atomic_json(run_path, run)
            final_validation = validate_clean_batch_run(run_dir, require_complete=True)
            print(json.dumps({"run_dir": str(run_dir), "fingerprint": fingerprint, "validation": final_validation}, indent=2))
            return
        chunk_reports = []
        for chunk_number, rows in enumerate(chunks):
            chunk_key = f"{int(rows[0]['sample_index']):06d}_{int(rows[-1]['sample_index']):06d}"
            chunk_plan_path, _generated = _prepare_chunk_plan(rows, wear_config, run_dir / "_staging", chunk_key)
            report_path = chunk_plan_path.parent / "chunk_report.json"
            script = REPO_ROOT / "scripts" / "blender" / "render_clean_batch.py"
            command = [
                str(blender), "--background", str(source), "--python-exit-code", "1", "--python", str(script), "--",
                "--batch-config", str(config_path),
                "--domain-config", str(domain_path),
                "--chunk-plan", str(chunk_plan_path),
                "--lighting-config", str(referenced_paths["lighting"]),
                "--wear-config", str(wear_path),
                "--pose-config", str(referenced_paths["camera_poses"]),
                "--sampling-config", str(referenced_paths["pose_sampling"]),
                "--run-dir", str(run_dir),
                "--manifest", str(manifest_path),
                "--chunk-report", str(report_path),
            ]
            print(f"Running clean chunk {chunk_number + 1}/{len(chunks)}:", subprocess.list2cmdline(command), flush=True)
            last_error = None
            for attempt in range(contract["max_chunk_retries"] + 1):
                try:
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
                    last_error = None
                    break
                except subprocess.CalledProcessError as error:
                    last_error = error
                    if attempt >= contract["max_chunk_retries"]:
                        raise
                    print(f"Chunk failed; restarting from immutable source ({attempt + 1}/{contract['max_chunk_retries']})", flush=True)
            if last_error is not None:
                raise last_error
            if _sha256(source) != source_sha256:
                raise RuntimeError("Immutable source Blend changed during rendering")
            if not report_path.is_file():
                raise RuntimeError("Blender chunk did not produce chunk_report.json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("ok") is not True or int(report["source_open_count"]) != 1:
                raise RuntimeError("Blender chunk report failed")
            if int(report.get("render_call_count", -1)) != int(report.get("rendered_count", -2)):
                raise RuntimeError("Blender chunk rendered-count contract failed")
            chunk_reports.append(report)
            _cleanup_wear_masks(run_dir / "_staging")
            manifest_rows = _read_manifest_for_resume(manifest_path, run_dir / "_staging" / "recovery")
            run["counts"]["completed"] = len(manifest_rows)
            run["status"] = "running"
            run["updated_at"] = _utc_now()
            _atomic_json(run_path, run)

        run["status"] = "validating"
        run["counts"]["completed"] = len(_read_manifest_for_resume(manifest_path, run_dir / "_staging" / "recovery"))
        if chunk_reports:
            run["performance"] = {
                "wall_seconds": float(time.perf_counter() - started),
                "sample_pipeline_seconds": sum(float(report.get("chunk_pipeline_seconds", 0.0)) for report in chunk_reports),
                "cpu_postprocess_seconds": sum(float(report.get("cpu_postprocess_seconds", 0.0)) for report in chunk_reports),
                "chunk_count": len(chunks),
                "source_open_count": sum(int(report["source_open_count"]) for report in chunk_reports),
                "render_call_count": sum(int(report["render_call_count"]) for report in chunk_reports),
            }
            runtimes = [report["runtime"] for report in chunk_reports]
            if any(runtime != runtimes[0] for runtime in runtimes[1:]):
                raise RuntimeError("Blender runtime/device changed between chunks")
            run["runtime"] = runtimes[0]
        elif "performance" not in run or "runtime" not in run:
            raise RuntimeError("Completed resume is missing preserved performance/runtime metadata")
        _cleanup_wear_masks(run_dir / "_staging")
        _atomic_json(run_path, run)
        validation = validate_clean_batch_run(run_dir, require_complete=False)
        run["status"] = "complete"
        run.pop("error", None)
        run.pop("failed_at", None)
        run["finished_at"] = _utc_now()
        run["validation"] = validation
        run["source"]["sha256_after"] = _sha256(source)
        _atomic_json(run_path, run)
        final_validation = validate_clean_batch_run(run_dir, require_complete=True)
        print(json.dumps({"run_dir": str(run_dir), "fingerprint": fingerprint, "validation": final_validation}, indent=2))
    except Exception as error:
        run["status"] = "failed"
        run["failed_at"] = _utc_now()
        run["error"] = f"{type(error).__name__}: {error}"
        run["counts"]["completed"] = len(_read_manifest_for_resume(manifest_path, run_dir / "_staging" / "recovery")) if manifest_path.exists() else 0
        _atomic_json(run_path, run)
        raise


if __name__ == "__main__":
    main()
