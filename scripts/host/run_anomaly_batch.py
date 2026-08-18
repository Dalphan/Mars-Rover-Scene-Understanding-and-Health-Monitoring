"""Orchestrate a deterministic, resumable paired clean/hole Blender batch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.host.run_clean_batch import (
    _atomic_json,
    _generate_selected_wear_mask,
    _read_manifest_for_resume,
    _repo_relative,
    _resolve_blender,
    _resolve_repo_path,
    _sha256,
    _utc_now,
)
from scripts.host.validate_anomaly_batch import (
    validate_anomaly_batch_run,
    validate_manifest_matches_plan,
    validate_pair_artifacts,
)
from src.wheel_preparation.anomaly_batch import (
    anomaly_run_fingerprint,
    build_anomaly_plan,
    chunk_pairs,
    plan_summary,
    validate_anomaly_batch_config,
    validate_run_id,
)
from src.wheel_preparation.clean_batch import canonical_jsonl_bytes, sha256_bytes
from src.wheel_preparation.domain_randomization import validate_domain_randomization_config
from src.wheel_preparation.hole_anomaly import validate_hole_anomaly_config


RUNTIME_CODE_PATHS = {
    "blender_renderer": REPO_ROOT / "scripts/blender/render_anomaly_batch.py",
    "blender_injector": REPO_ROOT / "scripts/blender/wheel_hole_anomaly.py",
    "blender_pose_sampling": REPO_ROOT / "scripts/blender/wheel_pose_sampling.py",
    "host_anomaly_contract": REPO_ROOT / "src/wheel_preparation/anomaly_batch.py",
    "host_hole_contract": REPO_ROOT / "src/wheel_preparation/hole_anomaly.py",
}


def _generate_hole_mask(descriptor: dict, output_path: Path) -> dict:
    size = 512
    points = [(float(point[0]), float(point[1])) for point in descriptor["points_long_short_m"]]
    half_x = max(abs(point[0]) for point in points) * 1.12 + float(descriptor["rim_width_m"])
    half_y = max(abs(point[1]) for point in points) * 1.12 + float(descriptor["rim_width_m"])
    extent = [2.0 * half_x, 2.0 * half_y]
    polygon = [
        (
            round((point[0] / extent[0] + 0.5) * (size - 1)),
            round((0.5 - point[1] / extent[1]) * (size - 1)),
        )
        for point in points
    ]
    image = Image.new("L", (size, size), 0)
    ImageDraw.Draw(image).polygon(polygon, fill=255)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, compress_level=4)
    return {"path": str(output_path.resolve()), "sha256": _sha256(output_path), "mapping_extent_m": extent}


def _prepare_chunk(rows: list[dict], wear_config: dict, staging: Path, key: str) -> Path:
    work = staging / f"chunk_{key}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    wear_placeholder = work / "wear_placeholder.png"
    hole_placeholder = work / "hole_placeholder.png"
    Image.new("L", (4, 4), 0).save(wear_placeholder)
    Image.new("L", (4, 4), 0).save(hole_placeholder)
    prepared = []
    for row in rows:
        item = copy.deepcopy(row)
        sample = item["domain_sample"]
        if sample["surface_wear"] == "surface_current":
            item.update({"wear_mask": None, "wear_mask_sha256": None, "wear_mask_coverage": 0.0})
        else:
            wear = _generate_selected_wear_mask(
                wear_config,
                sample["surface_wear"],
                int(sample["wear_seed"]),
                work / f"{item['pair_id']}_{sample['surface_wear']}.png",
            )
            item.update({"wear_mask": wear["path"], "wear_mask_sha256": wear["sha256"], "wear_mask_coverage": wear["coverage"]})
        hole = _generate_hole_mask(item["anomaly"], work / f"{item['pair_id']}_hole.png")
        item.update({"hole_mask": hole["path"], "hole_mask_sha256": hole["sha256"], "hole_mapping_extent_m": hole["mapping_extent_m"]})
        prepared.append(item)
    plan_path = work / "chunk_plan.json"
    _atomic_json(plan_path, {
        "schema_version": 1,
        "placeholder_wear_mask": str(wear_placeholder.resolve()),
        "placeholder_hole_mask": str(hole_placeholder.resolve()),
        "pairs": prepared,
    })
    return plan_path


def _run_preflight(blender: Path, source: Path, domain_path: Path, anomaly_path: Path, domain: dict, resolution: list[int], output: Path) -> None:
    command = [
        str(blender), "--background", str(source), "--python-exit-code", "1",
        "--python", str(REPO_ROOT / "scripts/blender/audit_anomaly_matrix.py"), "--",
        "--domain-config", str(domain_path), "--anomaly-config", str(anomaly_path),
        "--pose-config", str(_resolve_repo_path(domain["pose_config"])),
        "--sampling-config", str(_resolve_repo_path(domain["sampling_config"])),
        "--resolution", str(resolution[0]), str(resolution[1]), "--output", str(output),
    ]
    print("Running 360-stratum anomaly preflight:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("ok") is not True or int(report.get("tested_count", -1)) != 360:
        raise RuntimeError("Anomaly preflight failed")


def _contact_sheets(run_dir: Path, manifest: list[dict]) -> None:
    qa = run_dir / "qa"
    qa.mkdir(exist_ok=True)
    font = ImageFont.load_default()
    for pose in ("A_overhead", "C_leading_three_quarter", "C_trailing_three_quarter", "D_upper_detail"):
        rows = [row for row in manifest if row["sampling"]["camera_pose"] == pose]
        tile = (300, 225)
        header = 28
        sheet = Image.new("RGB", (tile[0] * 4, header + (tile[1] + header) * len(rows)), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        for column, label in enumerate(("clean", "anomaly", "difference x4", "mask overlay")):
            draw.text((column * tile[0] + 8, 8), label, fill="white", font=font)
        for row_index, row in enumerate(rows):
            clean = Image.open(run_dir / row["artifacts"]["clean_rgb"]["path"]).convert("RGB")
            anomaly = Image.open(run_dir / row["artifacts"]["anomaly_rgb"]["path"]).convert("RGB")
            mask = Image.open(run_dir / row["artifacts"]["anomaly_mask"]["path"]).convert("L")
            difference = ImageEnhance.Brightness(ImageChops.difference(clean, anomaly)).enhance(4.0)
            overlay = anomaly.copy()
            red = Image.new("RGB", anomaly.size, (255, 32, 24))
            overlay = Image.composite(Image.blend(overlay, red, 0.55), overlay, mask)
            y = header + row_index * (tile[1] + header)
            draw.text((8, y + 5), f"{row['pair_id']}  {row['sampling']['target_wheel']}", fill="white", font=font)
            for column, image in enumerate((clean, anomaly, difference, overlay)):
                sheet.paste(image.resize(tile, Image.Resampling.LANCZOS), (column * tile[0], y + header))
        sheet.save(qa / f"contact_sheet_{pose}.png", compress_level=4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/blender/anomaly_batch.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--blender-executable", type=Path)
    args = parser.parse_args()
    run_id = validate_run_id(args.run_id)
    config_path = args.config.resolve()
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw.decode("utf-8"))
    contract = validate_anomaly_batch_config(config)
    domain_path = _resolve_repo_path(config["domain_randomization_config"])
    anomaly_path = _resolve_repo_path(config["anomaly_config"])
    domain_raw, anomaly_raw = domain_path.read_bytes(), anomaly_path.read_bytes()
    domain = json.loads(domain_raw.decode("utf-8"))
    anomaly = json.loads(anomaly_raw.decode("utf-8"))
    validate_domain_randomization_config(domain)
    validate_hole_anomaly_config(anomaly)
    refs = {
        "lighting": _resolve_repo_path(domain["lighting_config"]),
        "surface_wear": _resolve_repo_path(domain["wear_config"]),
        "camera_poses": _resolve_repo_path(domain["pose_config"]),
        "pose_sampling": _resolve_repo_path(domain["sampling_config"]),
    }
    refs_raw = {name: path.read_bytes() for name, path in refs.items()}
    refs_hash = {name: sha256_bytes(raw) for name, raw in refs_raw.items()}
    runtime_code_hash = {name: _sha256(path) for name, path in RUNTIME_CODE_PATHS.items()}
    source = _resolve_repo_path(config["source"]["blend"])
    source_sha = _sha256(source)
    if source_sha != contract["source_sha256"]:
        raise RuntimeError(f"Source Blend SHA-256 mismatch: {source_sha}")
    output_root = _resolve_repo_path(config["output"]["root"])
    if not output_root.is_relative_to((REPO_ROOT / "outputs").resolve()):
        raise ValueError("Anomaly output root must stay below outputs/")
    run_dir = output_root / run_id
    blender = _resolve_blender(args.blender_executable)
    plan_rows = build_anomaly_plan(config, domain, anomaly)
    plan_bytes = canonical_jsonl_bytes(plan_rows)
    plan_sha = sha256_bytes(plan_bytes)
    hashes = {
        "batch": sha256_bytes(config_raw), "domain_randomization": sha256_bytes(domain_raw),
        "anomaly": sha256_bytes(anomaly_raw), **refs_hash,
    }
    fingerprint = anomaly_run_fingerprint(
        source_sha256=source_sha, batch_config_sha256=hashes["batch"],
        domain_config_sha256=hashes["domain_randomization"], anomaly_config_sha256=hashes["anomaly"],
        referenced_config_sha256s=refs_hash, runtime_code_sha256s=runtime_code_hash, plan_sha256=plan_sha,
    )
    run_path, plan_path, manifest_path = run_dir / "run.json", run_dir / "plan.jsonl", run_dir / "manifest.jsonl"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run exists; use --resume only for the same fingerprint: {run_dir}")
    if args.resume:
        if not run_path.is_file() or not plan_path.is_file():
            raise RuntimeError("Cannot resume without run.json and plan.jsonl")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("fingerprint") != fingerprint or plan_path.read_bytes() != plan_bytes:
            raise RuntimeError("Resume fingerprint/plan mismatch")
    else:
        for path in (
            run_dir / "rgb/clean", run_dir / "rgb/anomaly", run_dir / "target_wheel_mask",
            run_dir / "anomaly_mask", run_dir / "qa", run_dir / "_staging",
        ):
            path.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(plan_bytes)
        manifest_path.write_bytes(b"")
        run = {
            "schema_version": 1, "run_id": run_id, "status": "running", "fingerprint": fingerprint,
            "started_at": _utc_now(), "source": {"path": _repo_relative(source), "sha256": source_sha},
            "configs": {
                "batch": {"path": _repo_relative(config_path), "sha256": hashes["batch"]},
                "domain_randomization": {"path": _repo_relative(domain_path), "sha256": hashes["domain_randomization"]},
                "anomaly": {"path": _repo_relative(anomaly_path), "sha256": hashes["anomaly"]},
                **{name: {"path": _repo_relative(refs[name]), "sha256": refs_hash[name]} for name in sorted(refs)},
            },
            "runtime_code": {
                name: {"path": _repo_relative(RUNTIME_CODE_PATHS[name]), "sha256": runtime_code_hash[name]}
                for name in sorted(RUNTIME_CODE_PATHS)
            },
            "plan": {"path": "plan.jsonl", "sha256": plan_sha, "summary": plan_summary(plan_rows)},
            "selection": {key: config[key] for key in ("sample_indices", "range") if key in config},
            "render": config["render"], "execution": config["execution"],
            "counts": {"planned_pairs": len(plan_rows), "completed_pairs": 0},
        }
        _atomic_json(run_path, run)

    started = time.perf_counter()
    try:
        manifest = _read_manifest_for_resume(manifest_path, run_dir / "_staging/recovery")
        expected = {row["pair_id"]: row for row in plan_rows}
        completed, last_pair_index = set(), -1
        for row in manifest:
            pair_id = row.get("pair_id")
            if pair_id not in expected or pair_id in completed or int(row["pair_index"]) <= last_pair_index:
                raise RuntimeError(f"Invalid committed manifest row: {pair_id}")
            validate_manifest_matches_plan(row, expected[pair_id])
            validate_pair_artifacts(run_dir, row, tuple(contract["resolution"]))
            completed.add(pair_id)
            last_pair_index = int(row["pair_index"])
        preflight = run_dir / "anomaly_preflight.json"
        if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8")).get("ok") is not True:
            benchmark = config.get("benchmark", {})
            if bool(benchmark.get("enabled", False)):
                reusable = _resolve_repo_path(benchmark["reuse_preflight"])
                report = json.loads(reusable.read_text(encoding="utf-8"))
                if report.get("ok") is not True or int(report.get("tested_count", -1)) != 360 or int(report.get("failed_count", -1)) != 0:
                    raise RuntimeError("Reusable benchmark preflight is invalid")
                shutil.copyfile(reusable, preflight)
            else:
                _run_preflight(blender, source, domain_path, anomaly_path, domain, contract["resolution"], preflight)
        chunks = chunk_pairs(plan_rows, completed, contract["chunk_size_pairs"])
        if args.resume and run.get("status") == "complete" and not chunks:
            result = validate_anomaly_batch_run(run_dir, require_complete=True)
            print(json.dumps({"run_dir": str(run_dir), "validation": result}, indent=2))
            return
        wear_config = json.loads(refs_raw["surface_wear"].decode("utf-8"))
        reports = []
        for chunk_number, rows in enumerate(chunks):
            key = f"{rows[0]['pair_index']:03d}_{rows[-1]['pair_index']:03d}"
            chunk_plan = _prepare_chunk(rows, wear_config, run_dir / "_staging", key)
            report_path = chunk_plan.parent / "chunk_report.json"
            command = [
                str(blender), "--background", str(source), "--python-exit-code", "1",
                "--python", str(REPO_ROOT / "scripts/blender/render_anomaly_batch.py"), "--",
                "--batch-config", str(config_path), "--domain-config", str(domain_path),
                "--anomaly-config", str(anomaly_path), "--chunk-plan", str(chunk_plan),
                "--lighting-config", str(refs["lighting"]), "--wear-config", str(refs["surface_wear"]),
                "--pose-config", str(refs["camera_poses"]), "--sampling-config", str(refs["pose_sampling"]),
                "--run-dir", str(run_dir), "--manifest", str(manifest_path), "--chunk-report", str(report_path),
            ]
            print(f"Running anomaly chunk {chunk_number+1}/{len(chunks)}:", subprocess.list2cmdline(command), flush=True)
            for attempt in range(contract["max_chunk_retries"] + 1):
                try:
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
                    break
                except subprocess.CalledProcessError:
                    if attempt >= contract["max_chunk_retries"]:
                        raise
                    print("Chunk failed; reopening immutable source for one retry", flush=True)
            if _sha256(source) != source_sha:
                raise RuntimeError("Immutable source Blend changed")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("ok") is not True or int(report["source_open_count"]) != 1:
                raise RuntimeError("Invalid Blender chunk report")
            if int(report["render_call_count"]) != 2 * int(report["rendered_pair_count"]):
                raise RuntimeError("Two-render-per-pair contract failed")
            reports.append(report)
            manifest = _read_manifest_for_resume(manifest_path, run_dir / "_staging/recovery")
            run["counts"]["completed_pairs"] = len(manifest)
            run["updated_at"] = _utc_now()
            _atomic_json(run_path, run)

        manifest = _read_manifest_for_resume(manifest_path, run_dir / "_staging/recovery")
        run["status"] = "validating"
        run["counts"]["completed_pairs"] = len(manifest)
        committed_render_calls = 2 * len(manifest)
        committed_pair_seconds = sum(float(row.get("render", {}).get("total_seconds", 0.0)) for row in manifest)
        if reports:
            runtimes = [report["runtime"] for report in reports]
            if any(runtime != runtimes[0] for runtime in runtimes[1:]):
                raise RuntimeError("Blender runtime/device changed between chunks")
            run["runtime"] = runtimes[0]
            run["performance"] = {
                "wall_seconds": float(time.perf_counter() - started), "chunk_count": len(reports),
                "source_open_count": 1 + sum(int(report["source_open_count"]) for report in reports),
                # A resume may start after earlier pairs were atomically committed
                # but before run.json received its final performance summary.
                "render_call_count": committed_render_calls,
                "pair_pipeline_seconds": committed_pair_seconds,
            }
        elif "performance" not in run:
            raise RuntimeError("Completed resume lacks performance metadata")
        else:
            run["performance"]["render_call_count"] = committed_render_calls
            run["performance"]["pair_pipeline_seconds"] = committed_pair_seconds
        _atomic_json(run_path, run)
        validate_anomaly_batch_run(run_dir, require_complete=False)
        _contact_sheets(run_dir, manifest)
        run["status"] = "complete"
        run.pop("error", None)
        run.pop("failed_at", None)
        run["finished_at"] = _utc_now()
        run["source"]["sha256_after"] = _sha256(source)
        _atomic_json(run_path, run)
        result = validate_anomaly_batch_run(run_dir, require_complete=True)
        run["validation"] = result
        _atomic_json(run_path, run)
        print(json.dumps({"run_dir": str(run_dir), "fingerprint": fingerprint, "validation": result}, indent=2))
    except Exception as error:
        run["status"] = "failed"
        run["failed_at"] = _utc_now()
        run["error"] = f"{type(error).__name__}: {error}"
        run["counts"]["completed_pairs"] = len(_read_manifest_for_resume(manifest_path, run_dir / "_staging/recovery")) if manifest_path.exists() else 0
        _atomic_json(run_path, run)
        raise


if __name__ == "__main__":
    main()
