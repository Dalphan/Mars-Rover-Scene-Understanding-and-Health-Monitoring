"""Build and render a stratified preview of the final domain-randomization sampler."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.host.run_wheel_surface_wear_pilot import _generate_nested_masks
from src.wheel_preparation.domain_randomization import (
    preview_tokens,
    sample_domain_randomization,
    select_stratified_preview,
    validate_domain_randomization_config,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_plan(config: dict, wear_config: dict, output_dir: Path, candidates: list[dict]) -> dict:
    samples = select_stratified_preview(config, candidates=candidates)
    mask_dir = output_dir / "wear_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for sample in samples:
        sample = copy.deepcopy(sample)
        if sample["surface_wear"] != "surface_current":
            per_sample = copy.deepcopy(wear_config)
            per_sample["mask_generation"]["seed"] = int(sample["wear_seed"])
            for variant in per_sample["variants"]:
                if bool(variant["wear_enabled"]):
                    variant["mask_filename"] = f"{sample['sample_id']}_{variant['id']}.png"
            masks = _generate_nested_masks(per_sample, mask_dir)
            sample["wear_mask"] = masks[sample["surface_wear"]]["path"]
            sample["wear_mask_sha256"] = masks[sample["surface_wear"]]["sha256"]
            sample["wear_mask_coverage"] = masks[sample["surface_wear"]]["coverage"]
        else:
            sample["wear_mask"] = None
            sample["wear_mask_sha256"] = None
            sample["wear_mask_coverage"] = 0.0
        prepared.append(sample)
    return {
        "schema_version": 1,
        "selection": "stratified subset of genuine weighted samples; counts do not represent production frequencies",
        "samples": prepared,
    }


def _coverage(report: dict, config: dict) -> None:
    categories = config["preview"]["stratify_categories"]
    actual = set().union(*(preview_tokens(sample, categories) for sample in report["samples"]))
    expected = set()
    for category, distribution in (
        ("lighting", config["distributions"]["lighting"]),
        ("surface_wear", config["distributions"]["surface_wear"]),
        ("camera_pose", config["distributions"]["camera_pose"]),
        ("target_wheel", config["distributions"]["target_wheel"]),
    ):
        if category in categories:
            expected.update((category, key) for key in distribution)
    if actual != expected:
        raise RuntimeError(f"Rendered preview lost stratified coverage: {sorted(expected - actual)}")


def _contact_sheet(report: dict, config: dict, output_path: Path) -> None:
    samples = report["samples"]
    columns = int(config["preview"]["contact_sheet_columns"])
    width, height, header = 600, 450, 116
    rows = (len(samples) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, (height + header) * rows), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=20)
    detail_font = ImageFont.load_default(size=14)
    for index, sample in enumerate(samples):
        x = (index % columns) * width
        y = (index // columns) * (height + header)
        image = Image.open(sample["image"]).convert("RGB")
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + header))
        wheel = sample["target_wheel"].removeprefix("wheel_")
        pose = sample["camera_pose"].split("_")[0]
        lighting = sample["lighting_preset"].removeprefix("mars_").removesuffix("_refined")
        wear = sample["surface_wear"].replace("surface_", "").replace("wear_", "")
        light = sample["lighting_jitter"]
        camera = sample["camera_jitter"]
        position_cm = 100.0 * sum(value * value for value in camera["position_basis_m"]) ** 0.5
        aim = (camera["aim_yaw_degrees"] ** 2 + camera["aim_pitch_degrees"] ** 2) ** 0.5
        draw.text((x + 10, y + 7), f"{sample['sample_id']} | {wheel} | {pose}", font=title_font, fill=(245, 245, 245))
        draw.text((x + 10, y + 35), f"{lighting} | wear {wear} | roll {sample['healthy_roll_degrees']:.0f} deg", font=detail_font, fill=(205, 205, 205))
        draw.text((x + 10, y + 58), f"sun dAz {light['sun_azimuth_degrees']:+.1f} dEl {light['sun_elevation_degrees']:+.1f} | exp {light['exposure_ev']:+.2f} EV", font=detail_font, fill=(185, 185, 185))
        draw.text((x + 10, y + 81), f"cam {position_cm:.1f} cm | aim {aim:.2f} deg | focal x{camera['focal_length_scale']:.3f} | terrain {sample['terrain_footprint']['minimum_edge_margin_m']:.2f} m", font=detail_font, fill=(170, 170, 170))
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "blender" / "domain_randomization.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_domain_randomization_config(config)
    source = (REPO_ROOT / config["source_blend"]).resolve()
    output = (REPO_ROOT / config["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    wear_config_path = (REPO_ROOT / config["wear_config"]).resolve()
    wear_config = json.loads(wear_config_path.read_text(encoding="utf-8"))
    source_sha256 = _sha256(source)
    candidates = [
        sample_domain_randomization(config, index)
        for index in range(int(config["preview"]["candidate_pool_size"]))
    ]
    candidate_plan_path = output / "domain_randomization_candidate_plan.json"
    candidate_plan_path.write_text(json.dumps({"schema_version": 1, "samples": candidates}, indent=2), encoding="utf-8")
    preflight_path = output / "domain_randomization_preflight.json"
    preflight_script = REPO_ROOT / "scripts" / "blender" / "audit_domain_randomization_candidates.py"
    preflight_command = [
        str(args.blender_executable.resolve()), "--background", str(source), "--python", str(preflight_script), "--",
        "--config", str(config_path), "--plan", str(candidate_plan_path),
        "--pose-config", str((REPO_ROOT / config["pose_config"]).resolve()),
        "--sampling-config", str((REPO_ROOT / config["sampling_config"]).resolve()),
        "--output", str(preflight_path),
    ]
    print("Running preflight:", subprocess.list2cmdline(preflight_command), flush=True)
    preflight_started = time.time()
    subprocess.run(preflight_command, cwd=REPO_ROOT, check=True)
    if not preflight_path.is_file() or preflight_path.stat().st_mtime < preflight_started:
        raise RuntimeError("Blender did not produce a fresh domain-randomization preflight")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    valid_indices = {int(entry["sample_index"]) for entry in preflight["results"] if bool(entry["ok"])}
    valid_candidates = [sample for sample in candidates if sample["sample_index"] in valid_indices]
    plan = _prepare_plan(config, wear_config, output, valid_candidates)
    plan_path = output / "domain_randomization_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    script = REPO_ROOT / "scripts" / "blender" / "render_domain_randomization_preview.py"
    command = [
        str(args.blender_executable.resolve()), "--background", str(source), "--python", str(script), "--",
        "--source-blend", str(source), "--config", str(config_path), "--plan", str(plan_path),
        "--lighting-config", str((REPO_ROOT / config["lighting_config"]).resolve()),
        "--wear-config", str(wear_config_path),
        "--pose-config", str((REPO_ROOT / config["pose_config"]).resolve()),
        "--sampling-config", str((REPO_ROOT / config["sampling_config"]).resolve()),
        "--output-dir", str(output),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    report_path = output / "domain_randomization_preview.json"
    if not report_path.is_file() or report_path.stat().st_mtime < started:
        raise RuntimeError("Blender did not produce a fresh domain-randomization report")
    if _sha256(source) != source_sha256:
        raise RuntimeError("Pose-sampling source Blend changed during domain-randomization preview")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("validation", {}).get("ok", False)):
        raise RuntimeError("Domain-randomization Blender validation failed")
    _coverage(report, config)
    report["source_sha256"] = source_sha256
    report["config"] = str(config_path)
    report["config_sha256"] = _sha256(config_path)
    report["plan"] = str(plan_path)
    report["plan_sha256"] = _sha256(plan_path)
    report["preview_selection_bias"] = plan["selection"]
    report["preflight"] = {
        "path": str(preflight_path),
        "candidate_count": int(preflight["candidate_count"]),
        "valid_count": int(preflight["valid_count"]),
        "valid_fraction": float(preflight["valid_count"]) / float(preflight["candidate_count"]),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    contact_sheet = output / "domain_randomization_contact_sheet.png"
    _contact_sheet(report, config, contact_sheet)
    print(f"Contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
