"""Generate deterministic wear masks, run Blender, and build a contact sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coverage(image: Image.Image, threshold: int) -> float:
    histogram = image.histogram()
    return sum(histogram[threshold:]) / float(image.width * image.height)


def _draw_scratch(draw: ImageDraw.ImageDraw, rng: random.Random, settings: dict, size: int) -> None:
    angle = math.radians(
        rng.uniform(
            float(settings["dominant_direction_degrees"]) - float(settings["direction_spread_degrees"]),
            float(settings["dominant_direction_degrees"]) + float(settings["direction_spread_degrees"]),
        )
    )
    if rng.random() < float(settings["cross_scratch_probability"]):
        angle += math.pi / 2.0
    length = rng.randint(*map(int, settings["length_pixels"]))
    width = rng.randint(*map(int, settings["width_pixels"]))
    intensity = rng.randint(*map(int, settings["intensity"]))
    start_x = rng.randint(0, size - 1)
    start_y = rng.randint(0, size - 1)
    segments = rng.randint(2, 5)
    points = []
    for index in range(segments + 1):
        distance = length * index / segments
        sideways = rng.uniform(-0.045, 0.045) * length if 0 < index < segments else 0.0
        x = start_x + math.cos(angle) * distance - math.sin(angle) * sideways
        y = start_y + math.sin(angle) * distance + math.cos(angle) * sideways
        points.append((round(x), round(y)))
    draw.line(points, fill=intensity, width=width, joint="curve")


def _generate_nested_masks(config: dict, output_dir: Path) -> dict[str, dict]:
    settings = config["mask_generation"]
    size = int(settings["resolution"])
    threshold = int(settings["coverage_threshold"])
    rng = random.Random(int(settings["seed"]))
    image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(image)
    enabled = sorted(
        (variant for variant in config["variants"] if bool(variant["wear_enabled"])),
        key=lambda variant: float(variant["target_mask_coverage"]),
    )
    results = {}
    scratches = 0
    for variant in enabled:
        target = float(variant["target_mask_coverage"])
        while _coverage(image, threshold) < target:
            for _ in range(24):
                _draw_scratch(draw, rng, settings, size)
                scratches += 1
        path = output_dir / variant["mask_filename"]
        image.save(path, optimize=True)
        results[variant["id"]] = {
            "path": str(path),
            "sha256": _sha256(path),
            "coverage": _coverage(image, threshold),
            "target_coverage": target,
            "scratch_count_prefix": scratches,
            "nested_prefix": True,
        }
    return results


def _difference_metrics(reference: Path, candidate: Path) -> dict:
    left = Image.open(reference).convert("RGB")
    right = Image.open(candidate).convert("RGB")
    left.thumbnail((400, 300), Image.Resampling.BILINEAR)
    right.thumbnail((400, 300), Image.Resampling.BILINEAR)
    difference = ImageChops.difference(left, right)
    histogram = difference.histogram()
    channel_pixels = left.width * left.height
    mean_absolute = sum((index % 256) * count for index, count in enumerate(histogram)) / float(channel_pixels * 3)
    grayscale = difference.convert("L")
    changed = sum(grayscale.histogram()[3:]) / float(channel_pixels)
    return {"mean_absolute_rgb_delta": mean_absolute, "changed_pixel_fraction_above_2": changed}


def _contact_sheet(report: dict, output_path: Path) -> None:
    variants = report["variants"]
    width, height, header = 540, 405, 82
    sheet = Image.new("RGB", (width * len(variants), height + header), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=23)
    detail_font = ImageFont.load_default(size=15)
    for index, variant in enumerate(variants):
        x = index * width
        image = Image.open(variant["image"]).convert("RGB")
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        sheet.paste(image, (x, header))
        draw.text((x + 12, 8), variant["label"], font=title_font, fill=(245, 245, 245))
        if variant["wear_enabled"]:
            draw.text(
                (x + 12, 42),
                f"mask {variant['mask_metrics']['coverage'] * 100:.2f}% | delta RGB {variant['difference_from_current']['mean_absolute_rgb_delta']:.2f}",
                font=detail_font,
                fill=(190, 190, 190),
            )
        else:
            draw.text((x + 12, 42), "baseline dusty refined", font=detail_font, fill=(190, 190, 190))
    sheet.save(output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=repo_root / "configs" / "blender" / "wheel_surface_wear_pilot.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = (repo_root / config["source_blend"]).resolve()
    output = (repo_root / config["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mask_metrics = _generate_nested_masks(config, output)
    source_sha256 = _sha256(source)
    script = repo_root / "scripts" / "blender" / "render_wheel_surface_wear_pilot.py"
    command = [
        str(args.blender_executable.resolve()),
        "--background",
        str(source),
        "--python",
        str(script),
        "--",
        "--source-blend",
        str(source),
        "--wear-config",
        str(config_path),
        "--lighting-config",
        str((repo_root / config["lighting_config"]).resolve()),
        "--pose-config",
        str((repo_root / config["pose_config"]).resolve()),
        "--sampling-config",
        str((repo_root / config["sampling_config"]).resolve()),
        "--output-dir",
        str(output),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=repo_root, check=True)
    report_path = output / "surface_wear_pilot.json"
    if not report_path.is_file() or report_path.stat().st_mtime < started:
        raise RuntimeError("Blender did not produce a fresh surface-wear report")
    if _sha256(source) != source_sha256:
        raise RuntimeError("Pose-sampling source Blend changed during surface-wear pilot")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("validation", {}).get("ok", False)):
        raise RuntimeError("Surface-wear Blender validation failed")
    current = Path(next(entry for entry in report["variants"] if entry["id"] == "surface_current")["image"])
    for variant in report["variants"]:
        if variant["wear_enabled"]:
            variant["mask_metrics"] = mask_metrics[variant["id"]]
            variant["difference_from_current"] = _difference_metrics(current, Path(variant["image"]))
    weights = report["dataset_sampling"]["weights"]
    if abs(sum(map(float, weights.values())) - 1.0) > 1e-9:
        raise RuntimeError("Dataset wear weights must sum to one")
    report["source_sha256"] = source_sha256
    report["config"] = str(config_path)
    report["config_sha256"] = _sha256(config_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    contact_sheet = output / "surface_wear_contact_sheet.png"
    _contact_sheet(report, contact_sheet)
    print(f"Contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
