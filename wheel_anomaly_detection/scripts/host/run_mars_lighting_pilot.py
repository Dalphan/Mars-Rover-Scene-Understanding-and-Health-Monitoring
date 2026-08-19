"""Launch the controlled Mars-lighting pilot and build a metric contact sheet."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_metrics(path: Path) -> dict:
    image = Image.open(path).convert("RGB")
    image.thumbnail((320, 240), Image.Resampling.BILINEAR)
    pixels = list(image.get_flattened_data())
    hsv = [colorsys.rgb_to_hsv(*(value / 255.0 for value in pixel)) for pixel in pixels]
    luminance = [0.2126 * red + 0.7152 * green + 0.0722 * blue for red, green, blue in pixels]
    ordered_luminance = sorted(luminance)
    ordered_hue = sorted(value[0] * 360.0 for value in hsv)
    ordered_saturation = sorted(value[1] for value in hsv)
    quantile = lambda values, fraction: values[int((len(values) - 1) * fraction)]
    return {
        "mean_rgb": [sum(pixel[channel] for pixel in pixels) / len(pixels) for channel in range(3)],
        "median_hue_deg": quantile(ordered_hue, 0.5),
        "median_saturation": quantile(ordered_saturation, 0.5),
        "luminance_p05_p50_p95": [quantile(ordered_luminance, fraction) for fraction in (0.05, 0.5, 0.95)],
    }


def _contact_sheet(report: dict, output_path: Path, comparison_ids: list[str] | None = None) -> None:
    presets_by_id = {preset["id"]: preset for preset in report["presets"]}
    presets = [presets_by_id[identifier] for identifier in comparison_ids] if comparison_ids else report["presets"]
    width, height, header, columns = 720, 540, 92, 2
    rows = (len(presets) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, (height + header) * rows), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.load_default(size=24)
    detail_font = ImageFont.load_default(size=16)
    for index, preset in enumerate(presets):
        column, row = index % columns, index // columns
        x, y = column * width, row * (height + header)
        image = Image.open(preset["image"]).convert("RGB")
        image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + header))
        draw.text((x + 12, y + 8), preset["label"], font=title_font, fill=(245, 245, 245))
        metrics = preset["metrics"]
        rgb = "/".join(str(round(value)) for value in metrics["mean_rgb"])
        luma = metrics["luminance_p05_p50_p95"]
        draw.text((x + 12, y + 38), f"RGB {rgb} | hue {metrics['median_hue_deg']:.1f} | sat {metrics['median_saturation']:.2f}", font=detail_font, fill=(195, 195, 195))
        draw.text((x + 12, y + 61), f"luma p05/p50/p95 {luma[0]:.0f}/{luma[1]:.0f}/{luma[2]:.0f}", font=detail_font, fill=(175, 175, 175))
    sheet.save(output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--source-blend", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "pose_sampling" / "wheel_roll_pose_sampling.blend")
    parser.add_argument("--pilot-config", type=Path, default=repo_root / "configs" / "blender" / "mars_lighting_pilot.json")
    parser.add_argument("--pose-config", type=Path, default=repo_root / "configs" / "blender" / "wheel_camera_poses.json")
    parser.add_argument("--sampling-config", type=Path, default=repo_root / "configs" / "blender" / "wheel_pose_sampling.json")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "lighting_pilot")
    parser.add_argument("--reference-image", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = args.source_blend.resolve()
    source_sha256 = _sha256(source)
    script = repo_root / "scripts" / "blender" / "render_mars_lighting_pilot.py"
    command = [
        str(args.blender_executable.resolve()), "--background", str(source), "--python", str(script), "--",
        "--source-blend", str(source), "--pilot-config", str(args.pilot_config.resolve()),
        "--pose-config", str(args.pose_config.resolve()), "--sampling-config", str(args.sampling_config.resolve()),
        "--output-dir", str(output),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    started = time.time()
    subprocess.run(command, cwd=repo_root, check=True)
    report_path = output / "lighting_pilot.json"
    if not report_path.is_file() or report_path.stat().st_mtime < started:
        raise RuntimeError("Blender did not produce a fresh lighting-pilot report")
    if _sha256(source) != source_sha256:
        raise RuntimeError("Pose-sampling source Blend changed during the lighting pilot")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_sha256"] = source_sha256
    if not bool(report.get("validation", {}).get("ok", False)):
        raise RuntimeError("Lighting pilot frame-consistency gate failed")
    for preset in report["presets"]:
        preset["metrics"] = _image_metrics(Path(preset["image"]))
    if args.reference_image:
        report["reference"] = {"image": str(args.reference_image.resolve()), "metrics": _image_metrics(args.reference_image.resolve())}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pilot_config = json.loads(args.pilot_config.resolve().read_text(encoding="utf-8"))
    contact_sheet = output / "refinement_contact_sheet.png"
    _contact_sheet(report, contact_sheet, pilot_config.get("comparison_ids"))
    print(f"Contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
