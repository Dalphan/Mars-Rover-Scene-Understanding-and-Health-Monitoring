"""Launch the Blender wheel-camera pilot and assemble a labeled contact sheet."""

from __future__ import annotations

import argparse
import colorsys
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _terrain_color_diagnostic(image_path: Path) -> dict:
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((240, 180), Image.Resampling.BILINEAR)
    pixels = list(image.get_flattened_data())
    selected = [
        (red, green, blue)
        for red, green, blue in pixels
        if red >= green * 1.08 and green >= blue * 1.03 and 55 <= max(red, green, blue) <= 250
    ]
    fraction = len(selected) / len(pixels)
    if not selected:
        return {"ok": False, "terrain_like_fraction": 0.0, "mean_rgb": [0.0, 0.0, 0.0]}
    mean_rgb = [sum(pixel[channel] for pixel in selected) / len(selected) for channel in range(3)]
    mean_hue = sum(colorsys.rgb_to_hsv(*(value / 255.0 for value in pixel))[0] for pixel in selected) / len(selected)
    return {
        "ok": fraction >= 0.20 and mean_rgb[0] > mean_rgb[1] > mean_rgb[2],
        "terrain_like_fraction": fraction,
        "mean_rgb": mean_rgb,
        "mean_hue_degrees": mean_hue * 360.0,
    }


def _contact_sheet(report: dict, output_path: Path) -> None:
    cell_width, image_height, header_height = 800, 600, 54
    columns = 3
    rows = (len(report["poses"]) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * (image_height + header_height)), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=24)
    detail_font = ImageFont.load_default(size=17)
    for index, pose in enumerate(report["poses"]):
        column, row = index % columns, index // columns
        x, y = column * cell_width, row * (image_height + header_height)
        image = Image.open(pose["image"]).convert("RGB")
        image = ImageOps.fit(image, (cell_width, image_height), method=Image.Resampling.LANCZOS)
        sheet.paste(image, (x, y + header_height))
        draw.text((x + 12, y + 5), pose["label"], font=font, fill=(245, 245, 245))
        detail = f"{pose['distance_m']:.2f} m | elev {pose['elevation_deg']:.0f}° | tangent {pose['tangential_deg']:+.0f}°"
        draw.text((x + 12, y + 31), detail, font=detail_font, fill=(185, 185, 185))
    sheet.save(output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender-executable", type=Path, required=True)
    parser.add_argument("--pose-config", type=Path, default=repo_root / "configs" / "blender" / "wheel_camera_poses.json")
    parser.add_argument("--source-blend", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "microterrain" / "level3" / "microterrain_L3.blend")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "anomaly_detection_2" / "camera_pose_pilot")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    pose_config = json.loads(args.pose_config.resolve().read_text(encoding="utf-8"))
    active_pose_ids = {str(pose["id"]) for pose in pose_config["poses"]}
    previous_report_path = output_dir / "camera_pose_pilot.json"
    if previous_report_path.exists():
        previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
        for pose in previous_report.get("poses", []):
            stale_path = Path(pose["image"]).resolve()
            if (
                str(pose.get("id")) not in active_pose_ids
                and stale_path.parent == output_dir
                and stale_path.is_file()
            ):
                stale_path.unlink()
    script = repo_root / "scripts" / "blender" / "render_wheel_camera_pose_pilot.py"
    command = [
        str(args.blender_executable.resolve()), "--background", str(args.source_blend.resolve()), "--python", str(script), "--",
        "--pose-config", str(args.pose_config.resolve()), "--source-blend", str(args.source_blend.resolve()), "--output-dir", str(output_dir),
    ]
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)
    report_path = output_dir / "camera_pose_pilot.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not bool(report.get("validation", {}).get("ok", False)):
        errors = ", ".join(report.get("validation", {}).get("framing_errors", []))
        raise RuntimeError(f"Blender framing validation failed: {errors}")
    terrain_diagnostics = {
        pose["id"]: _terrain_color_diagnostic(Path(pose["image"]))
        for pose in report["poses"]
        if pose["crop_policy"] == "full_wheel"
    }
    terrain_color_ok = bool(terrain_diagnostics) and all(item["ok"] for item in terrain_diagnostics.values())
    report["validation"]["terrain_color"] = {
        "ok": terrain_color_ok,
        "method": "downsampled sRGB red-brown dominance gate on full-wheel views",
        "poses": terrain_diagnostics,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not terrain_color_ok:
        raise RuntimeError("Terrain color validation failed")
    _contact_sheet(report, output_dir / "contact_sheet.png")
    print(f"Contact sheet: {output_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
