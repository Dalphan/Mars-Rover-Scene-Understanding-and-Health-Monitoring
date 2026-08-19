"""Create georeferenced DTM/color crops and select a validated terrain texture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deps-dir", type=Path)
    args = parser.parse_args()
    if args.deps_dir:
        sys.path.insert(0, str(args.deps_dir.resolve()))
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bilinear(array, column, row, np):
    column = np.clip(column, 0.0, array.shape[1] - 1.0)
    row = np.clip(row, 0.0, array.shape[0] - 1.0)
    c0, r0 = np.floor(column).astype(int), np.floor(row).astype(int)
    c1, r1 = np.minimum(c0 + 1, array.shape[1] - 1), np.minimum(r0 + 1, array.shape[0] - 1)
    wc, wr = column - c0, row - r0
    return array[r0, c0] * (1 - wc) * (1 - wr) + array[r0, c1] * wc * (1 - wr) + array[r1, c0] * (1 - wc) * wr + array[r1, c1] * wc * wr


def _label_number(text: str, keyword: str) -> float:
    match = re.search(rf"^\s*{re.escape(keyword)}\s*=\s*([-+0-9.eE]+)", text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Missing {keyword} in product label")
    return float(match.group(1))


def _north_up(transform) -> bool:
    return transform.a > 0 and transform.e < 0 and abs(transform.b) < 1e-12 and abs(transform.d) < 1e-12


def _source_metadata(source) -> dict:
    return {
        "driver": source.driver,
        "crs_wkt": source.crs.to_wkt(),
        "crs_parameters": source.crs.to_dict(),
        "pixel_size_m": [abs(float(source.transform.a)), abs(float(source.transform.e))],
        "extent_m": list(map(float, source.bounds)),
        "dimensions": [source.height, source.width],
        "bands": source.count,
        "dtype": list(source.dtypes),
        "nodata": source.nodata,
        "orientation": "north-up" if _north_up(source.transform) else "rotated_or_flipped",
        "transform": list(source.transform)[:6],
    }


def _luma(rgb, np):
    return 0.2126 * rgb[0].astype(np.float32) + 0.7152 * rgb[1].astype(np.float32) + 0.0722 * rgb[2].astype(np.float32)


def _gradient_correlation(reference_rgb, candidate_rgb, search_pixels: int, np) -> dict:
    reference = _luma(reference_rgb, np)
    candidate = _luma(candidate_rgb, np)
    ref_gradient = np.hypot(*np.gradient(reference))
    candidate_gradient = np.hypot(*np.gradient(candidate))
    best = {"score": -1.0, "shift_pixels": [0, 0]}
    zero_score = None
    margin = search_pixels + 2
    for row_shift in range(-search_pixels, search_pixels + 1):
        for column_shift in range(-search_pixels, search_pixels + 1):
            ref = ref_gradient[margin:-margin, margin:-margin]
            cand = candidate_gradient[margin + row_shift : candidate_gradient.shape[0] - margin + row_shift, margin + column_shift : candidate_gradient.shape[1] - margin + column_shift]
            ref_flat, cand_flat = ref.ravel(), cand.ravel()
            if ref_flat.std() == 0 or cand_flat.std() == 0:
                score = -1.0
            else:
                score = float(np.corrcoef(ref_flat, cand_flat)[0, 1])
            if row_shift == 0 and column_shift == 0:
                zero_score = score
            if score > best["score"]:
                best = {"score": score, "shift_pixels": [column_shift, -row_shift]}
    best["zero_shift_score"] = zero_score
    return best


def _write_rgb_geotiff(path, array, transform, crs, rasterio) -> None:
    with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1], count=3, dtype="uint8", transform=transform, crs=crs, nodata=0, compress="deflate") as destination:
        destination.write(array)


def _mars_rgb_from_irb(irb_reflectance, mrgb, mrgb_color_mask, percentiles, np):
    """Build HiRISE synthetic RGB and match only the MRGB color-swath palette."""
    red = irb_reflectance[1]
    blue_green = irb_reflectance[2]
    synthetic = np.stack((red, blue_green, np.clip(2.0 * blue_green - 0.3 * red, 0.0, None)))
    low_percentile, high_percentile = map(float, percentiles)
    synthetic_luma = 0.2126 * synthetic[0] + 0.7152 * synthetic[1] + 0.0722 * synthetic[2]
    source_low, source_high = np.percentile(synthetic_luma, [low_percentile, high_percentile])
    normalized = np.clip((synthetic_luma - source_low) / max(source_high - source_low, 1e-12), 0.0, 1.0)
    output = np.empty_like(mrgb, dtype=np.uint8)
    channel_report = []
    for channel in range(3):
        target_values = mrgb[channel][mrgb_color_mask]
        target_low, target_high = np.percentile(target_values, [low_percentile, high_percentile])
        output[channel] = np.rint(target_low + normalized * (target_high - target_low)).astype(np.uint8)
        channel_report.append({"shared_synthetic_luma_percentiles": [float(source_low), float(source_high)], "target_mrgb_dn_percentiles": [float(target_low), float(target_high)]})
    return output, channel_report


def main() -> None:
    args = _arguments()
    import numpy as np
    import rasterio
    from PIL import Image
    from pyproj import CRS, Transformer
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src.gale_terrain.core import CropBounds, crs_parameters_equivalent, validate_grid_contract
    from src.gale_terrain.validation import validate_crop_report

    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract = validate_grid_contract(config)
    bounds: CropBounds = contract["bounds"]
    product = config["product"]
    raw_dir = args.data_root / product["id"] / "raw"
    by_role = {item["role"]: item for item in product["files"]}
    paths = {role: raw_dir / item["name"] for role, item in by_role.items()}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtm_shape = tuple(config["crop"]["dtm_shape"])
    color_shape = tuple(config["crop"]["ortho_shape"])
    dtm_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, dtm_shape[1], dtm_shape[0])
    color_transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, color_shape[1], color_shape[0])

    with rasterio.open(paths["dtm"]) as dtm_source, rasterio.open(paths["irb_ortho"]) as irb_source, rasterio.open(paths["mrgb"]) as mrgb_source:
        source_metadata = {"dtm": _source_metadata(dtm_source), "irb_ortho": _source_metadata(irb_source), "mrgb": _source_metadata(mrgb_source)}
        dtm_irb_equivalent = crs_parameters_equivalent(dtm_source.crs.to_dict(), irb_source.crs.to_dict())
        if not dtm_irb_equivalent:
            raise RuntimeError("DTM and coregistered IRB ortho CRS are not numerically equivalent")
        dtm = np.full(dtm_shape, np.nan, dtype=np.float32)
        reproject(rasterio.band(dtm_source, 1), dtm, src_transform=dtm_source.transform, src_crs=dtm_source.crs, src_nodata=dtm_source.nodata, dst_transform=dtm_transform, dst_crs=dtm_source.crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
        irb = np.zeros((3, *color_shape), dtype=np.uint8)
        mrgb = np.zeros((3, *color_shape), dtype=np.uint8)
        for band in range(3):
            reproject(rasterio.band(irb_source, band + 1), irb[band], src_transform=irb_source.transform, src_crs=irb_source.crs, src_nodata=irb_source.nodata, dst_transform=color_transform, dst_crs=dtm_source.crs, dst_nodata=0, resampling=Resampling.bilinear)
            reproject(rasterio.band(mrgb_source, band + 1), mrgb[band], src_transform=mrgb_source.transform, src_crs=mrgb_source.crs, src_nodata=mrgb_source.nodata, dst_transform=color_transform, dst_crs=dtm_source.crs, dst_nodata=0, resampling=Resampling.bilinear)
        mrgb_extent_in_irb_crs = list(map(float, transform_bounds(mrgb_source.crs, dtm_source.crs, *mrgb_source.bounds, densify_pts=21)))
        crs = dtm_source.crs

    valid_dtm = np.isfinite(dtm)
    valid_irb = np.any(irb != 0, axis=0)
    valid_mrgb = np.any(mrgb != 0, axis=0)
    mrgb_chroma = mrgb.max(axis=0).astype(np.int16) - mrgb.min(axis=0).astype(np.int16)
    mrgb_color_mask = valid_mrgb & (mrgb_chroma > int(config["color_coregistration"]["mrgb_color_chroma_threshold_dn"]))
    mrgb_color_fraction = float(mrgb_color_mask.mean())
    minimum_fraction = float(config["crop"]["minimum_valid_fraction"])
    if valid_dtm.mean() < minimum_fraction or valid_irb.mean() < minimum_fraction:
        raise RuntimeError("Configured crop does not satisfy the DTM/IRB NoData gate")

    # Compare both products on a common one-metre grid after standards-based CRS
    # reprojection only. No empirical translation is ever applied to the output.
    comparison_step = int(round(float(config["color_coregistration"]["comparison_resolution_m"]) / contract["ortho_resolution_m"][0]))
    search_pixels = int(round(float(config["color_coregistration"]["maximum_search_shift_m"]) / float(config["color_coregistration"]["comparison_resolution_m"])))
    correlation = _gradient_correlation(irb[:, ::comparison_step, ::comparison_step], mrgb[:, ::comparison_step, ::comparison_step], search_pixels, np)
    best_shift_m = [value * float(config["color_coregistration"]["comparison_resolution_m"]) for value in correlation["shift_pixels"]]
    orientations_ok = all(source_metadata[name]["orientation"] == "north-up" for name in ("dtm", "irb_ortho", "mrgb"))
    coverage_ok = float(valid_mrgb.mean()) >= minimum_fraction
    full_color_coverage_ok = mrgb_color_fraction >= float(config["color_coregistration"]["minimum_mrgb_color_coverage"])
    shift_ok = max(map(abs, best_shift_m)) <= float(config["color_coregistration"]["maximum_accepted_shift_m"])
    correlation_ok = correlation["score"] >= float(config["color_coregistration"]["minimum_gradient_correlation"])
    mrgb_passed = orientations_ok and coverage_ok and full_color_coverage_ok and shift_ok and correlation_ok

    dtm_tif = args.output_dir / "dtm.tif"
    irb_tif = args.output_dir / "irb.tif"
    mrgb_tif = args.output_dir / "mrgb_candidate.tif"
    with rasterio.open(dtm_tif, "w", driver="GTiff", width=dtm_shape[1], height=dtm_shape[0], count=1, dtype="float32", transform=dtm_transform, crs=crs, nodata=np.nan, compress="deflate") as destination:
        destination.write(dtm, 1)
    _write_rgb_geotiff(irb_tif, irb, color_transform, crs, rasterio)
    _write_rgb_geotiff(mrgb_tif, mrgb, color_transform, crs, rasterio)

    label_text = paths["irb_label"].read_text(encoding="ascii")
    radiometric_offset = _label_number(label_text, "OFFSET")
    radiometric_scale = _label_number(label_text, "SCALING_FACTOR")
    reflectance = np.clip(irb.astype(np.float32) * radiometric_scale + radiometric_offset, 0.0, 1.0)
    irb_srgb = np.where(reflectance <= 0.0031308, reflectance * 12.92, 1.055 * np.power(reflectance, 1.0 / 2.4) - 0.055)
    irb_display = np.rint(np.clip(irb_srgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    mars_rgb, palette_channels = _mars_rgb_from_irb(reflectance, mrgb, mrgb_color_mask, config["color_coregistration"]["palette_percentiles"], np)
    Image.fromarray(np.moveaxis(irb, 0, -1), mode="RGB").save(args.output_dir / "irb_color.png")
    Image.fromarray(np.moveaxis(irb_display, 0, -1), mode="RGB").save(args.output_dir / "irb_reflectance_srgb.png")
    Image.fromarray(np.moveaxis(mrgb, 0, -1), mode="RGB").save(args.output_dir / "mrgb_candidate.png")
    Image.fromarray(np.moveaxis(mars_rgb, 0, -1), mode="RGB").save(args.output_dir / "irb_mars_rgb.png")
    chosen_array = mrgb if mrgb_passed else mars_rgb
    chosen = "MRGB" if mrgb_passed else "IRB"
    color_type = "MRGB visual RGB" if mrgb_passed else "Mars-style synthetic RGB derived from coregistered IRB"
    texture_path = args.output_dir / "terrain_color.png"
    Image.fromarray(np.moveaxis(chosen_array, 0, -1), mode="RGB").save(texture_path)

    point_count = int(config["crop"]["heightfield_shape"][0])
    projected_x = np.linspace(bounds.left, bounds.right, point_count, dtype=np.float64)
    projected_y = np.linspace(bounds.bottom, bounds.top, point_count, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(projected_x, projected_y)
    x_res, y_res = contract["dtm_resolution_m"]
    absolute_z = _bilinear(dtm, (grid_x - bounds.left) / x_res - 0.5, (bounds.top - grid_y) / y_res - 0.5, np).astype(np.float32)
    center_index = point_count // 2
    origin_elevation = float(absolute_z[center_index, center_index])
    local_x = (projected_x - bounds.center[0]).astype(np.float32)
    local_y = (projected_y - bounds.center[1]).astype(np.float32)
    local_z = (absolute_z - origin_elevation).astype(np.float32)
    np.savez_compressed(args.output_dir / "terrain_heightfield.npz", x_local=local_x, y_local=local_y, z_local=local_z, projected_bounds=np.asarray([bounds.left, bounds.bottom, bounds.right, bounds.top]), projected_origin=np.asarray([bounds.center[0], bounds.center[1], origin_elevation]), crs_wkt=np.asarray(crs.to_wkt()), vertical_exaggeration=np.asarray(1.0))

    pyproj_crs = CRS.from_wkt(crs.to_wkt())
    center_lon, center_lat = Transformer.from_crs(pyproj_crs, pyproj_crs.geodetic_crs, always_xy=True).transform(*bounds.center)
    sources = {role: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for role, path in paths.items()}
    report = {
        "schema_version": 2,
        "product_id": product["id"],
        "stereo_observations": product["stereo_observations"],
        "sources": sources,
        "source_metadata": source_metadata,
        "crs": {"wkt": crs.to_wkt(), "projected_name": pyproj_crs.name, "center_longitude_deg": center_lon, "center_latitude_deg": center_lat},
        "crop": {
            "projected_bounds_m": [bounds.left, bounds.bottom, bounds.right, bounds.top], "size_m": [bounds.width, bounds.height],
            "dtm_shape": list(dtm_shape), "dtm_resolution_m": list(contract["dtm_resolution_m"]), "color_shape": list(color_shape), "color_resolution_m": list(contract["ortho_resolution_m"]),
            "dtm_valid_fraction": float(valid_dtm.mean()), "irb_valid_fraction": float(valid_irb.mean()), "mrgb_valid_fraction": float(valid_mrgb.mean()),
            "min_elevation_m": float(np.nanmin(dtm)), "max_elevation_m": float(np.nanmax(dtm)), "origin_elevation_m": origin_elevation, "vertical_exaggeration": 1.0,
        },
        "coregistration": {
            "dtm_irb_crs_equivalent": dtm_irb_equivalent, "exact_shared_bounds": True, "integer_resolution_ratio": contract["resolution_ratio"] == 4, "resolution_ratio": contract["resolution_ratio"],
            "uv_convention": "U=east normalized over crop; V=north normalized over crop",
            "mrgb": {
                "passed": mrgb_passed, "method": "official CRS reprojection followed by gradient cross-correlation; no empirical transform applied",
                "native_crs_differs_from_irb": source_metadata["mrgb"]["crs_parameters"] != source_metadata["irb_ortho"]["crs_parameters"],
                "extent_in_irb_crs_m": mrgb_extent_in_irb_crs, "orientation_passed": orientations_ok, "crop_coverage_passed": coverage_ok,
                "valid_fraction": float(valid_mrgb.mean()), "comparison_resolution_m": float(config["color_coregistration"]["comparison_resolution_m"]),
                "merged_product_color_fraction": mrgb_color_fraction, "color_chroma_threshold_dn": int(config["color_coregistration"]["mrgb_color_chroma_threshold_dn"]), "minimum_color_coverage": float(config["color_coregistration"]["minimum_mrgb_color_coverage"]), "full_color_coverage_passed": full_color_coverage_ok,
                "best_gradient_correlation": correlation["score"], "zero_shift_gradient_correlation": correlation["zero_shift_score"], "minimum_gradient_correlation": float(config["color_coregistration"]["minimum_gradient_correlation"]), "correlation_passed": correlation_ok,
                "best_shift_m": best_shift_m, "maximum_accepted_shift_m": float(config["color_coregistration"]["maximum_accepted_shift_m"]), "shift_passed": shift_ok,
            },
        },
        "texture_selection": {"chosen": chosen, "color_type": color_type, "output_png": str(texture_path), "reason": "MRGB passed every coregistration gate" if mrgb_passed else "MRGB merged product does not provide complete usable color over the crop and failed the structural gate; geometry comes only from the guaranteed-coregistered IRB ortho"},
        "visualization": {"method": "HiRISE synthetic RGB from IRB, with robust palette range taken from the valid color portion of the same-observation MRGB", "spatial_source": "IRB only", "mrgb_spatial_data_used": False, "synthetic_rgb_equation": {"R": "RED", "G": "BG", "B": "max(2*BG - 0.3*RED, 0)"}, "palette_percentiles": config["color_coregistration"]["palette_percentiles"], "channel_mapping_report": palette_channels},
        "radiometry": {"irb_equation": "I/F = DN * SCALING_FACTOR + OFFSET", "irb_scaling_factor": radiometric_scale, "irb_offset": radiometric_offset, "texture_encoding": "8-bit Mars-style RGB visualization; spatial luminance derived from calibrated IRB and color range from the non-grayscale MRGB swath", "diagnostic_reflectance_encoding": "per-channel I/F converted to sRGB"},
        "outputs": {"dtm_tif": str(dtm_tif), "irb_tif": str(irb_tif), "irb_png": str(args.output_dir / "irb_color.png"), "irb_reflectance_srgb_png": str(args.output_dir / "irb_reflectance_srgb.png"), "irb_mars_rgb_png": str(args.output_dir / "irb_mars_rgb.png"), "mrgb_candidate_tif": str(mrgb_tif), "mrgb_candidate_png": str(args.output_dir / "mrgb_candidate.png"), "texture_png": str(texture_path), "heightfield_npz": str(args.output_dir / "terrain_heightfield.npz")},
    }
    errors = validate_crop_report(config, report)
    report["validation"] = {"ok": not errors, "errors": errors}
    (args.output_dir / "metadata.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("Crop validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
