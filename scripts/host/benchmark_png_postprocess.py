"""Benchmark lossless PNG encoding choices on canonical pre-bulk rasters."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import statistics
import struct
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(payload, binascii.crc32(kind)) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _encode_unfiltered(image: Image.Image, path: Path, level: int) -> None:
    channels = 3 if image.mode == "RGB" else 1
    color_type = 2 if image.mode == "RGB" else 0
    raw = image.tobytes()
    stride = image.width * channels
    filtered = b"".join(b"\x00" + raw[offset : offset + stride] for offset in range(0, len(raw), stride))
    header = struct.pack(">IIBBBBB", image.width, image.height, 8, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", header) + _chunk(b"IDAT", zlib.compress(filtered, level=level)) + _chunk(b"IEND", b""))


def _encode_pillow(image: Image.Image, path: Path, level: int) -> None:
    image.save(path, format="PNG", compress_level=level, optimize=False)


VARIANTS = {
    "current_unfiltered_zlib4": lambda image, path: _encode_unfiltered(image, path, 4),
    "unfiltered_zlib1": lambda image, path: _encode_unfiltered(image, path, 1),
    "adaptive_pillow_zlib1": lambda image, path: _encode_pillow(image, path, 1),
    "adaptive_pillow_zlib4": lambda image, path: _encode_pillow(image, path, 4),
}


def _even_sample(values: list[Path], count: int) -> list[Path]:
    values = sorted(set(path.resolve() for path in values))
    if len(values) <= count:
        return values
    return [values[(index * len(values)) // count] for index in range(count)]


def _pool_paths(pool_manifest: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for row in _read_jsonl(pool_manifest):
        for member in row["members"]:
            result["rgb"].append(REPO_ROOT / member["rgb"]["path"])
            result["target_wheel_mask"].append(REPO_ROOT / member["target_wheel_mask"]["path"])
            if "anomaly_mask" in member:
                result["anomaly_mask"].append(REPO_ROOT / member["anomaly_mask"]["path"])
    return result


def benchmark(pool_manifest: Path, sample_count: int) -> dict:
    paths = _pool_paths(pool_manifest)
    selected = {
        "rgb": _even_sample(paths["rgb"], sample_count),
        "target_wheel_mask": _even_sample(paths["target_wheel_mask"], sample_count),
        "anomaly_mask": _even_sample(paths["anomaly_mask"], sample_count),
    }
    results = {name: {kind: {"seconds": 0.0, "bytes": 0, "files": 0, "encode_seconds": []} for kind in selected} for name in VARIANTS}
    with tempfile.TemporaryDirectory(prefix="png_postprocess_", dir=REPO_ROOT / "outputs") as temporary:
        temp = Path(temporary)
        for kind, source_paths in selected.items():
            for source_index, source in enumerate(source_paths):
                with Image.open(source) as opened:
                    image = opened.copy()
                expected = hashlib.sha256(image.tobytes()).hexdigest()
                for variant_name, encoder in VARIANTS.items():
                    output = temp / f"{kind}_{source_index:03d}_{variant_name}.png"
                    started = time.perf_counter()
                    encoder(image, output)
                    elapsed = time.perf_counter() - started
                    with Image.open(output) as check:
                        check.load()
                        actual = hashlib.sha256(check.tobytes()).hexdigest()
                        if check.mode != image.mode or check.size != image.size or actual != expected:
                            raise RuntimeError(f"Pixel mismatch for {variant_name}: {source}")
                    row = results[variant_name][kind]
                    row["seconds"] += elapsed
                    row["bytes"] += output.stat().st_size
                    row["files"] += 1
                    row["encode_seconds"].append(elapsed)

    summary = {}
    for variant_name, by_kind in results.items():
        total_seconds = sum(row["seconds"] for row in by_kind.values())
        total_bytes = sum(row["bytes"] for row in by_kind.values())
        summary[variant_name] = {
            "seconds": total_seconds,
            "bytes": total_bytes,
            "mib": total_bytes / (1024 * 1024),
            "files": sum(row["files"] for row in by_kind.values()),
            "by_kind": {
                kind: {
                    "files": row["files"],
                    "seconds": row["seconds"],
                    "mean_seconds": statistics.mean(row["encode_seconds"]) if row["encode_seconds"] else 0.0,
                    "median_seconds": statistics.median(row["encode_seconds"]) if row["encode_seconds"] else 0.0,
                    "bytes": row["bytes"],
                    "mib": row["bytes"] / (1024 * 1024),
                }
                for kind, row in by_kind.items()
            },
        }
    baseline = summary["current_unfiltered_zlib4"]
    for row in summary.values():
        row["time_change_vs_current_fraction"] = row["seconds"] / baseline["seconds"] - 1.0
        row["size_change_vs_current_fraction"] = row["bytes"] / baseline["bytes"] - 1.0
    return {
        "schema_version": 1,
        "scope": "lossless final PNG encode only; decode, RGBA split, validation, hashing and filesystem commit are outside timed sections",
        "pool_manifest": pool_manifest.resolve().relative_to(REPO_ROOT).as_posix(),
        "selection": {kind: [path.relative_to(REPO_ROOT).as_posix() for path in values] for kind, values in selected.items()},
        "sample_counts": {kind: len(values) for kind, values in selected.items()},
        "variants": summary,
        "pixel_identity": "verified for every output",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample-count must be positive")
    output = args.output.resolve()
    if not output.is_relative_to((REPO_ROOT / "outputs").resolve()):
        raise ValueError("Benchmark report must be written below outputs/")
    report = benchmark(args.pool_manifest.resolve(), args.sample_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {key: row[key] for key in ("seconds", "mib", "time_change_vs_current_fraction", "size_change_vs_current_fraction")} for name, row in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
