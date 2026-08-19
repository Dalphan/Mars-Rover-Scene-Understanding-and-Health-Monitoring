"""Download the configured HiRISE Gale product with resume and hash gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_bytes: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if expected_bytes is not None and existing == expected_bytes:
        return
    request = urllib.request.Request(url)
    mode = "wb"
    if existing:
        request.add_header("Range", f"bytes={existing}-")
        mode = "ab"
    with urllib.request.urlopen(request) as response:
        if existing and response.status != 206:
            existing = 0
            mode = "wb"
        with destination.open(mode) as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
    if expected_bytes is not None and destination.stat().st_size != expected_bytes:
        raise RuntimeError(f"Unexpected size for {destination.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    product = config["product"]
    raw_dir = args.data_root / product["id"] / "raw"
    results = []
    for item in product["files"]:
        path = raw_dir / item["name"]
        download(item.get("url", f"{product['base_url']}/{item['name']}"), path, item.get("bytes"))
        digest = sha256(path)
        if item.get("sha256") and digest.lower() != item["sha256"].lower():
            raise RuntimeError(f"SHA-256 mismatch for {item['name']}")
        results.append({"name": item["name"], "bytes": path.stat().st_size, "sha256": digest})
    print(json.dumps({"product": product["id"], "raw_dir": str(raw_dir), "files": results}, indent=2))


if __name__ == "__main__":
    main()
