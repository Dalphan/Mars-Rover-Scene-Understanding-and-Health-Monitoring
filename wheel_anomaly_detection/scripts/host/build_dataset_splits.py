"""Build deterministic index-only train/validation/test splits from a pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wheel_preparation.dataset_splits import build_dataset_splits, write_split_indexes


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing pool manifest: {path}")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Pool manifest has a truncated final line")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--overwrite-splits", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_dir = args.dataset_dir.resolve()
    if ((dataset_dir / "splits").exists() or (dataset_dir / "dataset.json").exists()) and not args.overwrite_splits:
        raise ValueError("Split outputs already exist; pass --overwrite-splits to replace indexes")
    rows = _read_jsonl(dataset_dir / config["pool"]["manifest"])
    print(json.dumps(write_split_indexes(dataset_dir, config, build_dataset_splits(config, rows)), indent=2))


if __name__ == "__main__":
    main()
