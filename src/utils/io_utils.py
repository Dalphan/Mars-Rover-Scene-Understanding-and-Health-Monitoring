from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path, logger: logging.Logger | None = None) -> Path:
    logger = logger or logging.getLogger(__name__)
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    logger.info("Ensured directory exists: %s", directory)
    return directory


def save_json(data: dict[str, Any], path: str | Path, logger: logging.Logger | None = None) -> None:
    logger = logger or logging.getLogger(__name__)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    logger.info("Saved JSON: %s", output_path)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path, logger: logging.Logger | None = None) -> None:
    logger = logger or logging.getLogger(__name__)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved CSV: %s", output_path)
