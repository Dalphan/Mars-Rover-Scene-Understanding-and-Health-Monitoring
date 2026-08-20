from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


def save_json_atomic(data: Mapping[str, Any], path: str | Path) -> Path:
    """Write JSON without leaving a partially written result file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path
