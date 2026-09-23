from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

def require_module(name: str, install_hint: str = "requirements-quantization.txt"):
    try:
        return __import__(name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional dependency {name!r} is missing; install {install_hint}"
        ) from exc


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def save_json(data, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
    return path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
