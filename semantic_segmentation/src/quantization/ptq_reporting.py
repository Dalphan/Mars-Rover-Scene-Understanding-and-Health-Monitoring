from __future__ import annotations

from omegaconf import OmegaConf

from src.quantization.core import save_json


def persist_results(results, cfg, paths) -> None:
    """Persist incremental PTQ results together with the resolved config."""

    results["configuration"] = OmegaConf.to_container(cfg, resolve=True)
    save_json(results, paths.results)
