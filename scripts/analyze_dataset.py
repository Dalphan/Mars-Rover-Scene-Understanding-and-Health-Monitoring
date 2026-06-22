from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

if not getattr(argparse.ArgumentParser, "_s5mars_hydra_patch", False):
    _original_check_help = argparse.ArgumentParser._check_help

    def _check_help_compatible(self, action):
        try:
            return _original_check_help(self, action)
        except ValueError as error:
            if action.dest == "shell_completion":
                return None
            raise error

    argparse.ArgumentParser._check_help = _check_help_compatible
    argparse.ArgumentParser._s5mars_hydra_patch = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.dataset_stats import run_full_analysis
from src.analysis.visualization import (
    denormalize_image,
    make_palette,
    plot_sample,
    save_sample_grid,
)
from src.data.dataloaders import build_dataloader, build_dataset
from src.utils.io_utils import ensure_dir, save_dataframe_csv, save_json
from src.utils.logging_utils import setup_logger
from src.utils.seed import set_seed


def _resolve_token(cfg, logger):
    token = os.environ.get("HF_TOKEN") or cfg.huggingface.token
    if token in ("", "HF_TOKEN_PLACEHOLDER", None):
        logger.warning("HF_TOKEN is not set; using unauthenticated Hugging Face access")
        return None
    logger.info("Using Hugging Face token from %s", "environment" if os.environ.get("HF_TOKEN") else "config")
    return token


def _save_analysis_outputs(results, cfg, output_dir: Path, logger) -> None:
    if cfg.analysis.save_json and "summary" in results:
        save_json(results["summary"], output_dir / "dataset_summary.json", logger)
    if cfg.analysis.save_json and "ignore_pixel_ratio" in results:
        save_json(results["ignore_pixel_ratio"], output_dir / "ignore_pixel_ratio.json", logger)
    if cfg.analysis.save_csv:
        for key, filename in (
            ("image_level_class_distribution", "image_level_class_distribution.csv"),
            ("mask_pixel_distribution", "mask_pixel_distribution.csv"),
        ):
            if key in results:
                save_dataframe_csv(results[key], output_dir / filename, logger)


def _save_visualizations(dataset, cfg, output_dir: Path, logger) -> None:
    if not cfg.visualization.enabled:
        logger.info("Visualization disabled")
        return

    vis_dir = ensure_dir(output_dir / "visualizations", logger)
    for stale_path in list(vis_dir.glob("sample_*.png")) + [vis_dir / "sample_grid.png"]:
        if stale_path.exists():
            stale_path.unlink()
            logger.info("Removed stale visualization: %s", stale_path)

    palette = make_palette(cfg.dataset.num_classes)
    sample_count = min(int(cfg.visualization.num_samples), len(dataset))
    if cfg.visualization.random_samples:
        indices = random.sample(range(len(dataset)), k=sample_count)
    else:
        indices = list(range(sample_count))

    class_names = dict(cfg.dataset.class_names)
    if cfg.visualization.save_grid and indices:
        save_sample_grid(
            dataset,
            indices,
            class_names,
            palette,
            cfg.visualization.alpha,
            vis_dir / "sample_grid.png",
            logger,
        )

    if cfg.visualization.save_individual:
        for index in indices:
            sample = dataset[index]
            image = sample["image"]
            if getattr(dataset, "transform", None) is not None and dataset.transform.normalize_enabled:
                image = denormalize_image(image, dataset.transform.mean.flatten(), dataset.transform.std.flatten())
            plot_sample(
                image,
                sample["mask"],
                class_names,
                palette,
                cfg.visualization.alpha,
                save_path=vis_dir / f"sample_{index}.png",
                title=f"S5Mars sample {index}",
            )
    logger.info("Saved visualizations to %s", vis_dir)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = ensure_dir(cfg.paths.output_dir)
    log_filename = cfg.logging.filename if cfg.logging.log_to_file else ""
    logger = setup_logger("s5mars_analysis", output_dir, log_filename, cfg.logging.level)
    logger.info("Starting S5Mars dataset analysis")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.seed))
    _resolve_token(cfg, logger)

    dataset = build_dataset(cfg, cfg.dataset.split, logger)
    dataloader = build_dataloader(dataset, cfg, cfg.dataset.split, logger)

    batch = next(iter(dataloader))
    logger.info(
        "Smoke batch: image dtype=%s shape=%s | mask dtype=%s shape=%s",
        batch["image"].dtype,
        tuple(batch["image"].shape),
        batch["mask"].dtype,
        tuple(batch["mask"].shape),
    )
    logger.info("First mask unique values: %s", sorted(batch["mask"][0].unique().tolist()))

    results = run_full_analysis(dataset, cfg, logger)
    _save_analysis_outputs(results, cfg, output_dir, logger)
    _save_visualizations(dataset, cfg, output_dir, logger)
    logger.info("Completed S5Mars analysis. Outputs saved to: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
