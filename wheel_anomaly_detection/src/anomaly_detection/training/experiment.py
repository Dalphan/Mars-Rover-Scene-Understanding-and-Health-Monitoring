from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..data import build_dataloaders, build_imagenet_penalty_loader
from ..evaluation import (
    AnomalyDiagnostics,
    evaluate_anomaly_detector,
    save_anomaly_diagnostics,
    save_anomaly_visualizations,
)
from ..models import build_model
from ..utils import save_json_atomic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_anomaly_experiment(cfg: DictConfig) -> dict[str, Any]:
    """Fit, evaluate, diagnose, and save the configured anomaly detector."""
    set_seed(int(cfg.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume = bool(cfg.get("resume", False))
    if resume:
        resume_run_dir = cfg.get("resume_run_dir")
        if not resume_run_dir:
            raise ValueError("resume=true requires resume_run_dir=<existing run directory>")
        output_dir = Path(str(resume_run_dir))
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {output_dir}")
        run_id = output_dir.name
    else:
        run_id = (
            f"{cfg.model.run_name}__seed{cfg.seed}__"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir = Path(str(cfg.output.root)) / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, output_dir / "config.yaml", resolve=True)

    train_loader, validation_loader, test_loader = build_dataloaders(cfg)
    detector = build_model(cfg)
    fit_kwargs = {
        "device": device,
        "validation_loader": validation_loader,
        "work_dir": output_dir,
        "resume": resume,
    }
    if str(cfg.model.name) == "efficientad_s":
        fit_kwargs["penalty_loader"] = build_imagenet_penalty_loader(cfg)
    print(f"[1/5] Fitting {cfg.model.run_name} on {device}")
    detector.fit(train_loader, **fit_kwargs)
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_summary": getattr(detector, "fit_summary", {}),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    checkpoint_path = detector.save(output_dir / "model.ckpt", metadata=metadata)

    evaluation_kwargs = {
        "device": device,
        "histogram_bins": int(cfg.evaluation.histogram_bins),
        "restrict_pixels_to_target_mask": bool(
            cfg.evaluation.restrict_pixels_to_target_mask
        ),
    }
    diagnostics_config = cfg.evaluation.diagnostics
    diagnostics_enabled = bool(diagnostics_config.enabled)

    def build_diagnostics(loader):
        if not diagnostics_enabled:
            return None
        return AnomalyDiagnostics(
            loader.dataset.rows,
            preprocessor=loader.dataset.preprocessing,
            histogram_bins=int(cfg.evaluation.histogram_bins),
            pro_bins=int(diagnostics_config.pro_bins),
            pro_max_fpr=float(diagnostics_config.pro_max_fpr),
            group_fields=tuple(
                str(value) for value in diagnostics_config.group_fields
            ),
            num_extreme_examples=int(diagnostics_config.num_extreme_examples),
            restrict_pixels_to_target_mask=bool(
                cfg.evaluation.restrict_pixels_to_target_mask
            ),
        )

    validation_diagnostics = build_diagnostics(validation_loader)
    test_diagnostics = build_diagnostics(test_loader)
    print("[2/5] Evaluating validation split")
    validation_metrics = evaluate_anomaly_detector(
        detector,
        validation_loader,
        description="Validation",
        diagnostics=validation_diagnostics,
        **evaluation_kwargs,
    )
    print("[3/5] Evaluating test split")
    test_metrics = evaluate_anomaly_detector(
        detector,
        test_loader,
        description="Test",
        diagnostics=test_diagnostics,
        **evaluation_kwargs,
    )
    results = {"validation": validation_metrics, "test": test_metrics}

    visualization_config = cfg.evaluation.visualization
    diagnostics_results: dict[str, Any] = {}
    diagnostics_paths: list[Path] = []
    if diagnostics_enabled:
        for split, split_diagnostics in (
            ("validation", validation_diagnostics),
            ("test", test_diagnostics),
        ):
            diagnostic_result, split_paths = save_anomaly_diagnostics(
                split_diagnostics,
                output_dir,
                split=split,
                colormap=str(visualization_config.colormap),
                overlay_alpha=float(visualization_config.overlay_alpha),
                dpi=int(visualization_config.dpi),
            )
            diagnostics_results[split] = diagnostic_result
            diagnostics_paths.extend(split_paths)

    visualization_paths: list[Path] = []
    if bool(visualization_config.enabled):
        print("[4/5] Saving qualitative anomaly visualizations")
        for split, loader in (
            ("validation", validation_loader),
            ("test", test_loader),
        ):
            visualization_paths.append(
                save_anomaly_visualizations(
                    detector,
                    loader,
                    output_dir / f"{split}_examples.png",
                    device=device,
                    preprocessor=loader.dataset.preprocessing,
                    title=f"{split.title()} anomaly examples",
                    num_clean=int(visualization_config.num_clean),
                    num_anomalous=int(visualization_config.num_anomalous),
                    colormap=str(visualization_config.colormap),
                    overlay_alpha=float(visualization_config.overlay_alpha),
                    dpi=int(visualization_config.dpi),
                )
            )

    metrics_path = save_json_atomic(results, output_dir / "metrics.json")
    manifest_path = save_json_atomic(
        {
            **metadata,
            "checkpoint": checkpoint_path.name,
            "metrics": metrics_path.name,
            "visualizations": [path.name for path in visualization_paths],
            "diagnostics": [path.name for path in diagnostics_paths],
        },
        output_dir / "run_manifest.json",
    )
    print(f"[5/5] Artifacts saved to {output_dir}")
    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "checkpoint_path": checkpoint_path,
        "metrics_path": metrics_path,
        "manifest_path": manifest_path,
        "visualization_paths": visualization_paths,
        "diagnostics_paths": diagnostics_paths,
        "diagnostics": diagnostics_results,
        "metrics": results,
    }
