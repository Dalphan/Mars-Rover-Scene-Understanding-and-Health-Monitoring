from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..data import (
    build_dataloaders,
    build_efficientad_train_calibration_loaders,
    build_imagenet_penalty_loader,
)
from ..evaluation import (
    AnomalyDiagnostics,
    evaluate_anomaly_detector,
    evaluate_efficientad_calibration_ablation,
    evaluate_efficientad_global_topk_ablation,
    evaluate_gaussian_sigma_ablation,
    evaluate_image_score_aggregations,
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
    calibration_loader = None
    if (
        str(cfg.model.name) == "efficientad_s"
        and bool(cfg.model.spatial_calibration_enabled)
    ):
        train_loader, calibration_loader = (
            build_efficientad_train_calibration_loaders(cfg)
        )
    detector = build_model(cfg)
    fit_kwargs = {
        "device": device,
        "validation_loader": validation_loader,
        "work_dir": output_dir,
        "resume": resume,
    }
    if str(cfg.model.name) == "efficientad_s":
        fit_kwargs["penalty_loader"] = build_imagenet_penalty_loader(cfg)
        fit_kwargs["calibration_loader"] = calibration_loader
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

    def build_diagnostics(loader, *, enabled=diagnostics_enabled):
        if not enabled:
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

    calibration_ablation_path = None
    calibration_ablation_config = cfg.evaluation.efficientad_calibration_ablation
    if (
        str(cfg.model.name) == "efficientad_s"
        and bool(calibration_ablation_config.enabled)
    ):
        print("[post-hoc] Running EfficientAD calibration/ROI diagnostics")
        calibration_ablation_results = evaluate_efficientad_calibration_ablation(
            detector,
            validation_loader,
            test_loader,
            device=device,
            diagnostics_factory=lambda split: build_diagnostics(
                validation_loader if split == "validation" else test_loader,
                enabled=True,
            ),
            histogram_bins=int(cfg.evaluation.histogram_bins),
            restrict_pixels_to_target_mask=bool(
                cfg.evaluation.restrict_pixels_to_target_mask
            ),
        )
        results["efficientad_calibration_ablation"] = (
            calibration_ablation_results
        )
        calibration_ablation_path = save_json_atomic(
            calibration_ablation_results,
            output_dir / "efficientad_calibration_ablation.json",
        )

    global_topk_ablation_path = None
    global_topk_config = cfg.evaluation.efficientad_global_topk_ablation
    if str(cfg.model.name) == "efficientad_s" and bool(global_topk_config.enabled):
        if bool(cfg.model.spatial_calibration_enabled):
            raise ValueError(
                "EfficientAD global top-k ablation requires "
                "model.spatial_calibration_enabled=false"
            )
        print("[post-hoc] Selecting EfficientAD global native top-k on validation")
        global_topk_results = evaluate_efficientad_global_topk_ablation(
            detector,
            validation_loader,
            test_loader,
            device=device,
            topk_pixels=OmegaConf.to_container(
                global_topk_config.native_pixel_candidates, resolve=True
            ),
            diagnostics_factory=lambda split: build_diagnostics(
                validation_loader if split == "validation" else test_loader,
                enabled=True,
            ),
            histogram_bins=int(cfg.evaluation.histogram_bins),
            restrict_pixels_to_target_mask=bool(
                cfg.evaluation.restrict_pixels_to_target_mask
            ),
        )
        results["efficientad_global_topk_ablation"] = global_topk_results
        global_topk_ablation_path = save_json_atomic(
            global_topk_results,
            output_dir / "efficientad_global_topk_ablation.json",
        )

    aggregation_path = None
    model_name = str(cfg.model.name)
    aggregation_config = (
        cfg.evaluation.image_score_aggregation
        if model_name == "tinyglass"
        else cfg.evaluation.supersimplenet_image_score_aggregation
        if model_name == "supersimplenet"
        else None
    )
    if aggregation_config is not None and bool(aggregation_config.enabled):
        candidates = OmegaConf.to_container(
            aggregation_config.candidates, resolve=True
        )
        validation_aggregations = evaluate_image_score_aggregations(
            detector, validation_loader, candidates, device=device,
            description="Validation image aggregations",
        )
        if model_name == "supersimplenet":
            selected_name = str(aggregation_config.fixed_candidate)
            if selected_name not in validation_aggregations:
                raise ValueError(
                    "SuperSimpleNet fixed image-score candidate is not configured"
                )
            selection_strategy = "fixed_pre_registered"
            selection_split = None
            selection_metric = None
        else:
            selected_name = max(
                validation_aggregations,
                key=lambda name: validation_aggregations[name]["image_auroc"],
            )
            selection_strategy = "validation_best"
            selection_split = "validation"
            selection_metric = "image_auroc"
        selected_candidate = next(
            candidate for candidate in candidates
            if str(candidate["name"]) == selected_name
        )
        if model_name == "supersimplenet":
            if (
                selected_candidate["mode"] != detector.image_score_mode
                or abs(
                    float(selected_candidate["fraction"])
                    - detector.image_score_fraction
                ) > 1e-12
            ):
                raise RuntimeError(
                    "SuperSimpleNet fixed aggregation and prediction config differ"
                )
            for metric_name in ("image_auroc", "image_average_precision"):
                if abs(
                    validation_aggregations[selected_name][metric_name]
                    - validation_metrics[metric_name]
                ) > 1e-8:
                    raise RuntimeError(
                        "SuperSimpleNet fixed aggregation does not reproduce "
                        f"{metric_name}"
                    )
            test_aggregation = {
                metric_name: test_metrics[metric_name]
                for metric_name in ("image_auroc", "image_average_precision")
            }
        else:
            test_aggregation = evaluate_image_score_aggregations(
                detector, test_loader, [selected_candidate], device=device,
                description=f"Test image aggregation ({selected_name})",
            )[selected_name]
        aggregation_results = {
            "selection_strategy": selection_strategy,
            "selection_split": selection_split,
            "selection_metric": selection_metric,
            "selected": selected_name,
            "selected_candidate": selected_candidate,
            "candidates": candidates,
            "validation": validation_aggregations,
            "test": {selected_name: test_aggregation},
        }
        result_key = f"{model_name}_image_score_aggregation"
        results[result_key] = aggregation_results
        aggregation_path = save_json_atomic(
            aggregation_results, output_dir / f"{result_key}.json"
        )

    sigma_ablation_path = None
    model_name = str(cfg.model.name)
    sigma_ablation_config = (
        cfg.evaluation.supersimplenet_gaussian_sigma_ablation
        if model_name == "supersimplenet"
        else cfg.evaluation.gaussian_sigma_ablation
    )
    if model_name in {"tinyglass", "supersimplenet"} and bool(
        sigma_ablation_config.enabled
    ):
        print("[post-hoc] Evaluating Gaussian sigma candidates on validation")
        sigma_ablation_results = evaluate_gaussian_sigma_ablation(
            detector,
            validation_loader,
            OmegaConf.to_container(sigma_ablation_config.candidates, resolve=True),
            device=device,
            diagnostics_factory=lambda: build_diagnostics(
                validation_loader, enabled=True
            ),
            reference_image_metrics=validation_metrics,
            histogram_bins=int(cfg.evaluation.histogram_bins),
            restrict_pixels_to_target_mask=bool(
                cfg.evaluation.restrict_pixels_to_target_mask
            ),
            test_loader=test_loader if model_name == "supersimplenet" else None,
            test_diagnostics_factory=(
                (lambda: build_diagnostics(test_loader, enabled=True))
                if model_name == "supersimplenet" else None
            ),
            reference_test_image_metrics=(
                test_metrics if model_name == "supersimplenet" else None
            ),
            selection_metric=(
                str(sigma_ablation_config.selection_metric)
                if model_name == "supersimplenet" else None
            ),
        )
        result_key = f"{model_name}_gaussian_sigma_ablation"
        results[result_key] = sigma_ablation_results
        sigma_ablation_path = save_json_atomic(
            sigma_ablation_results,
            output_dir / f"{result_key}.json",
        )

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
            "image_score_aggregation": (
                None if aggregation_path is None else aggregation_path.name
            ),
            "gaussian_sigma_ablation": (
                None if sigma_ablation_path is None else sigma_ablation_path.name
            ),
            "efficientad_calibration_ablation": (
                None
                if calibration_ablation_path is None
                else calibration_ablation_path.name
            ),
            "efficientad_global_topk_ablation": (
                None
                if global_topk_ablation_path is None
                else global_topk_ablation_path.name
            ),
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
        "aggregation_path": aggregation_path,
        "sigma_ablation_path": sigma_ablation_path,
        "calibration_ablation_path": calibration_ablation_path,
        "global_topk_ablation_path": global_topk_ablation_path,
        "diagnostics": diagnostics_results,
        "metrics": results,
    }
