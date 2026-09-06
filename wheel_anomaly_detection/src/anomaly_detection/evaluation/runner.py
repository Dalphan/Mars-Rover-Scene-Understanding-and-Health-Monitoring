from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import math
from typing import Any

import torch
from tqdm import tqdm

from ..models import AnomalyDetector
from .diagnostics import AnomalyDiagnostics
from .metrics import AnomalyMetrics, ExactBinaryMetrics, update_metrics_from_batch


def aggregate_patch_scores(
    patch_scores: torch.Tensor, specification: Mapping[str, Any],
) -> torch.Tensor:
    """Aggregate one patch-score map per image without changing pixel outputs."""
    flat_scores = patch_scores.flatten(1)
    mode = str(specification["mode"])
    if mode == "max":
        return flat_scores.amax(1)
    if mode == "topk_mean":
        topk = int(specification["topk"])
        if not 1 <= topk <= flat_scores.shape[1]:
            raise ValueError("topk must be between 1 and the number of patches")
        return flat_scores.topk(topk, dim=1).values.mean(1)
    if mode == "topk_fraction_mean":
        fraction = float(specification["fraction"])
        if not 0 < fraction <= 1:
            raise ValueError("topk fraction must be in (0, 1]")
        topk = min(flat_scores.shape[1], math.ceil(fraction * flat_scores.shape[1]))
        return flat_scores.topk(topk, dim=1).values.mean(1)
    if mode == "quantile":
        quantile = float(specification["quantile"])
        if not 0 <= quantile <= 1:
            raise ValueError("quantile must be in [0, 1]")
        return torch.quantile(flat_scores, quantile, dim=1)
    raise ValueError(f"Unknown image-score aggregation mode: {mode}")


@torch.no_grad()
def evaluate_image_score_aggregations(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    description: str = "Image-score aggregation",
) -> dict[str, dict[str, float]]:
    """Evaluate image-score candidates from one detector pass per batch."""
    candidate_list = [dict(candidate) for candidate in candidates]
    names = [str(candidate["name"]) for candidate in candidate_list]
    if not candidate_list or len(names) != len(set(names)):
        raise ValueError("Aggregation candidates must have unique names")
    if not detector.is_fitted:
        raise RuntimeError("The anomaly detector must be fitted before evaluation")
    has_components = hasattr(detector, "_image_score_components")
    if not has_components and not hasattr(detector, "_patch_scores"):
        raise TypeError("Image-score aggregation evaluation requires patch scores")
    if (
        any(str(candidate["mode"]) == "classification_head"
            for candidate in candidate_list)
        and not has_components
    ):
        raise TypeError(
            "classification_head aggregation requires image-score components"
        )

    device = torch.device(device)
    detector.to(device)
    detector.train(False)
    metrics = {name: ExactBinaryMetrics() for name in names}
    num_images = 0
    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device, non_blocking=True)
        if has_components:
            patch_scores, classification_scores = (
                detector._image_score_components(images)
            )
        else:
            patch_scores = detector._patch_scores(images)
            classification_scores = None
        labels = torch.as_tensor(batch["label"])
        for candidate, name in zip(candidate_list, names):
            if str(candidate["mode"]) == "classification_head":
                image_scores = classification_scores
            else:
                image_scores = aggregate_patch_scores(patch_scores, candidate)
            metrics[name].update(
                image_scores, labels
            )
        num_images += images.shape[0]
    if num_images == 0:
        raise ValueError("Evaluation loader produced no images")
    return {
        name: {
            "image_auroc": values["auroc"],
            "image_average_precision": values["average_precision"],
        }
        for name, metric in metrics.items()
        for values in (metric.compute(),)
    }


@torch.no_grad()
def evaluate_anomaly_detector(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    histogram_bins: int = 2048,
    restrict_pixels_to_target_mask: bool = False,
    description: str = "Evaluation",
    diagnostics: AnomalyDiagnostics | None = None,
) -> dict[str, float | int]:
    """Evaluate one fitted detector and optionally collect diagnostics."""
    if not detector.is_fitted:
        raise RuntimeError("The anomaly detector must be fitted before evaluation")
    device = torch.device(device)
    detector.to(device)
    detector.train(False)
    metrics = AnomalyMetrics(histogram_bins=histogram_bins)
    num_images = 0

    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device, non_blocking=True)
        predict_with_raw = getattr(detector, "predict_with_raw", None)
        if predict_with_raw is None:
            prediction = detector.predict(images)
            raw_image_scores = prediction.anomaly_score
        else:
            prediction, raw_image_scores, _ = predict_with_raw(images)
        update_metrics_from_batch(
            metrics,
            prediction,
            batch,
            image_scores=raw_image_scores,
            restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
        )
        if diagnostics is not None:
            diagnostics.update(prediction, raw_image_scores, batch)
        num_images += images.shape[0]
    if num_images == 0:
        raise ValueError("Evaluation loader produced no images")
    return metrics.compute()


def evaluate_efficientad_calibration_ablation(
    detector: AnomalyDetector,
    validation_loader: Iterable[Mapping[str, Any]],
    test_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    diagnostics_factory: Callable[[str], AnomalyDiagnostics],
    histogram_bins: int = 2048,
    restrict_pixels_to_target_mask: bool = False,
) -> dict[str, Any]:
    """Compare EfficientAD calibration and ROI effects without model selection."""
    configure = getattr(detector, "configure_diagnostic_inference", None)
    if configure is None:
        raise TypeError("Calibration ablation requires EfficientAD diagnostic modes")
    attribute_names = (
        "_diagnostic_spatial_override",
        "_diagnostic_static_roi_override",
        "_diagnostic_topk_fraction_override",
        "_diagnostic_global_topk_pixels_override",
    )
    previous = {name: getattr(detector, name) for name in attribute_names}
    configured_spatial = bool(detector.selected_spatial_calibration)
    configured_topk = float(detector.selected_topk_fraction)
    modes = ("global", "global_roi", "spatial_roi")
    results: dict[str, dict[str, Any]] = {
        "validation": {}, "test": {},
    }
    try:
        for mode in modes:
            configure(mode)
            for split, loader in (
                ("validation", validation_loader),
                ("test", test_loader),
            ):
                diagnostics = diagnostics_factory(split)
                metrics = evaluate_anomaly_detector(
                    detector,
                    loader,
                    device=device,
                    histogram_bins=histogram_bins,
                    restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
                    description=f"EfficientAD {split} diagnostic ({mode})",
                    diagnostics=diagnostics,
                )
                pro = diagnostics.compute()["pro"]
                results[split][mode] = {
                    **metrics,
                    "aupro": pro["aupro"],
                    "aupro_by_max_fpr": pro["aupro_by_max_fpr"],
                }
    finally:
        for name, value in previous.items():
            setattr(detector, name, value)

    configured_mode = "spatial_roi" if configured_spatial else "global"
    return {
        "purpose": "diagnostic_only",
        "selection": "none",
        "test_used_for_selection": False,
        "image_score_aggregation": "max",
        "configured_mode": configured_mode,
        "configured_topk_fraction": configured_topk,
        "modes": list(modes),
        **results,
    }


def evaluate_efficientad_global_topk_ablation(
    detector: AnomalyDetector,
    validation_loader: Iterable[Mapping[str, Any]],
    test_loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    topk_pixels: Iterable[int],
    diagnostics_factory: Callable[[str], AnomalyDiagnostics],
    histogram_bins: int = 2048,
    restrict_pixels_to_target_mask: bool = False,
) -> dict[str, Any]:
    """Select a global native-map top-k on validation and test it once."""
    configure_mode = getattr(detector, "configure_diagnostic_inference", None)
    configure_topk = getattr(
        detector, "configure_diagnostic_global_topk_pixels", None
    )
    if configure_mode is None or configure_topk is None:
        raise TypeError("Global top-k ablation requires EfficientAD diagnostics")
    pixel_counts = [int(value) for value in topk_pixels]
    if (
        not pixel_counts
        or len(pixel_counts) != len(set(pixel_counts))
        or any(value < 2 for value in pixel_counts)
    ):
        raise ValueError("topk_pixels must contain unique integers >= 2")

    candidates: list[tuple[str, int | None]] = [
        ("global_max_resized", None),
        ("global_max_native", 1),
        *[(f"global_topk_{value}_native", value) for value in pixel_counts],
    ]
    attribute_names = (
        "_diagnostic_spatial_override",
        "_diagnostic_static_roi_override",
        "_diagnostic_topk_fraction_override",
        "_diagnostic_global_topk_pixels_override",
    )
    previous = {name: getattr(detector, name) for name in attribute_names}
    validation: dict[str, Any] = {}
    selected_test: dict[str, Any] | None = None
    try:
        for name, count in candidates:
            configure_mode("global")
            configure_topk(count)
            diagnostics = diagnostics_factory("validation")
            metrics = evaluate_anomaly_detector(
                detector,
                validation_loader,
                device=device,
                histogram_bins=histogram_bins,
                restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
                description=f"EfficientAD validation image score ({name})",
                diagnostics=diagnostics,
            )
            pro = diagnostics.compute()["pro"]
            validation[name] = {
                "native_topk_pixels": count,
                **metrics,
                "aupro": pro["aupro"],
                "aupro_by_max_fpr": pro["aupro_by_max_fpr"],
            }

        invariant_names = (
            "pixel_auroc",
            "pixel_average_precision",
            "target_wheel_pixel_auroc",
            "target_wheel_pixel_average_precision",
            "aupro",
        )
        reference = validation["global_max_resized"]
        for name, result in validation.items():
            for metric_name in invariant_names:
                if abs(float(result[metric_name]) - float(
                    reference[metric_name]
                )) > 1e-8:
                    raise RuntimeError(
                        f"Image aggregation unexpectedly changed {metric_name} "
                        f"for {name}"
                    )

        selected_name, selected_count = max(
            candidates,
            key=lambda candidate: validation[candidate[0]]["image_auroc"],
        )
        configure_mode("global")
        configure_topk(selected_count)
        diagnostics = diagnostics_factory("test")
        test_metrics = evaluate_anomaly_detector(
            detector,
            test_loader,
            device=device,
            histogram_bins=histogram_bins,
            restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
            description=f"EfficientAD test image score ({selected_name})",
            diagnostics=diagnostics,
        )
        pro = diagnostics.compute()["pro"]
        selected_test = {
            "native_topk_pixels": selected_count,
            **test_metrics,
            "aupro": pro["aupro"],
            "aupro_by_max_fpr": pro["aupro_by_max_fpr"],
        }
    finally:
        for name, value in previous.items():
            setattr(detector, name, value)

    return {
        "selection_split": "validation",
        "selection_metric": "image_auroc",
        "test_used_for_selection": False,
        "selected": selected_name,
        "selected_native_topk_pixels": selected_count,
        "candidates": [name for name, _ in candidates],
        "map_metric_invariance_tolerance": 1e-8,
        "validation": validation,
        "test": {selected_name: selected_test},
    }


def evaluate_gaussian_sigma_ablation(
    detector: AnomalyDetector,
    loader: Iterable[Mapping[str, Any]],
    candidates: Iterable[float],
    *,
    device: torch.device | str,
    diagnostics_factory: Callable[[], AnomalyDiagnostics],
    reference_image_metrics: Mapping[str, float | int],
    histogram_bins: int = 2048,
    restrict_pixels_to_target_mask: bool = False,
    description: str = "Validation Gaussian sigma",
    test_loader: Iterable[Mapping[str, Any]] | None = None,
    test_diagnostics_factory: Callable[[], AnomalyDiagnostics] | None = None,
    reference_test_image_metrics: Mapping[str, float | int] | None = None,
    selection_metric: str | None = None,
) -> dict[str, Any]:
    """Measure smoothing on validation and optionally test its selected winner."""
    if not hasattr(detector, "gaussian_sigma"):
        raise TypeError("Gaussian sigma ablation requires detector.gaussian_sigma")
    sigmas = [float(value) for value in candidates]
    if not sigmas or len(sigmas) != len(set(sigmas)):
        raise ValueError("Gaussian sigma candidates must be non-empty and unique")
    if any(not math.isfinite(value) or value < 0 for value in sigmas):
        raise ValueError("Gaussian sigma candidates must be finite and non-negative")
    if (test_loader is None) != (selection_metric is None):
        raise ValueError("test_loader and selection_metric must be provided together")
    if test_loader is not None and test_diagnostics_factory is None:
        raise ValueError("Selected sigma test evaluation requires diagnostics")
    if test_loader is not None and reference_test_image_metrics is None:
        raise ValueError("Selected sigma test evaluation requires reference metrics")

    configured_sigma = float(detector.gaussian_sigma)
    validation: dict[str, Any] = {}
    selected_name = None
    selected_test = None
    try:
        for sigma in sigmas:
            detector.gaussian_sigma = sigma
            diagnostics = diagnostics_factory()
            metrics = evaluate_anomaly_detector(
                detector,
                loader,
                device=device,
                histogram_bins=histogram_bins,
                restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
                description=f"{description}={sigma:g}",
                diagnostics=diagnostics,
            )
            for metric_name in ("image_auroc", "image_average_precision"):
                if abs(float(metrics[metric_name]) - float(
                    reference_image_metrics[metric_name]
                )) > 1e-8:
                    raise RuntimeError(
                        f"Gaussian smoothing unexpectedly changed {metric_name}"
                    )
            pro = diagnostics.compute()["pro"]
            name = f"sigma_{sigma:g}".replace(".", "_")
            validation[name] = {
                "sigma": sigma,
                **metrics,
                "aupro": pro["aupro"],
                "aupro_by_max_fpr": pro["aupro_by_max_fpr"],
            }
        if selection_metric is not None:
            if any(selection_metric not in result for result in validation.values()):
                raise ValueError(
                    f"Unknown Gaussian sigma selection metric: {selection_metric}"
                )
            selected_name = max(
                validation, key=lambda name: validation[name][selection_metric]
            )
            detector.gaussian_sigma = validation[selected_name]["sigma"]
            diagnostics = test_diagnostics_factory()
            metrics = evaluate_anomaly_detector(
                detector,
                test_loader,
                device=device,
                histogram_bins=histogram_bins,
                restrict_pixels_to_target_mask=restrict_pixels_to_target_mask,
                description=f"Test Gaussian sigma={detector.gaussian_sigma:g}",
                diagnostics=diagnostics,
            )
            for metric_name in ("image_auroc", "image_average_precision"):
                if abs(
                    float(metrics[metric_name])
                    - float(reference_test_image_metrics[metric_name])
                ) > 1e-8:
                    raise RuntimeError(
                        f"Gaussian smoothing unexpectedly changed {metric_name}"
                    )
            pro = diagnostics.compute()["pro"]
            selected_test = {
                "sigma": detector.gaussian_sigma,
                **metrics,
                "aupro": pro["aupro"],
                "aupro_by_max_fpr": pro["aupro_by_max_fpr"],
            }
    finally:
        detector.gaussian_sigma = configured_sigma

    results = {
        "split": "validation",
        "selection": "none" if selection_metric is None else "validation_best",
        "selection_metric": selection_metric,
        "test_used_for_selection": False,
        "configured_sigma": configured_sigma,
        "candidates": sigmas,
        "image_metric_invariance_tolerance": 1e-8,
        "validation": validation,
    }
    if selected_name is not None:
        results["selected"] = selected_name
        results["test"] = {selected_name: selected_test}
    return results
