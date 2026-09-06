from __future__ import annotations

from omegaconf import DictConfig

from .base import AnomalyDetector


def _optional_string(value) -> str | None:
    return None if value is None else str(value)


EFFICIENTAD_HARD_QUANTILE_PRESETS = {
    "official": 0.999,
    "balanced": 0.995,
    "broad": 0.99,
}


def _efficientad_hard_quantile(cfg: DictConfig) -> float:
    preset = str(cfg.model.hard_quantile_preset)
    if preset not in EFFICIENTAD_HARD_QUANTILE_PRESETS:
        raise ValueError(
            "model.hard_quantile_preset must be official, balanced, or broad"
        )
    return EFFICIENTAD_HARD_QUANTILE_PRESETS[preset]


def build_model(cfg: DictConfig) -> AnomalyDetector:
    """Build the anomaly-detection model selected by Hydra."""
    name = str(cfg.model.name)
    if name == "patchcore":
        from .patchcore import PatchCore

        return PatchCore(
            backbone=str(cfg.model.backbone),
            pretrained=bool(cfg.model.pretrained),
            layers=tuple(cfg.model.layers),
            coreset_sampling_ratio=float(cfg.model.coreset_sampling_ratio),
            num_neighbors=int(cfg.model.num_neighbors),
            patch_size=int(cfg.model.patch_size),
            patch_stride=int(cfg.model.patch_stride),
            pretrain_embed_dimension=int(cfg.model.pretrain_embed_dimension),
            target_embed_dimension=int(cfg.model.target_embed_dimension),
            max_patches_per_image=int(cfg.model.max_patches_per_image),
            max_training_embeddings=int(cfg.model.max_training_embeddings),
            max_memory_bank_size=int(cfg.model.max_memory_bank_size),
            projection_dim=int(cfg.model.projection_dim),
            sampling_seed=int(cfg.model.sampling_seed),
            calibration_quantile=float(cfg.model.calibration_quantile),
            calibration_batches=int(cfg.model.calibration_batches),
            distance_query_chunk_size=int(cfg.model.distance_query_chunk_size),
            distance_bank_chunk_size=int(cfg.model.distance_bank_chunk_size),
            gaussian_sigma=float(cfg.model.gaussian_sigma),
        )

    from .supersimplenet import SuperSimpleNet
    from .tinyglass import TinyGLASS
    from .trainable import EfficientAD

    if name == "efficientad_s":
        return EfficientAD(
            input_size=tuple(int(value) for value in cfg.model.input_size),
            teacher_weights_path=_optional_string(cfg.model.teacher_weights_path),
            require_teacher_weights=bool(cfg.model.require_teacher_weights),
            channels=int(cfg.model.channels),
            max_steps=int(cfg.model.max_steps),
            learning_rate=float(cfg.model.learning_rate),
            weight_decay=float(cfg.model.weight_decay),
            hard_quantile=_efficientad_hard_quantile(cfg),
            checkpoint_interval=int(cfg.model.checkpoint_interval),
            validation_interval=int(cfg.model.validation_interval),
            early_stopping_patience=int(cfg.model.early_stopping_patience),
            early_stopping_min_steps=int(cfg.model.early_stopping_min_steps),
            early_stopping_min_relative_improvement=float(
                cfg.model.early_stopping_min_relative_improvement
            ),
            lr_scheduler_patience=int(cfg.model.lr_scheduler_patience),
            fixed_training_duration=bool(cfg.model.fixed_training_duration),
            mixed_precision=bool(cfg.model.mixed_precision),
            spatial_calibration_enabled=bool(
                cfg.model.spatial_calibration_enabled
            ),
            spatial_calibration_q_low=float(
                cfg.model.spatial_calibration_q_low
            ),
            spatial_calibration_q_high=float(
                cfg.model.spatial_calibration_q_high
            ),
            spatial_calibration_smoothing_sigma=float(
                cfg.model.spatial_calibration_smoothing_sigma
            ),
            spatial_scale_floor_fraction=float(
                cfg.model.spatial_scale_floor_fraction
            ),
            static_roi_min_coverage=float(cfg.model.static_roi_min_coverage),
            image_score_topk_candidates=tuple(
                float(value) for value in cfg.model.image_score_topk_candidates
            ),
        )
    if name == "supersimplenet":
        return SuperSimpleNet(
            backbone=str(cfg.model.backbone),
            pretrained=bool(cfg.model.pretrained),
            weights_name=str(cfg.model.weights_name),
            layers=tuple(cfg.model.layers),
            input_size=tuple(cfg.model.input_size),
            patch_size=int(cfg.model.patch_size),
            epochs=int(cfg.model.epochs),
            noise_std=float(cfg.model.noise_std),
            perlin_threshold=float(cfg.model.perlin_threshold),
            adaptor_learning_rate=float(cfg.model.adaptor_learning_rate),
            segmentation_learning_rate=float(cfg.model.segmentation_learning_rate),
            decision_learning_rate=float(cfg.model.decision_learning_rate),
            scheduler_gamma=float(cfg.model.scheduler_gamma),
            stop_grad=bool(cfg.model.stop_grad),
            adapt_classification_features=bool(
                cfg.model.adapt_classification_features
            ),
            gradient_clip=bool(cfg.model.gradient_clip),
            margin=float(cfg.model.margin),
            gaussian_sigma=float(cfg.model.gaussian_sigma),
            image_score_mode=str(cfg.model.image_score_mode),
            image_score_fraction=float(cfg.model.image_score_fraction),
            fixed_training_duration=bool(cfg.model.fixed_training_duration),
            validation_interval=int(cfg.model.validation_interval),
            validation_batches=(
                None if cfg.model.validation_batches is None
                else int(cfg.model.validation_batches)
            ),
            early_stopping_patience=(
                None if cfg.model.early_stopping_patience is None
                else int(cfg.model.early_stopping_patience)
            ),
            early_stopping_min_delta=float(
                cfg.model.early_stopping_min_delta
            ),
            restrict_synthetic_anomalies_to_target_mask=bool(
                cfg.model.restrict_synthetic_anomalies_to_target_mask
            ),
            max_samples_per_epoch=(
                None if cfg.model.max_samples_per_epoch is None
                else int(cfg.model.max_samples_per_epoch)
            ),
            checkpoint_interval=int(cfg.model.checkpoint_interval),
        )
    if name == "tinyglass":
        return TinyGLASS(
            pretrained=bool(cfg.model.pretrained),
            weights_name=str(cfg.model.weights_name),
            input_size=tuple(cfg.model.input_size),
            patch_size=int(cfg.model.patch_size),
            epochs=int(cfg.model.epochs),
            learning_rate=float(cfg.model.learning_rate),
            weight_decay=float(cfg.model.weight_decay),
            noise_std=float(cfg.model.noise_std),
            radius_quantile=float(cfg.model.radius_quantile),
            hard_mining_quantile=float(cfg.model.hard_mining_quantile),
            gas_steps=int(cfg.model.gas_steps),
            gas_step_size=float(cfg.model.gas_step_size),
            hypersphere_projection=bool(cfg.model.hypersphere_projection),
            feature_grid_resolution=str(cfg.model.feature_grid_resolution),
            max_samples_per_epoch=(
                None if cfg.model.max_samples_per_epoch is None
                else int(cfg.model.max_samples_per_epoch)
            ),
            texture_root=_optional_string(cfg.model.texture_root),
            require_texture_dataset=bool(cfg.model.require_texture_dataset),
            blend_mean=float(cfg.model.blend_mean),
            blend_std=float(cfg.model.blend_std),
            gaussian_sigma=float(cfg.model.gaussian_sigma),
            las_restrict_to_target_mask=bool(
                cfg.model.las_restrict_to_target_mask
            ),
            las_mode=str(cfg.model.las_mode),
            las_hole_probability=float(cfg.model.las_hole_probability),
            las_hole_luminance_range=tuple(
                float(value) for value in cfg.model.las_hole_luminance_range
            ),
            las_hole_severity_weights=tuple(
                float(value) for value in cfg.model.las_hole_severity_weights
            ),
            freeze_backbone=bool(cfg.model.freeze_backbone),
            backbone_learning_rate=float(cfg.model.backbone_learning_rate),
            fixed_training_duration=bool(cfg.model.fixed_training_duration),
            validation_interval=int(cfg.model.validation_interval),
            validation_batches=(
                None if cfg.model.validation_batches is None
                else int(cfg.model.validation_batches)
            ),
            cache_validation_features=bool(cfg.model.cache_validation_features),
            early_stopping_patience=(
                None if cfg.model.early_stopping_patience is None
                else int(cfg.model.early_stopping_patience)
            ),
            early_stopping_min_delta=float(cfg.model.early_stopping_min_delta),
            checkpoint_interval=int(cfg.model.checkpoint_interval),
        )
    raise ValueError(
        f"Unknown model name {name!r}. Expected patchcore, efficientad_s, "
        "supersimplenet, or tinyglass."
    )
