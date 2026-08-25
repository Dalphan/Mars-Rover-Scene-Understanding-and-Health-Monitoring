from __future__ import annotations

from omegaconf import DictConfig

from .base import AnomalyDetector


def _optional_string(value) -> str | None:
    return None if value is None else str(value)


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
            teacher_weights_path=_optional_string(cfg.model.teacher_weights_path),
            require_teacher_weights=bool(cfg.model.require_teacher_weights),
            channels=int(cfg.model.channels),
            max_steps=int(cfg.model.max_steps),
            learning_rate=float(cfg.model.learning_rate),
            weight_decay=float(cfg.model.weight_decay),
            hard_quantile=float(cfg.model.hard_quantile),
            checkpoint_interval=int(cfg.model.checkpoint_interval),
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
            fixed_training_duration=bool(cfg.model.fixed_training_duration),
            validation_interval=int(cfg.model.validation_interval),
            validation_batches=int(cfg.model.validation_batches),
            max_samples_per_epoch=(
                None if cfg.model.max_samples_per_epoch is None
                else int(cfg.model.max_samples_per_epoch)
            ),
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
            max_samples_per_epoch=(
                None if cfg.model.max_samples_per_epoch is None
                else int(cfg.model.max_samples_per_epoch)
            ),
            texture_root=_optional_string(cfg.model.texture_root),
            require_texture_dataset=bool(cfg.model.require_texture_dataset),
            blend_mean=float(cfg.model.blend_mean),
            blend_std=float(cfg.model.blend_std),
            gaussian_sigma=float(cfg.model.gaussian_sigma),
            fixed_training_duration=bool(cfg.model.fixed_training_duration),
            validation_interval=int(cfg.model.validation_interval),
            validation_batches=int(cfg.model.validation_batches),
        )
    raise ValueError(
        f"Unknown model name {name!r}. Expected patchcore, efficientad_s, "
        "supersimplenet, or tinyglass."
    )
