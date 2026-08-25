from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F
from torchvision.models import get_model, get_model_weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

from .base import AnomalyDetector, AnomalyPrediction


class PatchCore(AnomalyDetector):
    """Memory-bounded PatchCore anomaly detector with a frozen backbone."""

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        pretrained: bool = True,
        layers: Sequence[str] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        patch_size: int = 3,
        patch_stride: int = 1,
        pretrain_embed_dimension: int = 384,
        target_embed_dimension: int = 384,
        max_patches_per_image: int = 128,
        max_training_embeddings: int = 50_000,
        max_memory_bank_size: int = 2_048,
        projection_dim: int = 64,
        sampling_seed: int = 42,
        calibration_quantile: float = 0.99,
        calibration_batches: int = 20,
        distance_query_chunk_size: int = 512,
        distance_bank_chunk_size: int = 2_048,
        gaussian_sigma: float = 4.0,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one feature node")
        if not 0 < coreset_sampling_ratio <= 1:
            raise ValueError("coreset_sampling_ratio must be in (0, 1]")
        if num_neighbors < 1:
            raise ValueError("num_neighbors must be at least 1")
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        for name, value in (
            ("max_patches_per_image", max_patches_per_image),
            ("max_training_embeddings", max_training_embeddings),
            ("max_memory_bank_size", max_memory_bank_size),
            ("projection_dim", projection_dim),
            ("patch_stride", patch_stride),
            ("pretrain_embed_dimension", pretrain_embed_dimension),
            ("target_embed_dimension", target_embed_dimension),
            ("calibration_batches", calibration_batches),
            ("distance_query_chunk_size", distance_query_chunk_size),
            ("distance_bank_chunk_size", distance_bank_chunk_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0 < calibration_quantile <= 1:
            raise ValueError("calibration_quantile must be in (0, 1]")
        if gaussian_sigma < 0:
            raise ValueError("gaussian_sigma cannot be negative")

        weights = get_model_weights(backbone).DEFAULT if pretrained else None
        backbone_model = get_model(backbone, weights=weights)
        self.feature_extractor = create_feature_extractor(
            backbone_model,
            return_nodes={layer: layer for layer in layers},
        )
        self.feature_extractor.requires_grad_(False)

        self.backbone = backbone
        self.pretrained = bool(pretrained)
        self.layers = tuple(layers)
        self.coreset_sampling_ratio = float(coreset_sampling_ratio)
        self.num_neighbors = int(num_neighbors)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.pretrain_embed_dimension = int(pretrain_embed_dimension)
        self.target_embed_dimension = int(target_embed_dimension)
        self.max_patches_per_image = int(max_patches_per_image)
        self.max_training_embeddings = int(max_training_embeddings)
        self.max_memory_bank_size = int(max_memory_bank_size)
        self.projection_dim = int(projection_dim)
        self.sampling_seed = int(sampling_seed)
        self.calibration_quantile = float(calibration_quantile)
        self.calibration_batches = int(calibration_batches)
        self.distance_query_chunk_size = int(distance_query_chunk_size)
        self.distance_bank_chunk_size = int(distance_bank_chunk_size)
        self.gaussian_sigma = float(gaussian_sigma)

        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.register_buffer("image_score_scale", torch.tensor(0.0), persistent=True)
        self.register_buffer("anomaly_map_scale", torch.tensor(0.0), persistent=True)
        self.fit_summary: dict[str, int | float | list[int]] = {}
        self.train(False)

    @property
    def is_fitted(self) -> bool:
        return (
            self.memory_bank.ndim == 2
            and self.memory_bank.shape[0] > 0
            and self.image_score_scale.item() > 0
            and self.anomaly_map_scale.item() > 0
        )

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "backbone": self.backbone,
            "layers": list(self.layers),
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "pretrain_embed_dimension": self.pretrain_embed_dimension,
            "target_embed_dimension": self.target_embed_dimension,
            "max_patches_per_image": self.max_patches_per_image,
            "max_training_embeddings": self.max_training_embeddings,
            "max_memory_bank_size": self.max_memory_bank_size,
            "projection_dim": self.projection_dim,
            "sampling_seed": self.sampling_seed,
            "calibration_quantile": self.calibration_quantile,
            "calibration_batches": self.calibration_batches,
            "distance_query_chunk_size": self.distance_query_chunk_size,
            "distance_bank_chunk_size": self.distance_bank_chunk_size,
            "gaussian_sigma": self.gaussian_sigma,
        }

    def train(self, mode: bool = True) -> PatchCore:
        """Keep the frozen backbone and its BatchNorm layers in evaluation mode."""
        super().train(False)
        return self

    @staticmethod
    def _validate_images(images: torch.Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected RGB images [B, 3, H, W], got {tuple(images.shape)}")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("images must contain finite floating-point values")

    def _patchify(
        self,
        feature: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Extract overlapping local patches without reducing their channels."""
        padding = self.patch_size // 2
        patches = F.unfold(
            feature,
            kernel_size=self.patch_size,
            stride=self.patch_stride,
            padding=padding,
        ).transpose(1, 2)
        height = (feature.shape[-2] + 2 * padding - self.patch_size) // self.patch_stride + 1
        width = (feature.shape[-1] + 2 * padding - self.patch_size) // self.patch_stride + 1
        patches = patches.reshape(
            feature.shape[0],
            height * width,
            feature.shape[1],
            self.patch_size,
            self.patch_size,
        )
        return patches, (height, width)

    @staticmethod
    def _align_patch_grid(
        patches: torch.Tensor,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Interpolate patch grids while preserving channel and patch axes."""
        if source_size == target_size:
            return patches
        batch_size, _, channels, patch_height, patch_width = patches.shape
        patches = patches.reshape(
            batch_size,
            *source_size,
            channels,
            patch_height,
            patch_width,
        ).permute(0, 3, 4, 5, 1, 2)
        patches = patches.reshape(-1, 1, *source_size)
        patches = F.interpolate(
            patches,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return patches.reshape(
            batch_size,
            channels,
            patch_height,
            patch_width,
            *target_size,
        ).permute(0, 4, 5, 1, 2, 3).reshape(
            batch_size,
            target_size[0] * target_size[1],
            channels,
            patch_height,
            patch_width,
        )

    @staticmethod
    def _map_embedding_dimension(
        patches: torch.Tensor,
        output_dimension: int,
    ) -> torch.Tensor:
        """Apply the reference MeanMapper operation to every local patch."""
        batch_size, num_patches = patches.shape[:2]
        flattened = patches.reshape(batch_size * num_patches, 1, -1)
        mapped = F.adaptive_avg_pool1d(flattened, output_dimension)
        return mapped.reshape(batch_size, num_patches, output_dimension)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return reference-style locally aggregated embeddings [B, D, h, w]."""
        self._validate_images(images)
        features = self.feature_extractor(images)
        patch_layers = [self._patchify(features[layer]) for layer in self.layers]
        target_size = patch_layers[0][1]
        mapped_layers = []
        for patches, patch_grid in patch_layers:
            patches = self._align_patch_grid(patches, patch_grid, target_size)
            mapped_layers.append(
                self._map_embedding_dimension(
                    patches,
                    self.pretrain_embed_dimension,
                )
            )
        embeddings = torch.cat(mapped_layers, dim=-1)
        embeddings = self._map_embedding_dimension(
            embeddings,
            self.target_embed_dimension,
        )
        return embeddings.transpose(1, 2).reshape(
            images.shape[0],
            self.target_embed_dimension,
            *target_size,
        )

    def _patch_embeddings(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        embedding_map = self(images)
        height, width = embedding_map.shape[-2:]
        embeddings = embedding_map.permute(0, 2, 3, 1).flatten(1, 2).contiguous()
        return embeddings, (height, width)

    def _sample_image_patches(
        self,
        embeddings: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        batch_size, num_patches, embedding_dim = embeddings.shape
        selected_count = min(self.max_patches_per_image, num_patches)
        if selected_count == num_patches:
            return embeddings.reshape(-1, embedding_dim)
        random_keys = torch.rand((batch_size, num_patches), generator=generator)
        indices = random_keys.topk(selected_count, dim=1, largest=False).indices
        indices = indices.to(embeddings.device)
        return embeddings.gather(
            1,
            indices.unsqueeze(-1).expand(-1, -1, embedding_dim),
        ).reshape(-1, embedding_dim)

    @staticmethod
    def _reduce_priority_reservoir(
        embedding_chunks: list[torch.Tensor],
        priority_chunks: list[torch.Tensor],
        maximum_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        embeddings = torch.cat(embedding_chunks)
        priorities = torch.cat(priority_chunks)
        if embeddings.shape[0] > maximum_size:
            indices = priorities.topk(maximum_size, largest=True, sorted=False).indices
            embeddings = embeddings[indices]
            priorities = priorities[indices]
        return [embeddings], [priorities], embeddings.shape[0]

    def _build_coreset(
        self,
        candidates: torch.Tensor,
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> torch.Tensor:
        target_size = max(1, math.ceil(candidates.shape[0] * self.coreset_sampling_ratio))
        target_size = min(target_size, self.max_memory_bank_size, candidates.shape[0])
        if target_size == candidates.shape[0]:
            return candidates.contiguous()

        projection_dim = min(self.projection_dim, candidates.shape[1])
        projection = torch.randn(
            candidates.shape[1],
            projection_dim,
            generator=generator,
        ) / math.sqrt(projection_dim)
        projected = candidates.to(device) @ projection.to(device)

        first_index = int(torch.randint(candidates.shape[0], (1,), generator=generator))
        selected = torch.empty(target_size, dtype=torch.long, device=device)
        selected_mask = torch.zeros(candidates.shape[0], dtype=torch.bool, device=device)
        minimum_distances = torch.full(
            (candidates.shape[0],),
            torch.inf,
            dtype=projected.dtype,
            device=device,
        )
        current_index = first_index
        for position in tqdm(range(target_size), desc="PatchCore coreset", leave=False):
            selected[position] = current_index
            selected_mask[current_index] = True
            if position + 1 == target_size:
                break
            center = projected[current_index]
            distances = (projected - center).square().sum(dim=1)
            minimum_distances = torch.minimum(minimum_distances, distances)
            minimum_distances[selected_mask] = -1
            current_index = int(minimum_distances.argmax())
        return candidates[selected.cpu()].contiguous()

    @staticmethod
    def _batch_images(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
        if "image" not in batch:
            raise KeyError("Every training batch must contain 'image'")
        labels = batch.get("label")
        if labels is not None and torch.any(torch.as_tensor(labels) != 0):
            raise ValueError("PatchCore.fit accepts only clean training images")
        return batch["image"].to(device, non_blocking=True)

    @torch.no_grad()
    def fit(
        self,
        train_loader: Iterable[Mapping[str, Any]],
        *,
        device: torch.device | str,
        validation_loader: Iterable[Mapping[str, Any]] | None = None,
        work_dir: str | None = None,
        resume: bool = False,
    ) -> PatchCore:
        """Build the bounded coreset memory bank and clean-score calibration."""
        device = torch.device(device)
        self.to(device)
        self.train(False)
        self.memory_bank = torch.empty(0, 0, device=device)
        self.image_score_scale.zero_()
        self.anomaly_map_scale.zero_()
        self.fit_summary = {}
        generator = torch.Generator().manual_seed(self.sampling_seed)
        embedding_chunks: list[torch.Tensor] = []
        priority_chunks: list[torch.Tensor] = []
        buffered_count = 0
        num_images = 0
        sampled_embeddings = 0
        patch_grid: tuple[int, int] | None = None

        for batch in tqdm(train_loader, desc="PatchCore feature extraction"):
            images = self._batch_images(batch, device)
            embeddings, patch_grid = self._patch_embeddings(images)
            sampled = self._sample_image_patches(embeddings, generator).cpu()
            priorities = torch.rand(sampled.shape[0], generator=generator)
            embedding_chunks.append(sampled)
            priority_chunks.append(priorities)
            buffered_count += sampled.shape[0]
            sampled_embeddings += sampled.shape[0]
            num_images += images.shape[0]
            if buffered_count >= 2 * self.max_training_embeddings:
                embedding_chunks, priority_chunks, buffered_count = (
                    self._reduce_priority_reservoir(
                        embedding_chunks,
                        priority_chunks,
                        self.max_training_embeddings,
                    )
                )

        if not embedding_chunks or patch_grid is None:
            raise ValueError("train_loader produced no images")
        embedding_chunks, _, candidate_count = self._reduce_priority_reservoir(
            embedding_chunks,
            priority_chunks,
            self.max_training_embeddings,
        )
        candidates = embedding_chunks[0]
        self.memory_bank = self._build_coreset(
            candidates,
            device=device,
            generator=generator,
        ).to(device)

        image_scores: list[torch.Tensor] = []
        patch_scores: list[torch.Tensor] = []
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc="PatchCore calibration", leave=False)
        ):
            if batch_index >= self.calibration_batches:
                break
            images = self._batch_images(batch, device)
            raw_image_scores, raw_patch_scores, _ = self._raw_scores(images)
            image_scores.append(raw_image_scores.cpu())
            patch_scores.append(raw_patch_scores.flatten().cpu())
        if not image_scores:
            raise ValueError("train_loader must be re-iterable for calibration")

        epsilon = torch.finfo(torch.float32).eps
        image_scale = torch.quantile(
            torch.cat(image_scores).float(), self.calibration_quantile
        ).clamp_min(epsilon)
        map_scale = torch.quantile(
            torch.cat(patch_scores).float(), self.calibration_quantile
        ).clamp_min(epsilon)
        self.image_score_scale.copy_(image_scale.to(device))
        self.anomaly_map_scale.copy_(map_scale.to(device))
        self.fit_summary = {
            "num_training_images": num_images,
            "sampled_embeddings": sampled_embeddings,
            "reservoir_embeddings": candidate_count,
            "memory_bank_embeddings": self.memory_bank.shape[0],
            "embedding_dimension": self.memory_bank.shape[1],
            "patch_grid": list(patch_grid),
            "image_score_scale": float(self.image_score_scale),
            "anomaly_map_scale": float(self.anomaly_map_scale),
        }
        return self

    def _nearest_memory(
        self,
        queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore must be fitted before nearest-neighbour search")
        if queries.shape[1] != self.memory_bank.shape[1]:
            raise ValueError("Query and memory-bank embedding dimensions do not match")

        nearest_distances = torch.empty(queries.shape[0], device=queries.device)
        nearest_indices = torch.empty(
            queries.shape[0], dtype=torch.long, device=queries.device
        )
        for query_start in range(0, queries.shape[0], self.distance_query_chunk_size):
            query_end = min(query_start + self.distance_query_chunk_size, queries.shape[0])
            query_chunk = queries[query_start:query_end]
            best_distances = torch.full(
                (query_chunk.shape[0],), torch.inf, device=queries.device
            )
            best_indices = torch.zeros(
                query_chunk.shape[0], dtype=torch.long, device=queries.device
            )
            for bank_start in range(0, self.memory_bank.shape[0], self.distance_bank_chunk_size):
                bank_end = min(
                    bank_start + self.distance_bank_chunk_size,
                    self.memory_bank.shape[0],
                )
                distances = torch.cdist(query_chunk, self.memory_bank[bank_start:bank_end])
                block_distances, block_indices = distances.min(dim=1)
                improved = block_distances < best_distances
                best_distances[improved] = block_distances[improved]
                best_indices[improved] = block_indices[improved] + bank_start
            nearest_distances[query_start:query_end] = best_distances
            nearest_indices[query_start:query_end] = best_indices
        return nearest_distances, nearest_indices

    def _reweighted_image_scores(
        self,
        embeddings: torch.Tensor,
        patch_scores: torch.Tensor,
        nearest_indices: torch.Tensor,
    ) -> torch.Tensor:
        maximum_scores, maximum_locations = patch_scores.max(dim=1)
        if self.num_neighbors == 1 or self.memory_bank.shape[0] == 1:
            return maximum_scores

        scores = []
        num_support = min(self.num_neighbors, self.memory_bank.shape[0])
        for batch_index in range(embeddings.shape[0]):
            patch_index = maximum_locations[batch_index]
            query = embeddings[batch_index, patch_index]
            anchor_index = nearest_indices[batch_index, patch_index]
            anchor = self.memory_bank[anchor_index]
            anchor_distances = torch.linalg.vector_norm(self.memory_bank - anchor, dim=1)
            support_indices = anchor_distances.topk(
                num_support, largest=False
            ).indices
            support_distances = torch.linalg.vector_norm(
                self.memory_bank[support_indices] - query,
                dim=1,
            )
            anchor_position = (support_indices == anchor_index).nonzero(as_tuple=False)[0, 0]
            weight = 1.0 - torch.softmax(support_distances, dim=0)[anchor_position]
            scores.append(maximum_scores[batch_index] * weight)
        return torch.stack(scores)

    def _raw_scores(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        embeddings, patch_grid = self._patch_embeddings(images)
        batch_size, num_patches, embedding_dim = embeddings.shape
        distances, nearest_indices = self._nearest_memory(
            embeddings.reshape(-1, embedding_dim)
        )
        patch_scores = distances.reshape(batch_size, num_patches)
        nearest_indices = nearest_indices.reshape(batch_size, num_patches)
        image_scores = self._reweighted_image_scores(
            embeddings, patch_scores, nearest_indices
        )
        return image_scores, patch_scores, patch_grid

    @staticmethod
    def _normalize(raw_scores: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(raw_scores.dtype).eps
        return raw_scores / (raw_scores + scale.clamp_min(epsilon))

    @torch.no_grad()
    def predict_with_raw(
        self,
        images: torch.Tensor,
    ) -> tuple[AnomalyPrediction, torch.Tensor, torch.Tensor]:
        if not self.is_fitted:
            raise RuntimeError("PatchCore must be fitted before predict")
        input_size = images.shape[-2:]
        raw_image_scores, raw_patch_scores, patch_grid = self._raw_scores(images)
        raw_map = raw_patch_scores.reshape(images.shape[0], 1, *patch_grid)
        raw_map = F.interpolate(
            raw_map,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        if self.gaussian_sigma > 0:
            kernel_size = 2 * math.ceil(3 * self.gaussian_sigma) + 1
            raw_map = gaussian_blur(
                raw_map,
                [kernel_size, kernel_size],
                [self.gaussian_sigma, self.gaussian_sigma],
            )
        prediction = AnomalyPrediction(
            anomaly_score=self._normalize(raw_image_scores, self.image_score_scale),
            anomaly_map=self._normalize(raw_map, self.anomaly_map_scale),
        )
        return prediction, raw_image_scores, raw_map

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyPrediction:
        prediction, _, _ = self.predict_with_raw(images)
        return prediction

    def _prepare_state_dict_for_load(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        memory_bank = state_dict.get("memory_bank")
        if memory_bank is not None and memory_bank.shape != self.memory_bank.shape:
            self.memory_bank = torch.empty_like(memory_bank)
