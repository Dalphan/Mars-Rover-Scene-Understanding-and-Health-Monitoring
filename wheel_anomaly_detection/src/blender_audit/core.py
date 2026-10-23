from __future__ import annotations

import math
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence


WHEEL_NAME_PATTERN = re.compile(
    r"(^|[^a-z])(wheel|wheels|tire|tyre|roue|pneu|ruota|ruote)([^a-z]|$)",
    re.IGNORECASE,
)
NON_WHEEL_NAME_PATTERN = re.compile(
    r"(chassis|body|mast|camera|antenna|deck|solar|arm|boom)",
    re.IGNORECASE,
)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def connected_components(
    vertex_count: int, edges: Iterable[Sequence[int]]
) -> list[list[int]]:
    """Return vertex-index components, including isolated vertices."""
    if vertex_count < 0:
        raise ValueError("vertex_count must be non-negative")
    union_find = _UnionFind(vertex_count)
    for edge in edges:
        if len(edge) != 2:
            raise ValueError(f"edge must contain two vertex indices, got {edge!r}")
        left, right = int(edge[0]), int(edge[1])
        if not (0 <= left < vertex_count and 0 <= right < vertex_count):
            raise ValueError(f"edge index outside [0, {vertex_count}): {edge!r}")
        union_find.union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for vertex_index in range(vertex_count):
        groups[union_find.find(vertex_index)].append(vertex_index)
    return sorted(groups.values(), key=lambda component: (-len(component), component[0]))


def bbox_from_points(
    points: Iterable[Sequence[float]],
) -> tuple[list[float], list[float]]:
    iterator = iter(points)
    try:
        first = [float(value) for value in next(iterator)]
    except StopIteration:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    minimum = first[:]
    maximum = first[:]
    for point in iterator:
        for axis, value in enumerate(point):
            numeric = float(value)
            minimum[axis] = min(minimum[axis], numeric)
            maximum[axis] = max(maximum[axis], numeric)
    return minimum, maximum


def dimensions_from_bbox(
    minimum: Sequence[float], maximum: Sequence[float]
) -> list[float]:
    return [max(0.0, float(high) - float(low)) for low, high in zip(minimum, maximum)]


def center_from_bbox(
    minimum: Sequence[float], maximum: Sequence[float]
) -> list[float]:
    return [(float(low) + float(high)) * 0.5 for low, high in zip(minimum, maximum)]


def wheel_shape_signals(dimensions: Sequence[float]) -> dict[str, float | int]:
    dims = [max(0.0, float(value)) for value in dimensions]
    if len(dims) != 3 or max(dims, default=0.0) <= 1e-12:
        return {
            "cylinder_like": 0.0,
            "radial_similarity": 0.0,
            "thickness_ratio": 0.0,
            "axle_axis": 0,
        }
    axle_axis = min(range(3), key=dims.__getitem__)
    small, middle, large = sorted(dims)
    radial_similarity = middle / large if large else 0.0
    thickness_ratio = small / large if large else 0.0
    radial_score = clamp((radial_similarity - 0.62) / 0.25)
    thin_low = clamp((thickness_ratio - 0.08) / 0.12)
    thin_high = clamp((0.96 - thickness_ratio) / 0.16)
    cylinder_like = radial_score * min(thin_low, thin_high)
    return {
        "cylinder_like": round(cylinder_like, 6),
        "radial_similarity": round(radial_similarity, 6),
        "thickness_ratio": round(thickness_ratio, 6),
        "axle_axis": axle_axis,
    }


def _repeat_signature(item: Mapping[str, Any]) -> tuple[int, ...]:
    dims = sorted(max(float(value), 1e-12) for value in item["dimensions"])
    largest = dims[-1]
    normalized = [value / largest for value in dims]
    scale_bin = int(round(math.log(largest, 1.20))) if largest > 1e-10 else -999
    vertex_count = max(int(item.get("vertex_count", 0)), 1)
    face_count = max(int(item.get("face_count", 0)), 1)
    return (
        int(round(normalized[0] * 10)),
        int(round(normalized[1] * 10)),
        scale_bin,
        int(round(math.log2(vertex_count) * 2)),
        int(round(math.log2(face_count) * 2)),
    )


def repeat_counts(items: Sequence[Mapping[str, Any]]) -> list[int]:
    """Count geometry/scale signatures without relying on object names."""
    signatures = [_repeat_signature(item) for item in items]
    histogram: dict[tuple[int, ...], int] = defaultdict(int)
    for signature in signatures:
        histogram[signature] += 1
    return [histogram[signature] for signature in signatures]


def repeat_group_labels(items: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return stable, report-safe labels for repeated geometry signatures."""
    signatures = [_repeat_signature(item) for item in items]
    unique = {
        signature: f"geometry_group_{index:04d}"
        for index, signature in enumerate(sorted(set(signatures)), start=1)
    }
    return [unique[signature] for signature in signatures]


def score_wheel_candidate(
    item: Mapping[str, Any],
    scene_bbox: Mapping[str, Sequence[float]],
    repeat_count: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Score a mesh object/component using independent geometric signals."""
    dimensions = [float(value) for value in item["dimensions"]]
    center = [float(value) for value in item["center"]]
    shape = wheel_shape_signals(dimensions)

    scene_min = [float(value) for value in scene_bbox["min"]]
    scene_max = [float(value) for value in scene_bbox["max"]]
    scene_dims = dimensions_from_bbox(scene_min, scene_max)
    z_fraction = (
        (center[2] - scene_min[2]) / scene_dims[2] if scene_dims[2] > 1e-12 else 0.5
    )
    lower_score = clamp((0.68 - z_fraction) / 0.38)

    horizontal_offsets = []
    for axis in (0, 1):
        half_extent = scene_dims[axis] * 0.5
        scene_center = (scene_min[axis] + scene_max[axis]) * 0.5
        horizontal_offsets.append(
            abs(center[axis] - scene_center) / half_extent if half_extent > 1e-12 else 0.0
        )
    peripheral_score = clamp((max(horizontal_offsets) - 0.28) / 0.52)

    name = str(item.get("name", ""))
    name_score = 1.0 if WHEEL_NAME_PATTERN.search(name) else 0.0
    name_penalty = 1.0 if NON_WHEEL_NAME_PATTERN.search(name) else 0.0
    repeat_score = clamp((int(repeat_count) - 1) / 5.0)
    topology_size = int(item.get("vertex_count", 0)) + int(item.get("face_count", 0))
    topology_score = clamp((math.log10(max(topology_size, 1)) - 1.5) / 2.5)

    weights = config.get(
        "weights",
        {
            "name": 0.20,
            "shape": 0.28,
            "lower": 0.14,
            "peripheral": 0.12,
            "repeat": 0.18,
            "topology": 0.08,
        },
    )
    score = (
        float(weights["name"]) * name_score
        + float(weights["shape"]) * float(shape["cylinder_like"])
        + float(weights["lower"]) * lower_score
        + float(weights["peripheral"]) * peripheral_score
        + float(weights["repeat"]) * repeat_score
        + float(weights["topology"]) * topology_score
        - float(config.get("non_wheel_name_penalty", 0.12)) * name_penalty
    )
    score = clamp(score)
    threshold = float(config.get("candidate_threshold", 0.50))
    return {
        "score": round(score, 6),
        "passes_threshold": score >= threshold,
        "signals": {
            **shape,
            "name_keyword": bool(name_score),
            "non_wheel_name_keyword": bool(name_penalty),
            "lower_position": round(lower_score, 6),
            "z_scene_fraction": round(z_fraction, 6),
            "peripheral_position": round(peripheral_score, 6),
            "repeat_count": int(repeat_count),
            "repeat_score": round(repeat_score, 6),
            "topology_size": topology_size,
            "topology_score": round(topology_score, 6),
        },
    }


def classify_asset(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Classify the asset into the requested A-D remediation categories."""
    viable = [
        candidate
        for candidate in candidates
        if float(candidate.get("score", 0.0)) >= 0.50
    ]
    if not viable:
        return {
            "category": "D",
            "title": "Asset inadatto alle modifiche geometriche richieste",
            "confidence": 0.90,
            "rationale": (
                "Nessuna candidata ruota supera la soglia geometrica minima; "
                "non è possibile dimostrare un percorso affidabile di estrazione."
            ),
        }

    directly_usable = []
    automatically_separable = []
    for candidate in viable:
        duplicate = candidate.get("duplication", {}).get("verdict")
        boolean = candidate.get("boolean_difference", {}).get("verdict")
        thickness = candidate.get("skin_thickness", {}).get("verdict")
        source_kind = candidate.get("source_kind")
        if (
            source_kind == "mesh_object"
            and duplicate == "yes"
            and boolean in {"yes", "conditional"}
            and thickness in {"yes", "uncertain"}
        ):
            directly_usable.append(candidate)
        if (
            source_kind in {
                "disconnected_component",
                "disconnected_component_cluster",
            }
            and candidate.get("automatic_separation", {}).get("verdict") == "yes"
            and boolean in {"yes", "conditional"}
        ):
            automatically_separable.append(candidate)

    if directly_usable:
        confidence = max(float(candidate["score"]) for candidate in directly_usable)
        return {
            "category": "A",
            "title": "Ruota già separata e direttamente utilizzabile",
            "confidence": round(confidence, 3),
            "rationale": (
                "Almeno una ruota è un oggetto mesh autonomo, duplicabile senza "
                "dipendenze esterne bloccanti e compatibile con modifiche booleane."
            ),
        }
    if automatically_separable:
        confidence = max(float(candidate["score"]) for candidate in automatically_separable)
        return {
            "category": "B",
            "title": "Ruota separabile automaticamente da una mesh più grande",
            "confidence": round(confidence, 3),
            "rationale": (
                "Almeno una ruota coincide con una componente geometrica disconnessa "
                "che può essere estratta programmaticamente dalla mesh sorgente."
            ),
        }
    return {
        "category": "C",
        "title": "Ruota utilizzabile solo dopo una correzione manuale minima",
        "confidence": round(max(float(candidate["score"]) for candidate in viable), 3),
        "rationale": (
            "Sono presenti candidate credibili, ma topologia, spessore o dipendenze "
            "richiedono una correzione locale prima di perforazioni e grousers rotti."
        ),
    }


def _yes_no(value: Any) -> str:
    if value is True:
        return "sì"
    if value is False:
        return "no"
    return str(value)


def make_markdown_report(report: Mapping[str, Any]) -> str:
    """Create the human-readable companion to the machine report."""
    classification = report.get("classification", {})
    asset = report.get("asset", {})
    environment = report.get("environment", {})
    detection = report.get("wheel_detection", {})
    gpu = environment.get("gpu", {})
    lines = [
        "# Audit geometrico Blender — Curiosity",
        "",
        f"- **Stato:** `{report.get('status', 'unknown')}`",
        f"- **Asset:** `{asset.get('path', 'n/d')}`",
        f"- **SHA-256 invariato:** {_yes_no(asset.get('unchanged', 'n/d'))}",
        f"- **Blender:** `{environment.get('blender_version', 'n/d')}`",
        f"- **Render engine:** `{environment.get('selected_render_engine', 'n/d')}`",
        (
            "- **GPU:** "
            f"`{gpu.get('vendor', 'n/d')} | {gpu.get('renderer', 'n/d')} | "
            f"{gpu.get('backend', 'n/d')}`"
        ),
        "",
        "## Verdetto",
        "",
        (
            f"**{classification.get('category', 'n/d')}. "
            f"{classification.get('title', 'Classificazione non disponibile')}**"
        ),
        "",
        str(classification.get("rationale", "")),
        "",
        "## Individuazione delle ruote",
        "",
        f"- Candidate: **{detection.get('candidate_count', 0)}**",
        f"- Sei ruote come oggetti separati: **{detection.get('six_wheels_separate', 'uncertain')}**",
        f"- Organizzazione: `{detection.get('organization', 'unknown')}`",
        (
            "- Strategia: nomi + proporzioni cilindriche + quota + posizione "
            "periferica + ripetizione geometrica/topologica."
        ),
        "",
    ]

    candidates = detection.get("candidates", [])
    if not candidates:
        lines.extend(
            [
                "Nessuna ruota candidata è stata individuata.",
                "",
            ]
        )
    for candidate in candidates:
        lines.extend(
            [
                f"### {candidate.get('id', 'candidate')} — `{candidate.get('object_name', '')}`",
                "",
                f"- Sorgente: `{candidate.get('source_kind', 'unknown')}`",
                f"- Punteggio: `{candidate.get('score', 0):.3f}`",
                f"- Dimensioni XYZ: `{candidate.get('dimensions', [])}`",
                (
                    "- Duplicazione indipendente: "
                    f"**{candidate.get('duplication', {}).get('verdict', 'unknown')}** — "
                    f"{candidate.get('duplication', {}).get('reason', '')}"
                ),
                (
                    "- Boolean Difference: "
                    f"**{candidate.get('boolean_difference', {}).get('verdict', 'unknown')}** — "
                    f"{candidate.get('boolean_difference', {}).get('reason', '')}"
                ),
                (
                    "- Spessore pelle: "
                    f"**{candidate.get('skin_thickness', {}).get('verdict', 'unknown')}** — "
                    f"{candidate.get('skin_thickness', {}).get('reason', '')}"
                ),
                (
                    "- Grousers: "
                    f"**{candidate.get('grousers', {}).get('verdict', 'unknown')}** — "
                    f"{candidate.get('grousers', {}).get('reason', '')}"
                ),
                (
                    "- Voxel remesh / retopology / ricostruzione: "
                    f"`{candidate.get('remediation', {})}`"
                ),
                "",
            ]
        )

    meshes = report.get("meshes", [])
    lines.extend(
        [
            "## Inventario",
            "",
            f"- Collezioni: **{len(report.get('scene', {}).get('collections', []))}**",
            f"- Oggetti: **{len(report.get('scene', {}).get('objects', []))}**",
            f"- Mesh: **{len(meshes)}**",
            f"- Texture mancanti: **{len(report.get('materials', {}).get('missing_textures', []))}**",
            "",
            "## Artefatti",
            "",
        ]
    )
    for key, value in report.get("artifacts", {}).items():
        if isinstance(value, list):
            lines.append(f"- `{key}`: {len(value)} file")
        else:
            lines.append(f"- `{key}`: `{value}`")

    warnings = report.get("warnings", [])
    errors = report.get("errors", [])
    lines.extend(["", "## Warning ed errori", ""])
    if not warnings and not errors:
        lines.append("Nessun warning o errore registrato.")
    else:
        lines.extend(f"- WARNING: {warning}" for warning in warnings)
        lines.extend(f"- ERROR: {error}" for error in errors)
    lines.extend(
        [
            "",
            "## Limiti dell’audit",
            "",
            (
                "La misura di spessore è una stima ray-cast sulla geometria importata; "
                "non sostituisce una verifica CAD. L’audit non applica Boolean, remesh "
                "o separazioni all’asset sorgente."
            ),
            "",
        ]
    )
    return "\n".join(lines)
