from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def find_candidate(
    audit_report: Mapping[str, Any], candidate_id: str
) -> dict[str, Any]:
    candidates = audit_report.get("wheel_detection", {}).get("candidates", [])
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("id") == candidate_id
    ]
    if not matches:
        available = ", ".join(
            str(candidate.get("id"))
            for candidate in candidates
            if isinstance(candidate, Mapping)
        )
        raise ValueError(
            f"Candidate {candidate_id!r} not found in audit report. "
            f"Available: {available or 'none'}"
        )
    if len(matches) != 1:
        raise ValueError(f"Candidate id is not unique: {candidate_id!r}")
    return dict(matches[0])


def _cluster_levels(values: Sequence[float], tolerance: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for value in sorted(float(item) for item in values):
        if not groups or abs(value - sum(groups[-1]) / len(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def canonical_axis_from_candidates(
    candidates: Sequence[Mapping[str, Any]],
    selected_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Infer the rover lateral/wheel axle axis from the six wheel centers.

    The lateral axis has two repeated levels (left/right), while the longitudinal
    axis has three repeated levels (front/middle/rear). This is more reliable for
    an assembled wheel than the smallest bounding-box dimension.
    """

    centers = [
        [float(value) for value in candidate.get("center", [])]
        for candidate in candidates
        if isinstance(candidate, Mapping) and len(candidate.get("center", [])) == 3
    ]
    if len(centers) < 4:
        raise ValueError("At least four candidate centers are required to infer axle axis")

    axis_descriptors: list[dict[str, Any]] = []
    for axis in range(3):
        values = [center[axis] for center in centers]
        span = max(values) - min(values)
        tolerance = max(span * 0.08, 1e-8)
        groups = _cluster_levels(values, tolerance)
        group_sizes = sorted((len(group) for group in groups), reverse=True)
        two_side_score = (
            1.0
            if len(groups) == 2 and group_sizes and min(group_sizes) >= 2
            else 1.0 / (1.0 + abs(len(groups) - 2))
        )
        axis_descriptors.append(
            {
                "axis": axis,
                "span": span,
                "level_count": len(groups),
                "level_centers": [
                    sum(group) / len(group) for group in groups
                ],
                "group_sizes": group_sizes,
                "score": two_side_score * max(span, 1e-12),
            }
        )

    descriptor = max(
        axis_descriptors,
        key=lambda item: (
            item["level_count"] == 2,
            min(item["group_sizes"], default=0),
            item["score"],
        ),
    )
    axis = int(descriptor["axis"])
    selected_center = [float(value) for value in selected_candidate.get("center", [])]
    if len(selected_center) != 3:
        raise ValueError("Selected candidate has no valid 3D center")
    scene_midpoint = (
        min(center[axis] for center in centers)
        + max(center[axis] for center in centers)
    ) * 0.5
    outboard_sign = 1 if selected_center[axis] >= scene_midpoint else -1
    return {
        "axis_index": axis,
        "axis_name": "XYZ"[axis],
        "outboard_sign": outboard_sign,
        "selected_center": selected_center,
        "scene_midpoint": scene_midpoint,
        "descriptors": axis_descriptors,
        "method": "two lateral levels across repeated wheel centers",
    }


def validate_audit_compatibility(
    audit_report: Mapping[str, Any],
    *,
    asset_sha256: str,
    blender_version: Sequence[int],
    object_name: str,
    mesh_counts: Mapping[str, int],
    candidate_id: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if audit_report.get("status") != "completed":
        errors.append("Audit report status is not completed")

    audited_sha = audit_report.get("asset", {}).get("sha256_before")
    if audited_sha != asset_sha256:
        errors.append(
            f"Asset SHA-256 mismatch: audit={audited_sha!r}, current={asset_sha256!r}"
        )

    audited_version = audit_report.get("environment", {}).get(
        "blender_version_tuple", []
    )
    if list(audited_version[:2]) != list(blender_version[:2]):
        errors.append(
            "Blender major/minor mismatch: "
            f"audit={audited_version!r}, current={list(blender_version)!r}"
        )

    matching_meshes = [
        mesh
        for mesh in audit_report.get("meshes", [])
        if isinstance(mesh, Mapping) and mesh.get("object_name") == object_name
    ]
    if len(matching_meshes) != 1:
        errors.append(
            f"Expected one audited mesh named {object_name!r}, found {len(matching_meshes)}"
        )
    else:
        audited_mesh = matching_meshes[0]
        for key in ("vertex_count", "edge_count", "face_count", "triangle_count"):
            expected = audited_mesh.get(key)
            actual = mesh_counts.get(key)
            if expected is not None and int(expected) != int(actual or -1):
                errors.append(
                    f"Mesh {key} mismatch: audit={expected}, current={actual}"
                )

    try:
        candidate = find_candidate(audit_report, candidate_id)
    except ValueError as exc:
        errors.append(str(exc))
        candidate = {}
    if candidate and candidate.get("object_name") != object_name:
        errors.append(
            f"Candidate source object is {candidate.get('object_name')!r}, "
            f"not {object_name!r}"
        )
    indices = candidate.get("component_indices") if candidate else None
    if not isinstance(indices, list) or not indices:
        errors.append("Candidate has no component_indices")

    return {
        "compatible": not errors,
        "errors": errors,
        "audited_sha256": audited_sha,
        "current_sha256": asset_sha256,
        "candidate": candidate,
    }


def select_merge_result(
    results: Sequence[Mapping[str, Any]],
    *,
    minimum_silhouette_iou: float,
    maximum_bbox_relative_error: float,
) -> dict[str, Any] | None:
    """Select the smallest merge tolerance satisfying every repair gate."""

    passing = []
    for result in results:
        if not bool(result.get("closed_manifold")):
            continue
        if float(result.get("silhouette_iou", 0.0)) < minimum_silhouette_iou:
            continue
        if (
            float(result.get("bbox_relative_error", math.inf))
            > maximum_bbox_relative_error
        ):
            continue
        if result.get("material_histogram_preserved") is not True:
            continue
        if result.get("uv_layers_preserved") is not True:
            continue
        passing.append(dict(result))
    if not passing:
        return None
    return min(passing, key=lambda result: float(result.get("tolerance_ratio", math.inf)))


def select_cylindrical_skin_faces(
    face_descriptors: Mapping[int, Mapping[str, float]],
    component_faces: Sequence[Sequence[int]],
    *,
    lower_radius: float,
    base_radius: float,
    upper_radius: float,
    minimum_alignment: float,
    minimum_component_radial_area_fraction: float = 0.55,
) -> dict[str, Any]:
    """Select base-skin faces without swallowing raised grouser geometry.

    Disconnected components that are almost entirely confined to the base-radius
    band are safe to select in full. Mixed components are handled face by face,
    using center radius, radial-normal alignment, and the complete vertex-radius
    interval. This removes base-skin panels welded to a grouser while preserving
    the grouser top and side walls above the band.
    """

    if not lower_radius < base_radius < upper_radius:
        raise ValueError("Expected lower_radius < base_radius < upper_radius")
    band_width = upper_radius - lower_radius
    interval_margin = band_width * 0.35

    face_band_selection: set[int] = set()
    for raw_index, descriptor in face_descriptors.items():
        face_index = int(raw_index)
        center_radius = float(descriptor["radius"])
        alignment = float(descriptor["alignment"])
        vertex_min = float(descriptor["vertex_radius_min"])
        vertex_max = float(descriptor["vertex_radius_max"])
        interval_overlaps = vertex_max >= lower_radius and vertex_min <= upper_radius
        radial_span = vertex_max - vertex_min
        interval_compact = radial_span <= band_width * 2.5
        if (
            lower_radius <= center_radius <= upper_radius
            and alignment >= minimum_alignment
            and interval_overlaps
            and interval_compact
        ):
            face_band_selection.add(face_index)

    component_selection: set[int] = set()
    component_selection_faces: set[int] = set()
    component_scores: dict[int, dict[str, float | bool]] = {}
    for component_index, raw_faces in enumerate(component_faces):
        faces = [int(index) for index in raw_faces if int(index) in face_descriptors]
        if not faces:
            continue
        total_area = sum(
            max(float(face_descriptors[index]["area"]), 1e-12)
            for index in faces
        )
        radial_area = sum(
            max(float(face_descriptors[index]["area"]), 1e-12)
            for index in faces
            if index in face_band_selection
        )
        radial_fraction = radial_area / total_area
        maximum_vertex_radius = max(
            float(face_descriptors[index]["vertex_radius_max"])
            for index in faces
        )
        mean_radius = sum(
            float(face_descriptors[index]["radius"])
            * max(float(face_descriptors[index]["area"]), 1e-12)
            for index in faces
        ) / total_area
        confined_to_base_band = maximum_vertex_radius <= upper_radius + interval_margin
        selected = (
            radial_fraction >= minimum_component_radial_area_fraction
            and confined_to_base_band
            and lower_radius - interval_margin <= mean_radius <= upper_radius
        )
        component_scores[component_index] = {
            "total_area": total_area,
            "radial_area_fraction": radial_fraction,
            "mean_radius": mean_radius,
            "maximum_vertex_radius": maximum_vertex_radius,
            "confined_to_base_band": confined_to_base_band,
            "selected": selected,
        }
        if selected:
            component_selection.add(component_index)
            component_selection_faces.update(faces)

    skin_faces = component_selection_faces | face_band_selection
    return {
        "skin_face_indices": skin_faces,
        "skin_component_indices": component_selection,
        "component_selected_face_indices": component_selection_faces,
        "face_band_selected_face_indices": face_band_selection,
        "selection_overlap_face_indices": (
            component_selection_faces & face_band_selection
        ),
        "component_scores": component_scores,
    }


def select_detail_components_within_wheel_envelope(
    component_descriptors: Sequence[Mapping[str, Any]],
    *,
    axial_min: float,
    axial_max: float,
    outer_radius: float,
    axial_margin_width_ratio: float = 0.02,
    outer_detail_radius_ratio: float = 0.72,
) -> dict[str, Any]:
    """Reject tread fragments and inboard attachments beyond the wheel width.

    Hub and spoke components may protrude, but their center must remain safely
    inside the fitted wheel. Components centered on/beyond the inboard edge are
    suspension attachments rather than part of the canonical isolated wheel.
    """

    width = axial_max - axial_min
    if width <= 0.0 or outer_radius <= 0.0:
        raise ValueError("Invalid fitted wheel envelope")
    margin = width * axial_margin_width_ratio
    kept: set[int] = set()
    rejected: set[int] = set()
    for descriptor in component_descriptors:
        index = int(descriptor["index"])
        radial_min = float(descriptor["radial_min"])
        component_axial_min = float(descriptor["axial_min"])
        component_axial_max = float(descriptor["axial_max"])
        component_axial_mean = float(
            descriptor.get(
                "axial_mean", (component_axial_min + component_axial_max) * 0.5
            )
        )
        near_tread = radial_min >= outer_radius * outer_detail_radius_ratio
        outside_axial = (
            component_axial_max < axial_min - margin
            or component_axial_min > axial_max + margin
        )
        inboard_attachment = (
            component_axial_min < axial_min - margin
            and component_axial_mean < axial_min + margin
        )
        if (near_tread and outside_axial) or inboard_attachment:
            rejected.add(index)
        else:
            kept.add(index)
    return {
        "kept_component_indices": kept,
        "rejected_component_indices": rejected,
        "axial_margin": margin,
        "outer_detail_radius_threshold": outer_radius * outer_detail_radius_ratio,
    }


def _yes_no(value: Any) -> str:
    return "sì" if value is True else "no" if value is False else str(value)


def make_markdown_report(report: Mapping[str, Any]) -> str:
    status = report.get("status", "unknown")
    candidate = report.get("candidate", {})
    extraction = report.get("extraction", {})
    axis = report.get("canonical_transform", {}).get("axis", {})
    repair = report.get("repair", {})
    boolean = report.get("perforation", {}).get("boolean", {})
    mask = report.get("perforation", {}).get("mask", {})
    validation = report.get("validation", {})
    lines = [
        "# Preparazione ruota Curiosity",
        "",
        f"- **Stato:** `{status}`",
        f"- **Candidata:** `{candidate.get('id', 'n/a')}`",
        f"- **Strategia pelle:** `{repair.get('strategy', 'n/a')}`",
        f"- **Validazione:** `{_yes_no(validation.get('valid'))}`",
        "",
        "## Estrazione",
        "",
        f"- Componenti: **{extraction.get('component_count', 'n/a')}**",
        f"- Vertici: **{extraction.get('vertex_count', 'n/a')}**",
        f"- Facce: **{extraction.get('face_count', 'n/a')}**",
        f"- Materiali conservati: **{_yes_no(extraction.get('materials_preserved'))}**",
        f"- UV conservate: **{_yes_no(extraction.get('uv_preserved'))}**",
        f"- Normali custom conservate: **{_yes_no(extraction.get('custom_normals_preserved'))}**",
        f"- Silhouette IoU minima: `{extraction.get('minimum_silhouette_iou', 'n/a')}`",
        "",
        "## Sistema canonico",
        "",
        f"- Asse stimato: **{axis.get('axis_name', 'n/a')}**",
        f"- Lato esterno sorgente: `{axis.get('outboard_sign', 'n/a')}`",
        "- Convenzione finale: centro all’origine, asse ruota lungo X, outboard verso +X.",
        "",
        "## Repair della pelle",
        "",
        f"- Strategia scelta: **{repair.get('strategy', 'n/a')}**",
        f"- Raggio esterno: `{repair.get('outer_radius', 'n/a')}`",
        f"- Spessore: `{repair.get('wall_thickness', 'n/a')}`",
        f"- Boundary edge finali: `{repair.get('final_topology', {}).get('boundary_edge_count', 'n/a')}`",
        f"- Multi-face edge finali: `{repair.get('final_topology', {}).get('multi_face_edge_count', 'n/a')}`",
        "",
        "## Perforazione",
        "",
        f"- Boolean EXACT riuscito: **{_yes_no(boolean.get('success'))}**",
        f"- Pelle anomala manifold: **{_yes_no(boolean.get('closed_manifold'))}**",
        f"- Area maschera / ruota: `{mask.get('area_ratio', 'n/a')}`",
        f"- Cambiamento RGB dentro la maschera: `{mask.get('inside_change_ratio', 'n/a')}`",
        f"- Differenza RGB media dentro la maschera: `{mask.get('mean_inside_difference', 'n/a')}`",
        f"- Differenza esterna alla maschera dilatata: `{mask.get('outside_change_ratio', 'n/a')}`",
        "",
        "## Artefatti",
        "",
    ]
    for key, value in report.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    warnings = report.get("warnings", [])
    errors = report.get("errors", [])
    lines.extend(["", "## Warning ed errori", ""])
    if not warnings and not errors:
        lines.append("Nessun warning o errore registrato.")
    else:
        lines.extend(f"- Warning: {warning}" for warning in warnings)
        lines.extend(f"- Errore: {error}" for error in errors)
    lines.append("")
    return "\n".join(lines)
