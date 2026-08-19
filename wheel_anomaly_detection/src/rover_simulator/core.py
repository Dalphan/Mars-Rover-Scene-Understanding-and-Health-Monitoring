from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    candidate_id: str
    local_position: tuple[float, float, float]
    side: str
    longitudinal: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "candidate_id": self.candidate_id,
            "local_position": list(self.local_position),
            "side": self.side,
            "longitudinal": self.longitudinal,
        }


def _axis_descriptor(
    axis: Mapping[str, Any], level_count: int
) -> Mapping[str, Any]:
    matches = [
        descriptor
        for descriptor in axis.get("descriptors", [])
        if int(descriptor.get("level_count", -1)) == level_count
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one axis with {level_count} repeated levels, found {len(matches)}"
        )
    return matches[0]


def derive_four_probe_specs(
    candidates: Sequence[Mapping[str, Any]],
    selected_candidate_id: str,
    canonical_axis: Mapping[str, Any],
    wheel_radius: float,
) -> list[ProbeSpec]:
    """Derive the four static corner probes from the audited 2x3 wheel layout."""

    if len(candidates) != 6:
        raise ValueError(f"Expected six audited wheels, found {len(candidates)}")
    selected = next(
        (
            candidate
            for candidate in candidates
            if str(candidate.get("id")) == selected_candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"Selected candidate not found: {selected_candidate_id}")
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")

    lateral_axis = int(_axis_descriptor(canonical_axis, 2)["axis"])
    longitudinal_axis = int(_axis_descriptor(canonical_axis, 3)["axis"])
    vertical_axes = {0, 1, 2} - {lateral_axis, longitudinal_axis}
    if len(vertical_axes) != 1:
        raise ValueError("Could not infer the vertical wheel-layout axis")
    vertical_axis = vertical_axes.pop()
    outboard_sign = float(canonical_axis.get("outboard_sign", 1.0))
    selected_center = [float(value) for value in selected["center"]]

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        center = [float(value) for value in candidate["center"]]
        records.append(
            {
                "id": str(candidate["id"]),
                "x": outboard_sign
                * (center[lateral_axis] - selected_center[lateral_axis]),
                "y": center[longitudinal_axis] - selected_center[longitudinal_axis],
                "z": center[vertical_axis] - selected_center[vertical_axis],
            }
        )

    lateral_midpoint = (
        min(record["x"] for record in records)
        + max(record["x"] for record in records)
    ) * 0.5
    sides = {
        "Left": [record for record in records if record["x"] < lateral_midpoint],
        "Right": [record for record in records if record["x"] >= lateral_midpoint],
    }
    if any(len(group) != 3 for group in sides.values()):
        raise ValueError("Audited wheels do not form two lateral groups of three")

    probes: list[ProbeSpec] = []
    for side, group in sides.items():
        for longitudinal, record in (
            ("Front", max(group, key=lambda value: value["y"])),
            ("Rear", min(group, key=lambda value: value["y"])),
        ):
            probes.append(
                ProbeSpec(
                    name=f"GroundProbe_{longitudinal}_{side}",
                    candidate_id=record["id"],
                    local_position=(
                        float(record["x"]),
                        float(record["y"]),
                        float(record["z"] - wheel_radius),
                    ),
                    side=side.lower(),
                    longitudinal=longitudinal.lower(),
                )
            )
    return sorted(probes, key=lambda probe: probe.name)
