from __future__ import annotations

import unittest

from src.blender_audit.core import (
    classify_asset,
    connected_components,
    deep_merge,
    repeat_counts,
    repeat_group_labels,
    score_wheel_candidate,
)


class BlenderAuditCoreTests(unittest.TestCase):
    def test_connected_components_includes_isolated_vertices(self):
        components = connected_components(6, [(0, 1), (1, 2), (3, 4)])
        self.assertEqual(components, [[0, 1, 2], [3, 4], [5]])

    def test_deep_merge_preserves_nested_defaults_and_inputs(self):
        base = {"render": {"width": 800, "height": 600}, "enabled": True}
        override = {"render": {"width": 400}}
        merged = deep_merge(base, override)
        self.assertEqual(
            merged,
            {
                "render": {"width": 400, "height": 600},
                "enabled": True,
            },
        )
        self.assertEqual(base["render"]["width"], 800)

    def test_geometry_only_wheel_candidate_can_pass_without_name_keyword(self):
        item = {
            "name": "mesh_042",
            "dimensions": [0.36, 0.52, 0.51],
            "center": [1.0, -0.8, 0.28],
            "vertex_count": 2400,
            "face_count": 2200,
        }
        scene_bbox = {"min": [-1.4, -1.1, 0.0], "max": [1.4, 1.1, 2.2]}
        result = score_wheel_candidate(
            item,
            scene_bbox,
            repeat_count=6,
            config={"candidate_threshold": 0.5},
        )
        self.assertTrue(result["passes_threshold"])
        self.assertFalse(result["signals"]["name_keyword"])
        self.assertGreater(result["signals"]["cylinder_like"], 0.8)
        self.assertEqual(result["signals"]["repeat_count"], 6)

    def test_repeat_counts_uses_geometry_not_names(self):
        items = [
            {
                "name": f"anonymous_{index}",
                "dimensions": [0.36, 0.52, 0.51],
                "vertex_count": 2400,
                "face_count": 2200,
            }
            for index in range(6)
        ]
        self.assertEqual(repeat_counts(items), [6, 6, 6, 6, 6, 6])
        self.assertEqual(len(set(repeat_group_labels(items))), 1)

    def test_classification_categories_a_to_d(self):
        self.assertEqual(classify_asset([candidate("mesh_object")])["category"], "A")
        self.assertEqual(
            classify_asset(
                [
                    candidate(
                        "disconnected_component",
                        separation="yes",
                    )
                ]
            )["category"],
            "B",
        )
        self.assertEqual(
            classify_asset(
                [
                    candidate(
                        "mesh_object",
                        duplicate="conditional",
                        boolean="no",
                        thickness="no",
                    )
                ]
            )["category"],
            "C",
        )
        self.assertEqual(classify_asset([])["category"], "D")


def candidate(
    source_kind: str,
    *,
    duplicate: str = "yes",
    boolean: str = "yes",
    thickness: str = "yes",
    separation: str = "not_needed",
):
    return {
        "score": 0.84,
        "source_kind": source_kind,
        "duplication": {"verdict": duplicate},
        "boolean_difference": {"verdict": boolean},
        "skin_thickness": {"verdict": thickness},
        "automatic_separation": {"verdict": separation},
    }


if __name__ == "__main__":
    unittest.main()
