import unittest

from src.wheel_preparation.core import (
    canonical_axis_from_candidates,
    find_candidate,
    select_merge_result,
    validate_audit_compatibility,
)


def candidate(candidate_id, center):
    return {
        "id": candidate_id,
        "object_name": "MSL",
        "center": center,
        "component_indices": [1, 2],
    }


class WheelPreparationCoreTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            candidate("wheel_candidate_01", [-1.0, -1.0, 0.25]),
            candidate("wheel_candidate_02", [-1.0, 0.0, 0.25]),
            candidate("wheel_candidate_03", [-1.0, 1.0, 0.25]),
            candidate("wheel_candidate_04", [1.0, -1.0, 0.25]),
            candidate("wheel_candidate_05", [1.0, 0.0, 0.25]),
            candidate("wheel_candidate_06", [1.0, 1.0, 0.25]),
        ]

    def test_finds_candidate_and_rejects_unknown_id(self):
        report = {"wheel_detection": {"candidates": self.candidates}}
        self.assertEqual(
            find_candidate(report, "wheel_candidate_05")["center"],
            [1.0, 0.0, 0.25],
        )
        with self.assertRaisesRegex(ValueError, "not found"):
            find_candidate(report, "wheel_candidate_99")

    def test_infers_lateral_axis_from_two_repeated_levels(self):
        result = canonical_axis_from_candidates(
            self.candidates, self.candidates[4]
        )
        self.assertEqual(result["axis_index"], 0)
        self.assertEqual(result["axis_name"], "X")
        self.assertEqual(result["outboard_sign"], 1)

        left_result = canonical_axis_from_candidates(
            self.candidates, self.candidates[1]
        )
        self.assertEqual(left_result["outboard_sign"], -1)

    def test_selects_smallest_passing_merge_tolerance(self):
        base = {
            "closed_manifold": True,
            "silhouette_iou": 1.0,
            "bbox_relative_error": 0.0,
            "material_histogram_preserved": True,
            "uv_layers_preserved": True,
        }
        results = [
            {**base, "tolerance_ratio": 1e-5},
            {**base, "tolerance_ratio": 1e-6},
            {
                **base,
                "tolerance_ratio": 1e-7,
                "closed_manifold": False,
            },
        ]
        selected = select_merge_result(
            results,
            minimum_silhouette_iou=0.995,
            maximum_bbox_relative_error=0.001,
        )
        self.assertEqual(selected["tolerance_ratio"], 1e-6)

    def test_merge_selection_rejects_preservation_failure(self):
        result = {
            "tolerance_ratio": 1e-6,
            "closed_manifold": True,
            "silhouette_iou": 1.0,
            "bbox_relative_error": 0.0,
            "material_histogram_preserved": True,
            "uv_layers_preserved": False,
        }
        self.assertIsNone(
            select_merge_result(
                [result],
                minimum_silhouette_iou=0.995,
                maximum_bbox_relative_error=0.001,
            )
        )

    def test_audit_compatibility_detects_checksum_mismatch(self):
        report = {
            "status": "completed",
            "asset": {"sha256_before": "expected"},
            "environment": {"blender_version_tuple": [5, 2, 0]},
            "meshes": [
                {
                    "object_name": "MSL",
                    "vertex_count": 10,
                    "edge_count": 20,
                    "face_count": 15,
                    "triangle_count": 15,
                }
            ],
            "wheel_detection": {"candidates": self.candidates},
        }
        result = validate_audit_compatibility(
            report,
            asset_sha256="different",
            blender_version=[5, 2, 1],
            object_name="MSL",
            mesh_counts={
                "vertex_count": 10,
                "edge_count": 20,
                "face_count": 15,
                "triangle_count": 15,
            },
            candidate_id="wheel_candidate_05",
        )
        self.assertFalse(result["compatible"])
        self.assertTrue(
            any("SHA-256 mismatch" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
