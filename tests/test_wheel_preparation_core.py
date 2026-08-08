import unittest

from src.wheel_preparation.core import (
    canonical_axis_from_candidates,
    find_candidate,
    select_cylindrical_skin_faces,
    select_detail_components_within_wheel_envelope,
    select_merge_result,
    validate_audit_compatibility,
)
from src.wheel_preparation.perforation import (
    FAMILIES,
    ORIENTATIONS,
    build_perforation_library,
    generate_perforation_profile,
    polygon_self_intersects,
    validate_perforation_profile,
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

    def test_face_level_skin_selection_preserves_raised_grouser_faces(self):
        descriptors = {
            0: {
                "radius": 1.00,
                "alignment": 0.99,
                "area": 2.0,
                "vertex_radius_min": 0.99,
                "vertex_radius_max": 1.01,
            },
            1: {
                "radius": 1.01,
                "alignment": 0.96,
                "area": 2.0,
                "vertex_radius_min": 1.00,
                "vertex_radius_max": 1.02,
            },
            2: {
                "radius": 1.18,
                "alignment": 0.98,
                "area": 0.7,
                "vertex_radius_min": 1.16,
                "vertex_radius_max": 1.20,
            },
            3: {
                "radius": 1.09,
                "alignment": 0.05,
                "area": 0.7,
                "vertex_radius_min": 1.01,
                "vertex_radius_max": 1.18,
            },
            4: {
                "radius": 1.00,
                "alignment": 0.94,
                "area": 1.0,
                "vertex_radius_min": 0.99,
                "vertex_radius_max": 1.01,
            },
        }
        result = select_cylindrical_skin_faces(
            descriptors,
            [[0, 1, 2, 3], [4]],
            lower_radius=0.96,
            base_radius=1.00,
            upper_radius=1.04,
            minimum_alignment=0.35,
        )
        self.assertEqual(result["skin_face_indices"], {0, 1, 4})
        self.assertEqual(result["skin_component_indices"], {1})
        self.assertNotIn(2, result["skin_face_indices"])
        self.assertNotIn(3, result["skin_face_indices"])

    def test_skin_selection_rejects_face_spanning_outside_radial_band(self):
        descriptors = {
            0: {
                "radius": 1.0,
                "alignment": 1.0,
                "area": 1.0,
                "vertex_radius_min": 0.80,
                "vertex_radius_max": 1.20,
            }
        }
        result = select_cylindrical_skin_faces(
            descriptors,
            [[0]],
            lower_radius=0.96,
            base_radius=1.00,
            upper_radius=1.04,
            minimum_alignment=0.35,
        )
        self.assertEqual(result["skin_face_indices"], set())

    def test_detail_envelope_rejects_only_outer_axial_fragments(self):
        descriptors = [
            {"index": 0, "radial_min": 0.80, "axial_min": -0.2, "axial_max": 0.2},
            {"index": 1, "radial_min": 0.85, "axial_min": -0.7, "axial_max": -0.6},
            {"index": 2, "radial_min": 0.20, "axial_min": -0.8, "axial_max": -0.6},
            {"index": 3, "radial_min": 0.90, "axial_min": -0.505, "axial_max": -0.501},
            {"index": 4, "radial_min": 0.20, "axial_min": -0.8, "axial_max": 0.2, "axial_mean": -0.2},
        ]
        result = select_detail_components_within_wheel_envelope(
            descriptors,
            axial_min=-0.5,
            axial_max=0.5,
            outer_radius=1.0,
            axial_margin_width_ratio=0.02,
            outer_detail_radius_ratio=0.72,
        )
        self.assertEqual(result["rejected_component_indices"], {1, 2})
        self.assertEqual(result["kept_component_indices"], {0, 3, 4})

    def test_irregular_library_is_deterministic_and_complete(self):
        first = build_perforation_library(diameter=0.4852948379, base_seed=41000)
        second = build_perforation_library(diameter=0.4852948379, base_seed=41000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual({item["family"] for item in first}, set(FAMILIES))
        self.assertEqual({item["orientation"] for item in first}, set(ORIENTATIONS))
        self.assertTrue(all(not validate_perforation_profile(item) for item in first))

    def test_irregular_profile_is_elongated_and_non_self_intersecting(self):
        profile = generate_perforation_profile(
            variant_id="test", family="branched_tear", orientation="axial",
            size="medium", diameter=1.0, seed=41009,
        )
        self.assertGreaterEqual(profile["quality"]["aspect_ratio"], 2.0)
        self.assertLessEqual(profile["quality"]["aspect_ratio"], 4.5)
        self.assertLessEqual(profile["quality"]["circularity"], 0.78)
        self.assertFalse(polygon_self_intersects(profile["points_axial_tangent"]))

    def test_orientation_changes_long_axis_without_changing_seed(self):
        axial = generate_perforation_profile(
            variant_id="a", family="jagged_slit", orientation="axial",
            size="small", diameter=1.0, seed=41001,
        )
        circumferential = generate_perforation_profile(
            variant_id="b", family="jagged_slit", orientation="circumferential",
            size="small", diameter=1.0, seed=41001,
        )
        self.assertAlmostEqual(
            axial["quality"]["width"], circumferential["quality"]["height"], places=6
        )
        self.assertAlmostEqual(
            axial["quality"]["height"], circumferential["quality"]["width"], places=6
        )


if __name__ == "__main__":
    unittest.main()
