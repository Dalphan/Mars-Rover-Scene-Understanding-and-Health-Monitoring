import copy
import unittest

import numpy as np

from src.microterrain.clasts import FAMILIES, generate_clast_scatter, validate_level2_config
from src.microterrain.level2_validation import REQUIRED_RENDERS, validate_level2_report


class MicroterrainClastTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "microterrain": {
                "enabled": True, "level": 2, "patch_size_m": 0.08, "grid_spacing_m": 0.002, "edge_transition_m": 0.01, "seed": 42,
                "meso_relief": {
                    "enabled": True,
                    "large_scale": {"wavelength_m": 0.04, "amplitude_m": 0.002, "components": 3},
                    "medium_scale": {"wavelength_m": 0.016, "amplitude_m": 0.0007, "components": 4},
                    "fine_scale": {"wavelength_m": 0.006, "amplitude_m": 0.0002, "components": 5},
                    "anisotropy": 0.5, "ripple_strength": 0.2, "ripple_wavelength_m": 0.02, "ripple_direction_deg": 30,
                    "depression_density_m2": 100, "depression_radius_m": [0.003, 0.008], "depression_depth_m": [0.0002, 0.0006],
                    "crust_patch_density_m2": 100, "crust_radius_m": [0.005, 0.012], "crust_flattening": 0.6, "crust_raise_m": 0.0001,
                    "maximum_abs_displacement_m": 0.01,
                },
                "clasts": {
                    "enabled": True, "library_variants": {"fine_grains": 3, "fragments": 3, "coarse_clasts": 4},
                    "fine_grains": {"density_m2": 1000, "size_min_m": 0.0005, "size_max_m": 0.002, "maximum_tilt_deg": 35, "buried_fraction": [0.25, 0.60]},
                    "fragments": {"density_m2": 300, "size_min_m": 0.001, "size_max_m": 0.005, "maximum_tilt_deg": 60, "buried_fraction": [0.22, 0.55]},
                    "coarse_clasts": {
                        "density_m2": 1562.5, "size_min_m": 0.002, "size_max_m": 0.006, "maximum_tilt_deg": 45,
                        "minimum_center_spacing_factor": 0.55,
                        "size_classes": [
                            {"name": "small", "fraction": 0.7, "size_min_m": 0.002, "size_max_m": 0.003, "buried_fraction": [0.25, 0.60]},
                            {"name": "medium", "fraction": 0.2, "size_min_m": 0.003, "size_max_m": 0.0045, "buried_fraction": [0.20, 0.50]},
                            {"name": "large", "fraction": 0.1, "size_min_m": 0.0045, "size_max_m": 0.006, "buried_fraction": [0.15, 0.40]},
                        ],
                    },
                    "clustering_strength": 0.5, "microrelief_correlation": 0.3,
                    "cluster_wavelength_m": 0.04, "edge_margin_m": 0.004,
                    "maximum_instances": 100, "material_palette": [],
                },
            }
        }
        self.axis = np.linspace(-0.04, 0.04, 41, dtype=np.float32)
        x, y = np.meshgrid(self.axis, self.axis)
        self.displacement = (0.0005 * np.sin(80 * x) * np.cos(70 * y)).astype(np.float32)
        self.surface = self.displacement.copy()

    def test_contract_counts_density_times_area(self):
        contract = validate_level2_config(self.config)
        self.assertEqual(contract["counts"], {"fine_grains": 6, "fragments": 2, "coarse_clasts": 10})
        self.assertEqual(contract["coarse_class_counts"], {"small": 7, "medium": 2, "large": 1})
        self.assertEqual(contract["total_instances"], 18)

    def test_scatter_is_deterministic_and_in_bounds(self):
        first, first_metrics = generate_clast_scatter(self.axis, self.axis, self.surface, self.displacement, self.config, np)
        second, second_metrics = generate_clast_scatter(self.axis, self.axis, self.surface, self.displacement, self.config, np)
        self.assertEqual(first_metrics["signature_sha256"], second_metrics["signature_sha256"])
        for family in FAMILIES:
            np.testing.assert_array_equal(first[family]["positions"], second[family]["positions"])
            positions = first[family]["positions"]
            self.assertTrue(np.all(positions[:, 0] >= self.axis[0]))
            self.assertTrue(np.all(positions[:, 0] <= self.axis[-1]))
            self.assertTrue(np.all(first[family]["positions"][:, 0] - 0.5 * first[family]["characteristic_size_m"] >= self.axis[0] + 0.004))
            self.assertTrue(np.all(first[family]["positions"][:, 0] + 0.5 * first[family]["characteristic_size_m"] <= self.axis[-1] - 0.004))
        coarse_classes = first["coarse_clasts"]["size_class_index"]
        self.assertEqual([int(np.count_nonzero(coarse_classes == index)) for index in range(3)], [7, 2, 1])
        coarse_sizes = first["coarse_clasts"]["characteristic_size_m"]
        configured_classes = self.config["microterrain"]["clasts"]["coarse_clasts"]["size_classes"]
        for class_index, entry in enumerate(configured_classes):
            selected = np.sort(coarse_sizes[coarse_classes == class_index])
            logarithmic_fraction = (np.log(selected) - np.log(entry["size_min_m"])) / (np.log(entry["size_max_m"]) - np.log(entry["size_min_m"]))
            count = len(selected)
            self.assertTrue(np.all(logarithmic_fraction >= np.arange(count) / count - 1e-6))
            self.assertTrue(np.all(logarithmic_fraction <= (np.arange(count) + 1) / count + 1e-6))
        self.assertTrue(np.all(first["fine_grains"]["burial"] >= 0.25))
        self.assertTrue(np.all(first["fine_grains"]["burial"] <= 0.60))

    def test_size_aware_exclusion_keeps_full_rock_outside_wheel_zone(self):
        zone = (0.0, 0.0, 0.008, 0.010)
        families, _ = generate_clast_scatter(self.axis, self.axis, self.surface, self.displacement, self.config, np, [zone])
        for values in families.values():
            positions = values["positions"]
            radius = 0.5 * values["characteristic_size_m"]
            ellipse = (positions[:, 0] / (zone[2] + radius)) ** 2 + (positions[:, 1] / (zone[3] + radius)) ** 2
            self.assertTrue(np.all(ellipse >= 1.0))

    def test_seed_changes_scatter_without_changing_counts(self):
        _, first = generate_clast_scatter(self.axis, self.axis, self.surface, self.displacement, self.config, np)
        changed = copy.deepcopy(self.config)
        changed["microterrain"]["seed"] = 43
        _, second = generate_clast_scatter(self.axis, self.axis, self.surface, self.displacement, changed, np)
        self.assertNotEqual(first["signature_sha256"], second["signature_sha256"])
        self.assertEqual(first["counts"], second["counts"])

    def test_performance_gate_rejects_excess_instances(self):
        self.config["microterrain"]["clasts"]["maximum_instances"] = 8
        with self.assertRaises(ValueError):
            validate_level2_config(self.config)

    def test_level2_report_gate(self):
        contract = validate_level2_config(self.config)
        configured = self.config["microterrain"]["clasts"]
        report = {
            "level": 2,
            "clasts": {
                "counts": contract["counts"], "total_instances": contract["total_instances"],
                "signature_sha256": "a" * 64,
                "family_signatures_sha256": {family: "b" * 64 for family in FAMILIES},
                "size_ranges_m": {family: [float(configured[family]["size_min_m"]), float(configured[family]["size_max_m"])] for family in FAMILIES},
                "burial_ranges": contract["burial_ranges"],
                "coarse_size_class_counts": contract["coarse_class_counts"],
                "coarse_size_sampling": "jittered_log_stratified_per_class",
                "size_aware_exclusion": True, "boundary_radius_guard": True,
                "exclusion_zones": [[0.0, 0.0, 0.01, 0.01]],
            },
            "instancing": {"geometry_nodes": True, "instances_realized": False, "scatter_object_count": 3},
            "library": {"source_mesh_count": 10},
            "scope": {"level3_shading_enabled": False},
            "renders": {name: "x.png" for name in REQUIRED_RENDERS},
        }
        self.assertEqual(validate_level2_report(self.config, report), [])


if __name__ == "__main__":
    unittest.main()
