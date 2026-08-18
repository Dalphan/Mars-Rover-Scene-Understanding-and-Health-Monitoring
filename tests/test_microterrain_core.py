import unittest

import numpy as np

from src.microterrain.core import derive_seed, generate_meso_relief, validate_level1_config
from src.microterrain.validation import validate_level1_report


class MicroterrainCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "microterrain": {
                "enabled": True, "level": 1, "patch_size_m": 0.08, "grid_spacing_m": 0.002, "edge_transition_m": 0.01, "seed": 42,
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
            }
        }

    def test_grid_contract(self):
        contract = validate_level1_config(self.config)
        self.assertEqual(contract["samples"], 41)
        self.assertEqual(contract["vertex_count"], 1681)

    def test_seed_derivation_is_stable_and_namespaced(self):
        self.assertEqual(derive_seed(42, "large"), derive_seed(42, "large"))
        self.assertNotEqual(derive_seed(42, "large"), derive_seed(42, "fine"))

    def test_relief_is_deterministic_and_edge_is_zero(self):
        axis = np.linspace(-0.04, 0.04, 41, dtype=np.float32)
        first, first_metrics = generate_meso_relief(axis, axis, self.config, np)
        second, second_metrics = generate_meso_relief(axis, axis, self.config, np)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_metrics["signature_sha256"], second_metrics["signature_sha256"])
        self.assertEqual(float(np.abs(first[[0, -1], :]).max()), 0.0)
        self.assertEqual(float(np.abs(first[:, [0, -1]]).max()), 0.0)

    def test_seed_changes_geometry(self):
        axis = np.linspace(-0.04, 0.04, 41, dtype=np.float32)
        first, _ = generate_meso_relief(axis, axis, self.config, np)
        self.config["microterrain"]["seed"] = 43
        second, _ = generate_meso_relief(axis, axis, self.config, np)
        self.assertFalse(np.array_equal(first, second))

    def test_report_scope_gate(self):
        contract = validate_level1_config(self.config)
        report = {
            "level": 1,
            "patch": {"vertex_count": contract["vertex_count"], "face_count": contract["face_count"]},
            "metrics": {"signature_sha256": "a" * 64, "maximum_absolute_displacement_m": 0.005, "edge_max_abs_displacement_m": 0.0, "clipped_fraction": 0.0},
            "macroterrain": {"source_modified": False},
            "scope": {"level2_clasts_enabled": False, "level3_shading_enabled": False},
            "renders": {name: "x.png" for name in ("L0_macro_only", "L1_meso_relief", "L0_terrain_10cm", "L1_terrain_10cm", "L0_terrain_30cm", "L1_terrain_30cm")},
        }
        self.assertEqual(validate_level1_report(self.config, report), [])


if __name__ == "__main__":
    unittest.main()
