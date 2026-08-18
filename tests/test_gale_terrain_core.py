import unittest

from src.gale_terrain.core import CropBounds, crs_parameters_equivalent, expected_resolution, normalized_uv, validate_grid_contract
from src.gale_terrain.validation import validate_crop_report


class GaleTerrainCoreTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "crop": {
                "projected_bounds_m": [-128.0, -128.0, 128.0, 128.0],
                "dtm_shape": [256, 256],
                "ortho_shape": [1024, 1024],
                "heightfield_shape": [257, 257],
                "minimum_valid_fraction": 0.999,
                "maximum_vertical_exaggeration": 1.0,
            }
        }

    def test_metric_grid_contract(self):
        contract = validate_grid_contract(self.config)
        self.assertEqual(contract["dtm_resolution_m"], (1.0, 1.0))
        self.assertEqual(contract["ortho_resolution_m"], (0.25, 0.25))
        self.assertEqual(contract["resolution_ratio"], 4)

    def test_uv_uses_projected_crop(self):
        bounds = CropBounds(-128.0, -128.0, 128.0, 128.0)
        self.assertEqual(normalized_uv(-128.0, -128.0, bounds), (0.0, 0.0))
        self.assertEqual(normalized_uv(128.0, 128.0, bounds), (1.0, 1.0))
        self.assertEqual(normalized_uv(0.0, 0.0, bounds), (0.5, 0.5))

    def test_invalid_vertical_exaggeration_is_rejected(self):
        self.config["crop"]["maximum_vertical_exaggeration"] = 1.5
        with self.assertRaises(ValueError):
            validate_grid_contract(self.config)

    def test_crs_equivalence_ignores_datum_names_but_not_parameters(self):
        first = {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": 137.41, "x_0": 0, "y_0": 0, "R": 3396190, "units": "m"}
        second = dict(first)
        self.assertTrue(crs_parameters_equivalent(first, second))
        second["R"] = 3396200
        self.assertFalse(crs_parameters_equivalent(first, second))

    def test_report_validation_passes_exact_coregistration(self):
        report = {
            "sources": {role: {"sha256": "abc"} for role in ("dtm", "irb_ortho", "irb_label", "mrgb", "mrgb_label")},
            "crop": {
                "projected_bounds_m": [-128.0, -128.0, 128.0, 128.0],
                "dtm_valid_fraction": 1.0,
                "irb_valid_fraction": 1.0,
                "vertical_exaggeration": 1.0,
            },
            "coregistration": {"dtm_irb_crs_equivalent": True, "exact_shared_bounds": True, "integer_resolution_ratio": True, "mrgb": {"passed": False}},
            "texture_selection": {"chosen": "IRB"},
            "visualization": {"spatial_source": "IRB only", "mrgb_spatial_data_used": False},
        }
        self.assertEqual(validate_crop_report(self.config, report), [])

    def test_report_rejects_mrgb_selection_when_coregistration_failed(self):
        report = {
            "sources": {role: {"sha256": "abc"} for role in ("dtm", "irb_ortho", "irb_label", "mrgb", "mrgb_label")},
            "crop": {"projected_bounds_m": [-128.0, -128.0, 128.0, 128.0], "dtm_valid_fraction": 1.0, "irb_valid_fraction": 1.0, "vertical_exaggeration": 1.0},
            "coregistration": {"dtm_irb_crs_equivalent": True, "exact_shared_bounds": True, "integer_resolution_ratio": True, "mrgb": {"passed": False}},
            "texture_selection": {"chosen": "MRGB"},
            "visualization": {"spatial_source": "IRB only", "mrgb_spatial_data_used": False},
        }
        self.assertIn("Texture selection is inconsistent with the MRGB coregistration result", validate_crop_report(self.config, report))

    def test_report_rejects_mrgb_geometry_in_irb_fallback(self):
        report = {
            "sources": {role: {"sha256": "abc"} for role in ("dtm", "irb_ortho", "irb_label", "mrgb", "mrgb_label")},
            "crop": {"projected_bounds_m": [-128.0, -128.0, 128.0, 128.0], "dtm_valid_fraction": 1.0, "irb_valid_fraction": 1.0, "vertical_exaggeration": 1.0},
            "coregistration": {"dtm_irb_crs_equivalent": True, "exact_shared_bounds": True, "integer_resolution_ratio": True, "mrgb": {"passed": False}},
            "texture_selection": {"chosen": "IRB"},
            "visualization": {"spatial_source": "IRB and MRGB", "mrgb_spatial_data_used": True},
        }
        self.assertIn("IRB fallback must not use MRGB spatial data", validate_crop_report(self.config, report))


if __name__ == "__main__":
    unittest.main()
