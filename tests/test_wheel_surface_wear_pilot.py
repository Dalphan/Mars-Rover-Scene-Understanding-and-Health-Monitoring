from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

from scripts.host.run_wheel_surface_wear_pilot import _generate_nested_masks


class SurfaceWearMaskTests(unittest.TestCase):
    def _config(self) -> dict:
        return {
            "mask_generation": {
                "resolution": 128,
                "seed": 731903,
                "dominant_direction_degrees": 72.0,
                "direction_spread_degrees": 24.0,
                "cross_scratch_probability": 0.22,
                "length_pixels": [4, 14],
                "width_pixels": [1, 2],
                "intensity": [180, 255],
                "coverage_threshold": 32,
            },
            "variants": [
                {"id": "surface_current", "wear_enabled": False, "target_mask_coverage": 0.0, "mask_filename": None},
                {"id": "wear_light", "wear_enabled": True, "target_mask_coverage": 0.01, "mask_filename": "light.png"},
                {"id": "wear_evident", "wear_enabled": True, "target_mask_coverage": 0.03, "mask_filename": "evident.png"},
            ],
        }

    def test_masks_are_deterministic_and_nested(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = _generate_nested_masks(self._config(), Path(first_dir))
            second = _generate_nested_masks(self._config(), Path(second_dir))
            self.assertEqual(first["wear_light"]["sha256"], second["wear_light"]["sha256"])
            self.assertEqual(first["wear_evident"]["sha256"], second["wear_evident"]["sha256"])
            self.assertGreaterEqual(first["wear_light"]["coverage"], 0.01)
            self.assertGreaterEqual(first["wear_evident"]["coverage"], 0.03)
            light = Image.open(first["wear_light"]["path"]).convert("L")
            evident = Image.open(first["wear_evident"]["path"]).convert("L")
            removed_prefix = ImageChops.subtract(light, evident)
            self.assertIsNone(removed_prefix.getbbox())


if __name__ == "__main__":
    unittest.main()
