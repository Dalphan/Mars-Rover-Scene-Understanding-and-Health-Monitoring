import copy
import json
import unittest
from pathlib import Path

from src.wheel_preparation.pose_sampling import (
    conditioned_roll_degrees,
    equivalent_travel_m,
    normalize_degrees,
    validate_pose_sampling_config,
)


class WheelPoseSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "configs" / "blender" / "wheel_pose_sampling.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_contract(self):
        contract = validate_pose_sampling_config(self.config)
        self.assertEqual(contract["wheel_count"], 6)
        self.assertEqual(contract["normal_phase_count"], 8)
        self.assertEqual(contract["anomaly_jitter_count"], 3)
        self.assertIn("non_carrier_occlusion", contract["hard_visibility_gates"])

    def test_conditioning_and_travel(self):
        self.assertEqual(conditioned_roll_degrees(-90.0, 20.0, 1.0), 110.0)
        self.assertEqual(conditioned_roll_degrees(-90.0, 20.0, -1.0), -110.0)
        self.assertAlmostEqual(equivalent_travel_m(45.0, 0.242647), 0.190576, places=5)
        self.assertEqual(normalize_degrees(315.0), -45.0)

    def test_anomaly_metadata_cannot_be_optional(self):
        invalid = copy.deepcopy(self.config)
        invalid["anomaly_sampling"]["require_anchor"] = False
        with self.assertRaises(ValueError):
            validate_pose_sampling_config(invalid)


if __name__ == "__main__":
    unittest.main()
