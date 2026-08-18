import copy
import json
import unittest
from pathlib import Path

from src.microterrain.camera_poses import diagonal_fov_deg, validate_wheel_camera_pose_config


class WheelCameraPoseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[1] / "configs" / "blender" / "wheel_camera_poses.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_pilot_contract_and_fov(self):
        contract = validate_wheel_camera_pose_config(self.config)
        self.assertEqual(contract["pose_count"], 4)
        self.assertTrue(34.0 <= contract["diagonal_fov_deg"] <= 39.5)
        self.assertAlmostEqual(
            contract["diagonal_fov_deg"],
            diagonal_fov_deg(21.0, 11.8, [1600, 1200]),
        )
        self.assertTrue(self.config["camera"]["pilot_fill_light"]["enabled"])
        self.assertNotIn("B_outer_oblique", contract["pose_ids"])
        self.assertTrue(self.config["pilot"]["terrain_color_grade"]["enabled"])

    def test_rejects_global_or_incomplete_target_offset(self):
        invalid = copy.deepcopy(self.config)
        invalid["poses"][0]["target_offset_m"] = {"x": 0, "y": 0, "z": 0}
        with self.assertRaises(ValueError):
            validate_wheel_camera_pose_config(invalid)

    def test_rejects_non_mahli_fov(self):
        invalid = copy.deepcopy(self.config)
        invalid["camera"]["sensor_width_mm"] = 20.0
        with self.assertRaises(ValueError):
            validate_wheel_camera_pose_config(invalid)

    def test_rejects_camera_frame_that_can_roll_with_wheel(self):
        invalid = copy.deepcopy(self.config)
        invalid["camera"].pop("rover_forward_world")
        with self.assertRaises(ValueError):
            validate_wheel_camera_pose_config(invalid)


if __name__ == "__main__":
    unittest.main()
