from __future__ import annotations

import unittest

from src.rover_simulator.core import derive_four_probe_specs


class RoverSimulatorCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {"id": "left_front", "center": [-1.0, 1.2, 0.02]},
            {"id": "left_middle", "center": [-1.0, 0.0, 0.01]},
            {"id": "left_rear", "center": [-1.0, -1.1, 0.03]},
            {"id": "right_front", "center": [1.0, 1.2, 0.00]},
            {"id": "right_middle", "center": [1.0, 0.0, 0.00]},
            {"id": "right_rear", "center": [1.0, -1.1, 0.01]},
        ]
        self.axis = {
            "outboard_sign": 1,
            "descriptors": [
                {"axis": 0, "level_count": 2},
                {"axis": 1, "level_count": 3},
                {"axis": 2, "level_count": 5},
            ],
        }

    def test_derives_four_corner_probes_and_excludes_middle_wheels(self) -> None:
        probes = derive_four_probe_specs(
            self.candidates, "right_middle", self.axis, 0.25
        )
        self.assertEqual(len(probes), 4)
        self.assertEqual(
            {probe.candidate_id for probe in probes},
            {"left_front", "left_rear", "right_front", "right_rear"},
        )

    def test_positions_are_relative_to_selected_wheel_and_touch_ground(self) -> None:
        probes = derive_four_probe_specs(
            self.candidates, "right_middle", self.axis, 0.25
        )
        front_right = next(
            probe for probe in probes if probe.name == "GroundProbe_Front_Right"
        )
        self.assertEqual(front_right.local_position, (0.0, 1.2, -0.25))

    def test_rejects_incomplete_wheel_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "six audited wheels"):
            derive_four_probe_specs(
                self.candidates[:-1], "right_middle", self.axis, 0.25
            )


if __name__ == "__main__":
    unittest.main()
