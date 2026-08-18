from __future__ import annotations

import json
import math
import copy
import unittest
from pathlib import Path

from src.wheel_preparation.domain_randomization import (
    preview_tokens,
    sample_domain_randomization,
    select_stratified_preview,
    validate_domain_randomization_config,
)


class DomainRandomizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.config = json.loads((root / "configs" / "blender" / "domain_randomization.json").read_text(encoding="utf-8"))

    def test_config_and_sampling_are_deterministic(self) -> None:
        validate_domain_randomization_config(self.config)
        self.assertEqual(sample_domain_randomization(self.config, 17), sample_domain_randomization(self.config, 17))
        self.assertNotEqual(sample_domain_randomization(self.config, 17), sample_domain_randomization(self.config, 18))

    def test_vertical_contact_contract_is_fail_closed(self) -> None:
        invalid = copy.deepcopy(self.config)
        invalid["gates"]["terrain_alignment"]["vertical_contact"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "vertical wheel contact"):
            validate_domain_randomization_config(invalid)
        invalid = copy.deepcopy(self.config)
        invalid["gates"]["terrain_alignment"]["vertical_contact"]["maximum_penetration_m"] = 0.02
        with self.assertRaisesRegex(ValueError, "clearance bounds"):
            validate_domain_randomization_config(invalid)

    def test_all_jitter_stays_inside_hard_bounds(self) -> None:
        for index in range(500):
            sample = sample_domain_randomization(self.config, index)
            camera = sample["camera_jitter"]
            position_length = math.sqrt(sum(value * value for value in camera["position_basis_m"]))
            aim_length = math.hypot(camera["aim_yaw_degrees"], camera["aim_pitch_degrees"])
            self.assertLessEqual(position_length, 0.02 + 1e-12)
            self.assertLessEqual(aim_length, 1.5 + 1e-12)
            self.assertTrue(0.98 <= camera["focal_length_scale"] <= 1.02)
            lighting = sample["lighting_jitter"]
            self.assertTrue(-8.0 <= lighting["sun_azimuth_degrees"] <= 8.0)
            self.assertTrue(-5.0 <= lighting["sun_elevation_degrees"] <= 5.0)
            self.assertTrue(0.92 <= lighting["sun_energy_scale"] <= 1.08)
            self.assertTrue(0.90 <= lighting["world_strength_scale"] <= 1.10)
            self.assertTrue(-0.15 <= lighting["exposure_ev"] <= 0.15)

    def test_preview_is_stratified_but_keeps_real_samples(self) -> None:
        samples = select_stratified_preview(self.config)
        self.assertEqual(len(samples), self.config["preview"]["sample_count"])
        categories = self.config["preview"]["stratify_categories"]
        tokens = set().union(*(preview_tokens(sample, categories) for sample in samples))
        contract = validate_domain_randomization_config(self.config)
        expected = set()
        mapping = {
            "lighting": contract["lighting_ids"],
            "surface_wear": contract["wear_ids"],
            "camera_pose": contract["pose_ids"],
            "target_wheel": contract["wheel_ids"],
        }
        for category in categories:
            expected.update((category, value) for value in mapping[category])
        self.assertEqual(tokens, expected)
        field_by_category = {
            "lighting": "lighting_preset",
            "surface_wear": "surface_wear",
            "camera_pose": "camera_pose",
            "target_wheel": "target_wheel",
        }
        for category, quotas in self.config["preview"]["target_counts"].items():
            actual = {key: sum(sample[field_by_category[category]] == key for sample in samples) for key in quotas}
            self.assertEqual(actual, quotas)
        for sample in samples:
            self.assertEqual(sample, sample_domain_randomization(self.config, sample["sample_index"]))

    def test_camera_gate_retry_preserves_non_camera_sample(self) -> None:
        first = sample_domain_randomization(self.config, 17, attempt=0)
        retry = sample_domain_randomization(self.config, 17, attempt=1)
        stable_fields = (
            "lighting_preset",
            "surface_wear",
            "camera_pose",
            "target_wheel",
            "healthy_roll_degrees",
            "wear_seed",
            "lighting_jitter",
        )
        for field in stable_fields:
            self.assertEqual(first[field], retry[field])
        self.assertNotEqual(first["camera_jitter"], retry["camera_jitter"])
        self.assertNotEqual(first["pair_lock_id"], retry["pair_lock_id"])


if __name__ == "__main__":
    unittest.main()
