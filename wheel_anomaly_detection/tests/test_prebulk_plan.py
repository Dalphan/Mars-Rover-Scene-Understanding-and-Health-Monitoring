from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from src.wheel_preparation.anomaly_batch import build_anomaly_plan
from src.wheel_preparation.clean_batch import build_plan


class PrebulkPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.clean_config = json.loads((root / "configs/blender/prebulk_200_clean.json").read_text(encoding="utf-8"))
        cls.pair_config = json.loads((root / "configs/blender/prebulk_200_anomaly.json").read_text(encoding="utf-8"))
        cls.domain = json.loads((root / cls.clean_config["domain_randomization_config"]).read_text(encoding="utf-8"))
        cls.anomaly = json.loads((root / cls.pair_config["anomaly_config"]).read_text(encoding="utf-8"))

    def test_prebulk_has_200_images_and_disjoint_global_indices(self) -> None:
        clean = build_plan(self.clean_config, self.domain)
        pairs = build_anomaly_plan(self.pair_config, self.domain, self.anomaly)
        self.assertEqual(len(clean), 150)
        self.assertEqual(len(pairs), 25)
        self.assertEqual(len(clean) + 2 * len(pairs), 200)
        self.assertFalse(set(self.clean_config["sample_indices"]) & set(self.pair_config["sample_indices"]))

    def test_prebulk_exact_marginal_quotas(self) -> None:
        clean = build_plan(self.clean_config, self.domain)
        pairs = build_anomaly_plan(self.pair_config, self.domain, self.anomaly)
        self.assertEqual(Counter(row["camera_pose"] for row in clean), Counter({"A_overhead": 60, "C_leading_three_quarter": 41, "C_trailing_three_quarter": 41, "D_upper_detail": 8}))
        self.assertEqual(Counter(row["lighting_preset"] for row in clean), Counter({"mars_dusty_refined": 113, "mars_clear_refined": 37}))
        self.assertEqual(Counter(row["surface_wear"] for row in clean), Counter({"surface_current": 30, "wear_light": 68, "wear_evident": 52}))
        self.assertEqual(Counter(Counter(row["target_wheel"] for row in clean).values()), Counter({25: 6}))

        descriptors = [row["anomaly"] for row in pairs]
        domains = [row["domain_sample"] for row in pairs]
        self.assertEqual(Counter(row["camera_pose"] for row in domains), Counter({"A_overhead": 10, "C_leading_three_quarter": 7, "C_trailing_three_quarter": 7, "D_upper_detail": 1}))
        self.assertEqual(Counter(row["lighting_preset"] for row in domains), Counter({"mars_dusty_refined": 19, "mars_clear_refined": 6}))
        self.assertEqual(Counter(row["surface_wear"] for row in domains), Counter({"surface_current": 5, "wear_light": 11, "wear_evident": 9}))
        self.assertEqual(Counter(row["severity"] for row in descriptors), Counter({"small": 13, "medium": 10, "large": 2}))
        self.assertEqual(Counter(row["surface"] for row in descriptors), Counter({"tread": 20, "shoulder": 5}))
        self.assertTrue(all(row["surface"] == "tread" for row in descriptors if row["severity"] == "large"))


if __name__ == "__main__":
    unittest.main()
