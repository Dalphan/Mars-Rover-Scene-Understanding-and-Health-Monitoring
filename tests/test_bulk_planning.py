from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from src.wheel_preparation.anomaly_batch import validate_anomaly_batch_config
from src.wheel_preparation.clean_batch import validate_clean_batch_config
from src.wheel_preparation.domain_randomization import sample_domain_randomization


class BulkPlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        load = lambda name: json.loads((cls.root / "configs" / "blender" / name).read_text(encoding="utf-8"))
        cls.clean = load("bulk_remaining_clean.json")
        cls.pairs = load("bulk_remaining_anomaly.json")
        cls.prebulk_clean = load("prebulk_200_clean.json")
        cls.prebulk_pairs = load("prebulk_200_anomaly.json")
        cls.domain = load("domain_randomization_dataset_v1.json")

    def test_remaining_counts_and_indices_are_disjoint(self) -> None:
        self.assertEqual(validate_clean_batch_config(self.clean)["count"], 7350)
        self.assertEqual(validate_anomaly_batch_config(self.pairs)["pair_count"], 1225)
        groups = [
            set(self.clean["sample_indices"]),
            set(self.pairs["sample_indices"]),
            set(self.prebulk_clean["sample_indices"]),
            set(self.prebulk_pairs["sample_indices"]),
        ]
        for index, left in enumerate(groups):
            for right in groups[index + 1 :]:
                self.assertFalse(left & right)

    def test_paired_semantic_marginals_include_prebulk_exactly(self) -> None:
        assignments = self.pairs["semantic_assignments"]
        self.assertEqual(Counter(row["severity"] for row in assignments) + Counter({"small": 13, "medium": 10, "large": 2}), Counter({"small": 625, "medium": 500, "large": 125}))
        self.assertEqual(Counter(row["surface"] for row in assignments) + Counter({"tread": 20, "shoulder": 5}), Counter({"tread": 1000, "shoulder": 250}))
        self.assertEqual(Counter(row["image_sector"] for row in assignments) + Counter({"leading": 9, "upper": 8, "trailing": 8}), Counter({"leading": 417, "upper": 417, "trailing": 416}))
        self.assertTrue(all(row["surface"] == "tread" for row in assignments if row["severity"] == "large"))

    def test_materialized_domain_indices_hit_exact_lighting_wear_and_roll_totals(self) -> None:
        clean_indices = self.prebulk_clean["sample_indices"] + self.clean["sample_indices"]
        pair_indices = self.prebulk_pairs["sample_indices"] + self.pairs["sample_indices"]
        clean = [sample_domain_randomization(self.domain, index) for index in clean_indices]
        pairs = [sample_domain_randomization(self.domain, index) for index in pair_indices]
        for field, expected in {
            "lighting_preset": {"mars_dusty_refined": 7500, "mars_clear_refined": 2500},
            "surface_wear": {"surface_current": 2000, "wear_light": 4500, "wear_evident": 3500},
            "healthy_roll_degrees": {value: 1250 for value in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)},
        }.items():
            counts = Counter(row[field] for row in clean)
            counts.update({key: 2 * value for key, value in Counter(row[field] for row in pairs).items()})
            self.assertEqual(counts, Counter(expected))


if __name__ == "__main__":
    unittest.main()
