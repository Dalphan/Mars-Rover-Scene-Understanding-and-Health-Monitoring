import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.wheel_preparation.dataset_splits import (
    build_dataset_splits,
    expand_split_items,
    validate_dataset_composition_config,
    write_split_indexes,
)


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "schema_version": 1, "dataset_id": "tiny", "master_seed": 7,
            "pool": {"manifest": "pool/manifest.jsonl", "standalone_clean_units": 4, "paired_clean_hole_units": 8, "artifact_policy": "reference_relative_paths_without_copy"},
            "composition": {
                "total_images": 20, "clean_images": 12, "anomaly_images": 8,
                "hole": {"severity": {"small": .5, "medium": .4, "large": .1}, "surface": {"tread": .8, "shoulder": .2}, "constraints": {"large_surface": "tread"}},
                "domain_randomization": {
                    "camera_pose": {"A_overhead": .4, "C_leading_three_quarter": .275, "C_trailing_three_quarter": .275, "D_upper_detail": .05},
                    "lighting": {"mars_dusty_refined": .75, "mars_clear_refined": .25},
                    "surface_wear": {"surface_current": .2, "wear_light": .45, "wear_evident": .35},
                },
            },
            "splits": {
                "train": {"fraction": .15, "image_count": 3, "standalone_clean_units": 3, "paired_clean_hole_units": 0},
                "validation": {"fraction": .15, "image_count": 3, "standalone_clean_units": 1, "paired_clean_hole_units": 1},
                "test": {"fraction": .70, "image_count": 14, "standalone_clean_units": 0, "paired_clean_hole_units": 7},
            },
            "evaluation_minimum_quotas": {
                "validation": {"wheel_camera_severity": 1, "wheel_camera_sector": 1, "wheel_camera_surface": 1},
                "test": {"wheel_camera_severity": 1, "wheel_camera_sector": 1, "wheel_camera_surface": 1},
            },
            "evaluation_categories": {
                "target_wheel": ["wheel_front_left"], "camera_pose": ["A_overhead"],
                "severity": ["small"], "image_sector": ["upper"], "surface": ["tread"],
            },
            "split_policy": {"train_clean_only": True, "pair_lock_indivisible": True, "artifact_materialization": "index_only"},
        }
        self.rows = [
            {"unit_id": f"c{i}", "unit_type": "standalone_clean", "pair_lock_id": None, "members": [{"item_id": f"c{i}", "condition": "clean", "artifacts": {"rgb": f"rgb/{i}.png"}}]}
            for i in range(4)
        ]
        for i in range(8):
            severity = "small"
            self.rows.append({
                "unit_id": f"p{i}", "unit_type": "paired_clean_hole", "pair_lock_id": f"lock{i}",
                "sampling": {"target_wheel": "wheel_front_left", "camera_pose": "A_overhead", "severity": severity, "image_sector": "upper", "surface": "tread"},
                "members": [
                    {"item_id": f"p{i}_clean", "condition": "clean", "artifacts": {"rgb": f"rgb/clean/{i}.png"}},
                    {"item_id": f"p{i}_hole", "condition": "hole", "artifacts": {"rgb": f"rgb/anomaly/{i}.png"}},
                ],
            })

    def test_contract_counts_and_requested_weights(self):
        self.assertEqual(validate_dataset_composition_config(self.config)["total_images"], 20)

    def test_split_is_deterministic_and_pairs_are_indivisible(self):
        first = build_dataset_splits(self.config, self.rows)
        second = build_dataset_splits(self.config, reversed(self.rows))
        self.assertEqual(first, second)
        self.assertTrue(all(row["unit_type"] == "standalone_clean" for row in first["train"]))
        self.assertEqual(len({row["unit_id"] for rows in first.values() for row in rows}), len(self.rows))
        self.assertEqual(len(expand_split_items(first["validation"], "validation")), 3)

    def test_large_shoulder_fails_closed(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]["sampling"].update({"severity": "large", "surface": "shoulder"})
        with self.assertRaisesRegex(ValueError, "Large shoulder"):
            build_dataset_splits(self.config, rows)

    def test_indexes_are_metadata_only(self):
        splits = build_dataset_splits(self.config, self.rows)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = write_split_indexes(root, self.config, splits)
            self.assertEqual(report["splits"]["train"]["images"], 3)
            self.assertEqual(len(list(root.rglob("*.png"))), 0)
            self.assertEqual(json.loads((root / "dataset.json").read_text())["dataset_id"], "tiny")


if __name__ == "__main__":
    unittest.main()
