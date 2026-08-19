from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.host.validate_anomaly_batch import validate_manifest_matches_plan, validate_pair_artifacts, validate_photometric_record


class AnomalyBatchValidationTests(unittest.TestCase):
    def _post_gates(self, *, changed: bool, median: bool, nonzero: bool = True) -> dict:
        return {
            "ok": nonzero,
            "photometric_status": "normal_contrast" if changed and median else ("low_contrast" if nonzero else "no_effect"),
            "gates": {
                "binary": True, "single_component": True, "minimum_area": True,
                "minimum_short_side": True, "inside_target_roi": True,
                "local_difference_energy": True, "maximum_effect_area": True,
                "nonzero_effect": nonzero, "changed_fraction": changed, "median_delta": median,
            },
        }

    def test_low_contrast_is_valid_diagnostic_record(self) -> None:
        validate_photometric_record(self._post_gates(changed=False, median=False), "dr_low")

    def test_no_effect_remains_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate false"):
            validate_photometric_record(self._post_gates(changed=False, median=False, nonzero=False), "dr_zero")

    def _artifact(self, path: Path) -> dict:
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_pair_artifacts_accept_binary_single_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = root / "clean.png"
            anomaly = root / "anomaly.png"
            target = root / "target.png"
            mask = root / "mask.png"
            Image.new("RGB", (16, 12), (30, 20, 10)).save(clean)
            Image.new("RGB", (16, 12), (31, 20, 10)).save(anomaly)
            target_image = Image.new("L", (16, 12), 0)
            ImageDraw.Draw(target_image).rectangle((2, 2, 13, 10), fill=255)
            target_image.save(target)
            mask_image = Image.new("L", (16, 12), 0)
            ImageDraw.Draw(mask_image).rectangle((5, 4, 8, 7), fill=255)
            mask_image.save(mask)
            row = {
                "pair_id": "dr_000001",
                "artifacts": {
                    "clean_rgb": self._artifact(clean),
                    "anomaly_rgb": self._artifact(anomaly),
                    "target_wheel_mask": self._artifact(target),
                    "anomaly_mask": self._artifact(mask),
                },
            }
            validate_pair_artifacts(root, row, (16, 12))

    def test_pair_artifacts_reject_disconnected_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("clean.png", "anomaly.png"):
                Image.new("RGB", (8, 8), (0, 0, 0)).save(root / name)
            target = Image.new("L", (8, 8), 255)
            target.putpixel((0, 0), 0)
            target.save(root / "target.png")
            mask = Image.new("L", (8, 8), 0)
            mask.putpixel((1, 1), 255)
            mask.putpixel((6, 6), 255)
            mask.save(root / "mask.png")
            row = {
                "pair_id": "dr_000001",
                "artifacts": {
                    key: self._artifact(root / filename)
                    for key, filename in {
                        "clean_rgb": "clean.png", "anomaly_rgb": "anomaly.png",
                        "target_wheel_mask": "target.png", "anomaly_mask": "mask.png",
                    }.items()
                },
            }
            with self.assertRaisesRegex(ValueError, "one 8-connected component"):
                validate_pair_artifacts(root, row, (8, 8))

    def test_manifest_pair_lock_is_fail_closed(self) -> None:
        planned = {
            "pair_id": "dr_000001", "pair_index": 0, "sample_index": 1,
            "members": {"clean": "a", "anomaly": "b"}, "pair_lock_id": "lock",
            "domain_sample": {
                "target_wheel": "wheel_front_left", "camera_pose": "A_overhead",
                "healthy_roll_degrees": 0.0, "lighting_preset": "mars_dusty_refined",
                "surface_wear": "surface_current", "lighting_jitter": {},
            },
            "anomaly": {"class_id": "hole", "severity": "small"},
        }
        row = {
            **{key: planned[key] for key in ("pair_id", "pair_index", "sample_index", "members", "pair_lock_id")},
            "condition": "paired_clean_hole", "sampling": dict(planned["domain_sample"]),
            "anomaly": planned["anomaly"],
            "gates": {
                "framing": True,
                "terrain_coverage": True,
                "terrain_contact": True,
                "anomaly_visibility": True,
                "through_opening": True,
                "terrain_contact_details": {"ok": True},
                "through_opening_details": {"ok": True},
                "opening_mask_details": {"ok": True},
                "mask_and_photometric": {
                    "ok": True,
                    "photometric_status": "normal_contrast",
                    "gates": {
                        "binary": True, "single_component": True, "minimum_area": True,
                        "minimum_short_side": True, "inside_target_roi": True,
                        "local_difference_energy": True, "maximum_effect_area": True,
                        "nonzero_effect": True, "changed_fraction": True, "median_delta": True,
                    },
                },
            },
            "resolved": {
                **{key: [1.0] * 16 for key in ("camera_matrix_world", "wheel_matrix_world", "rover_matrix_world")},
                "hole_material": {
                    "open_wall_depth_profile": "T3",
                    "open_wall_fold_arc_count": 2,
                    "effective_open_wall_depth_min_m": 0.00075,
                    "effective_open_wall_depth_max_m": 0.002,
                    "effective_open_wall_depth_m": 0.002,
                },
            },
        }
        validate_manifest_matches_plan(row, planned)
        row["sampling"]["lighting_preset"] = "mars_clear_refined"
        with self.assertRaisesRegex(ValueError, "Pair-locked field changed"):
            validate_manifest_matches_plan(row, planned)
        row["sampling"]["lighting_preset"] = "mars_dusty_refined"
        row["resolved"]["hole_material"]["effective_open_wall_depth_max_m"] = 0.008
        with self.assertRaisesRegex(ValueError, "T3 wall-depth contract"):
            validate_manifest_matches_plan(row, planned)


if __name__ == "__main__":
    unittest.main()
