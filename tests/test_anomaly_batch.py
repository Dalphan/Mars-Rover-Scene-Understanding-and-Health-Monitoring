from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.wheel_preparation.anomaly_batch import (
    POSE_ORDER,
    WHEEL_ORDER,
    anomaly_run_fingerprint,
    assert_pair_retry_preserves_semantics,
    build_anomaly_plan,
    chunk_pairs,
    plan_summary,
    validate_anomaly_batch_config,
)
from src.wheel_preparation.clean_batch import canonical_jsonl_bytes
from src.wheel_preparation.domain_randomization import sample_domain_randomization
from src.wheel_preparation.hole_anomaly import (
    sample_hole_descriptor,
    validate_hole_anomaly_config,
    validate_hole_descriptor,
)


class AnomalyBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.batch = json.loads((cls.root / "configs" / "blender" / "anomaly_batch.json").read_text(encoding="utf-8"))
        cls.domain = json.loads((cls.root / "configs" / "blender" / "domain_randomization.json").read_text(encoding="utf-8"))
        cls.anomaly = json.loads((cls.root / "configs" / "blender" / "wheel_hole_anomaly.json").read_text(encoding="utf-8"))

    def test_configs_and_plan_are_deterministic(self) -> None:
        contract = validate_anomaly_batch_config(self.batch)
        validate_hole_anomaly_config(self.anomaly)
        first = build_anomaly_plan(self.batch, self.domain, self.anomaly)
        second = build_anomaly_plan(self.batch, self.domain, self.anomaly)
        self.assertEqual(first, second)
        self.assertEqual(canonical_jsonl_bytes(first), canonical_jsonl_bytes(second))
        self.assertEqual(contract["pair_count"], 24)
        self.assertEqual(self.anomaly["geometry"]["open_wall_depth_profile"], "T3")
        self.assertEqual(
            self.anomaly["geometry"]["open_wall_depth_max_by_severity_m"],
            {"small": 0.002, "medium": 0.003, "large": 0.004},
        )
        self.assertEqual([row["sample_index"] for row in first], self.batch["sample_indices"])
        self.assertEqual(len({row["pair_lock_id"] for row in first}), 24)

    def test_pilot_is_exact_matrix_and_has_fixed_quotas(self) -> None:
        rows = build_anomaly_plan(self.batch, self.domain, self.anomaly)
        cells = {(row["domain_sample"]["target_wheel"], row["domain_sample"]["camera_pose"]) for row in rows}
        self.assertEqual(cells, {(wheel, pose) for wheel in WHEEL_ORDER for pose in POSE_ORDER})
        summary = plan_summary(rows)
        self.assertEqual(set(summary["wheels"].values()), {4})
        self.assertEqual(set(summary["poses"].values()), {6})
        self.assertEqual(summary["lighting"], {"mars_dusty_refined": 18, "mars_clear_refined": 6})
        self.assertEqual(summary["wear"], {"surface_current": 8, "wear_light": 11, "wear_evident": 5})
        self.assertEqual(set(summary["healthy_roll"].values()), {3})
        self.assertEqual(summary["severity"], {"small": 8, "medium": 8, "large": 8})
        self.assertEqual(summary["surface"], {"tread": 19, "shoulder": 5})
        self.assertEqual(summary["image_sector"], {"leading": 8, "upper": 8, "trailing": 8})
        self.assertEqual(summary["profile_family"], {"jagged_slit": 11, "branched_tear": 10, "peeled_window": 3})
        self.assertEqual(summary["flap_enabled"], 3)
        self.assertTrue(all(row["anomaly"]["surface"] == "tread" for row in rows if row["anomaly"]["severity"] == "large"))

    def test_hole_profiles_and_candidates_are_valid(self) -> None:
        rows = build_anomaly_plan(self.batch, self.domain, self.anomaly)
        for row in rows:
            descriptor = row["anomaly"]
            self.assertEqual(validate_hole_descriptor(self.anomaly, descriptor), [])
            self.assertEqual(len(descriptor["placement_candidates"]), 32)
            self.assertEqual(descriptor["material"]["emission_strength"], 0.0)
            bounds = self.anomaly["severity"][descriptor["severity"]]
            self.assertGreaterEqual(descriptor["quality"]["width"], bounds["long_axis_m"][0] - 1e-7)
            self.assertLessEqual(descriptor["quality"]["width"], bounds["long_axis_m"][1] + 1e-7)
            self.assertGreaterEqual(descriptor["quality"]["height"], bounds["short_axis_m"][0] - 1e-7)
            self.assertLessEqual(descriptor["quality"]["height"], bounds["short_axis_m"][1] + 1e-7)
            if descriptor["severity"] == "small":
                self.assertFalse(descriptor["flap"]["enabled"])

    def test_production_descriptor_is_deterministic(self) -> None:
        first = sample_hole_descriptor(
            self.anomaly,
            master_seed=self.domain["master_seed"],
            sample_index=999,
            surface_wear="wear_light",
        )
        second = sample_hole_descriptor(
            self.anomaly,
            master_seed=self.domain["master_seed"],
            sample_index=999,
            surface_wear="wear_light",
        )
        self.assertEqual(first, second)
        self.assertEqual(validate_hole_descriptor(self.anomaly, first), [])

    def test_production_semantic_assignments_are_explicit_and_fail_closed(self) -> None:
        config = copy.deepcopy(self.batch)
        config["sample_indices"] = config["sample_indices"][:1]
        config["pilot_stratification"] = {"enabled": False}
        config["semantic_assignments"] = [{
            "sample_index": config["sample_indices"][0],
            "severity": "large",
            "surface": "tread",
            "image_sector": "leading",
            "profile_family": "peeled_window",
        }]
        validate_anomaly_batch_config(config)
        descriptor = build_anomaly_plan(config, self.domain, self.anomaly)[0]["anomaly"]
        self.assertEqual(
            {key: descriptor[key] for key in ("severity", "surface", "image_sector", "profile_family")},
            {"severity": "large", "surface": "tread", "image_sector": "leading", "profile_family": "peeled_window"},
        )

        missing = copy.deepcopy(config)
        missing["semantic_assignments"] = []
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            validate_anomaly_batch_config(missing)
        invalid = copy.deepcopy(config)
        invalid["semantic_assignments"][0]["surface"] = "shoulder"
        with self.assertRaisesRegex(ValueError, "Large semantic assignments"):
            validate_anomaly_batch_config(invalid)

    def test_chunking_does_not_change_pair_order(self) -> None:
        rows = build_anomaly_plan(self.batch, self.domain, self.anomaly)
        self.assertEqual([row for chunk in chunk_pairs(rows, set(), 5) for row in chunk], rows)
        completed = {rows[0]["pair_id"], rows[1]["pair_id"]}
        self.assertEqual([row for chunk in chunk_pairs(rows, completed, 7) for row in chunk], rows[2:])

    def test_camera_retry_preserves_pair_semantics(self) -> None:
        row = build_anomaly_plan(self.batch, self.domain, self.anomaly)[0]
        retry = sample_domain_randomization(self.domain, row["sample_index"], attempt=7)
        assert_pair_retry_preserves_semantics(row["domain_sample"], retry)
        changed = copy.deepcopy(retry)
        changed["surface_wear"] = "wear_evident" if retry["surface_wear"] != "wear_evident" else "surface_current"
        with self.assertRaises(ValueError):
            assert_pair_retry_preserves_semantics(row["domain_sample"], changed)

    def test_config_rejects_ambiguous_selection_and_large_shoulder(self) -> None:
        ambiguous = copy.deepcopy(self.batch)
        ambiguous["range"] = {"start_index": 0, "count": 24}
        with self.assertRaises(ValueError):
            validate_anomaly_batch_config(ambiguous)
        invalid = copy.deepcopy(self.anomaly)
        invalid["distributions"]["surface_by_severity"]["large"] = {"tread": 0.9, "shoulder": 0.1}
        with self.assertRaises(ValueError):
            validate_hole_anomaly_config(invalid)

        invalid_speckle = copy.deepcopy(self.anomaly)
        invalid_speckle["mask_gates"]["maximum_raster_speckle_component_px"] = 5
        with self.assertRaisesRegex(ValueError, "at most 4 pixels"):
            validate_hole_anomaly_config(invalid_speckle)

        capped = copy.deepcopy(self.anomaly)
        capped["geometry"]["recessed_cap_enabled"] = True
        with self.assertRaisesRegex(ValueError, "true through opening"):
            validate_hole_anomaly_config(capped)

        unknown_rim = copy.deepcopy(self.anomaly)
        unknown_rim["geometry"]["active_rim_profile"] = "R9_unknown"
        with self.assertRaisesRegex(ValueError, "Rim reduction profiles"):
            validate_hole_anomaly_config(unknown_rim)

        wide_robust = copy.deepcopy(self.anomaly)
        wide_robust["geometry"]["rim_profiles"]["R4_robust"]["effective_width_m"][1] = 0.00105
        with self.assertRaisesRegex(ValueError, "exactly 0.45 to 0.85 mm"):
            validate_hole_anomaly_config(wide_robust)

        unknown_wall_profile = copy.deepcopy(self.anomaly)
        unknown_wall_profile["geometry"]["open_wall_depth_profile"] = "T4"
        with self.assertRaisesRegex(ValueError, "approved T3"):
            validate_hole_anomaly_config(unknown_wall_profile)

        deep_t3 = copy.deepcopy(self.anomaly)
        deep_t3["geometry"]["open_wall_depth_max_by_severity_m"]["large"] = 0.008
        with self.assertRaisesRegex(ValueError, "exactly 2, 3 and 4 mm"):
            validate_hole_anomaly_config(deep_t3)

    def test_production_execution_pipeline_is_explicit_and_fail_closed(self) -> None:
        self.assertTrue(self.batch["execution"]["geometry_cache"])
        self.assertTrue(self.batch["execution"]["async_postprocess"])
        self.assertEqual(self.batch["execution"]["async_queue_depth"], 4)
        validate_anomaly_batch_config(self.batch)

        invalid_depth = copy.deepcopy(self.batch)
        invalid_depth["execution"]["async_queue_depth"] = 0
        with self.assertRaisesRegex(ValueError, "between 1 and 16"):
            validate_anomaly_batch_config(invalid_depth)

        unknown = copy.deepcopy(self.batch)
        unknown["execution"]["workers"] = 2
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            validate_anomaly_batch_config(unknown)

    def test_async_pair_finalizer_does_not_use_blender_api(self) -> None:
        renderer = (self.root / "scripts" / "blender" / "render_anomaly_batch.py").read_text(encoding="utf-8")
        finalizer = renderer.split("def _finalize_pair_artifacts", 1)[1].split("def render_chunk", 1)[0]
        self.assertNotIn("bpy.", finalizer)
        self.assertNotIn("gpu.", finalizer)

    def test_benchmark_sample_override_is_explicit_and_fail_closed(self) -> None:
        production_32 = copy.deepcopy(self.batch)
        production_32["render"]["samples"] = 32
        with self.assertRaisesRegex(ValueError, "benchmark-only configs may use 32"):
            validate_anomaly_batch_config(production_32)

        benchmark_path = (
            self.root
            / "configs"
            / "blender"
            / "benchmarks"
            / "anomaly_24_cache_32.json"
        )
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_anomaly_batch_config(benchmark)["pair_count"], 24)

        incomplete = copy.deepcopy(benchmark)
        del incomplete["benchmark"]["reuse_preflight"]
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            validate_anomaly_batch_config(incomplete)

        invalid_compression = copy.deepcopy(benchmark)
        invalid_compression["benchmark"]["png_compression_level"] = 10
        with self.assertRaisesRegex(ValueError, "0..9"):
            validate_anomaly_batch_config(invalid_compression)

    def test_fingerprint_includes_anomaly_contract(self) -> None:
        args = dict(
            source_sha256="a" * 64,
            batch_config_sha256="b" * 64,
            domain_config_sha256="c" * 64,
            anomaly_config_sha256="d" * 64,
            plan_sha256="e" * 64,
        )
        first = anomaly_run_fingerprint(**args)
        args["anomaly_config_sha256"] = "f" * 64
        self.assertNotEqual(first, anomaly_run_fingerprint(**args))

    def test_runtime_code_is_part_of_run_fingerprint(self):
        args = dict(
            source_sha256="a" * 64,
            batch_config_sha256="b" * 64,
            domain_config_sha256="c" * 64,
            anomaly_config_sha256="d" * 64,
            plan_sha256="e" * 64,
            runtime_code_sha256s={"renderer": "1" * 64},
        )
        first = anomaly_run_fingerprint(**args)
        args["runtime_code_sha256s"] = {"renderer": "2" * 64}
        self.assertNotEqual(first, anomaly_run_fingerprint(**args))


if __name__ == "__main__":
    unittest.main()
