from __future__ import annotations

import hashlib
import json
import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.host.validate_clean_batch import (
    validate_artifact_inventory,
    validate_artifact_row,
    validate_manifest_matches_plan,
)
from src.wheel_preparation.clean_batch import (
    build_plan,
    canonical_jsonl_bytes,
    chunk_rows,
    run_fingerprint,
    sha256_bytes,
    validate_clean_batch_config,
    validate_run_id,
)
from src.wheel_preparation.domain_randomization import sample_domain_randomization


class CleanBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.batch = json.loads((cls.root / "configs" / "blender" / "clean_batch.json").read_text(encoding="utf-8"))
        cls.domain = json.loads((cls.root / "configs" / "blender" / "domain_randomization.json").read_text(encoding="utf-8"))

    def test_config_and_smoke_plan_are_deterministic(self) -> None:
        contract = validate_clean_batch_config(self.batch)
        first = build_plan(self.batch, self.domain)
        second = build_plan(self.batch, self.domain)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual([row["sample_index"] for row in first], list(range(78, 102)))
        self.assertEqual(canonical_jsonl_bytes(first), canonical_jsonl_bytes(second))
        self.assertEqual(contract["resolution"], [1200, 900])
        self.assertEqual({row["condition"] for row in first}, {"clean"})

    def test_smoke_plan_covers_all_approved_dimensions(self) -> None:
        rows = build_plan(self.batch, self.domain)
        self.assertEqual({row["target_wheel"] for row in rows}, set(self.domain["distributions"]["target_wheel"]))
        self.assertEqual({row["camera_pose"] for row in rows}, set(self.domain["distributions"]["camera_pose"]))
        self.assertEqual({row["lighting_preset"] for row in rows}, set(self.domain["distributions"]["lighting"]))
        self.assertEqual({row["surface_wear"] for row in rows}, set(self.domain["distributions"]["surface_wear"]))
        self.assertEqual({row["healthy_roll_degrees"] for row in rows}, set(map(float, self.domain["distributions"]["healthy_roll_degrees"])))

    def test_chunking_does_not_change_plan(self) -> None:
        rows = build_plan(self.batch, self.domain)
        flattened = [row for chunk in chunk_rows(rows, set(), 5) for row in chunk]
        self.assertEqual(flattened, rows)
        completed = {rows[0]["sample_id"], rows[1]["sample_id"]}
        flattened = [row for chunk in chunk_rows(rows, completed, 7) for row in chunk]
        self.assertEqual(flattened, rows[2:])

    def test_fingerprint_and_run_id_are_fail_closed(self) -> None:
        first = run_fingerprint(
            source_sha256="a" * 64,
            batch_config_sha256="b" * 64,
            domain_config_sha256="c" * 64,
            plan_sha256="d" * 64,
        )
        second = run_fingerprint(
            source_sha256="a" * 64,
            batch_config_sha256="b" * 64,
            domain_config_sha256="c" * 64,
            plan_sha256="e" * 64,
        )
        self.assertNotEqual(first, second)
        third = run_fingerprint(
            source_sha256="a" * 64,
            batch_config_sha256="b" * 64,
            domain_config_sha256="c" * 64,
            plan_sha256="d" * 64,
            referenced_config_sha256s={"lighting": "f" * 64},
        )
        self.assertNotEqual(first, third)
        self.assertEqual(validate_run_id("smoke_24-v1"), "smoke_24-v1")
        with self.assertRaises(ValueError):
            validate_run_id("../escape")

    def test_camera_retry_keeps_clean_semantics(self) -> None:
        planned = sample_domain_randomization(self.domain, 78, attempt=0)
        retry = sample_domain_randomization(self.domain, 78, attempt=4)
        for field in (
            "lighting_preset",
            "surface_wear",
            "camera_pose",
            "target_wheel",
            "healthy_roll_degrees",
            "wear_seed",
            "lighting_jitter",
        ):
            self.assertEqual(planned[field], retry[field])
        self.assertNotEqual(planned["camera_jitter"], retry["camera_jitter"])

    def test_artifact_validator_accepts_rgb_and_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "rgb").mkdir()
            (run_dir / "target_wheel_mask").mkdir()
            rgb = run_dir / "rgb" / "dr_000078.png"
            mask = run_dir / "target_wheel_mask" / "dr_000078.png"
            Image.new("RGB", (8, 6), (80, 40, 20)).save(rgb)
            mask_image = Image.new("L", (8, 6), 0)
            mask_image.putpixel((3, 2), 255)
            mask_image.save(mask)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            row = {
                "sample_id": "dr_000078",
                "condition": "clean",
                "gates": {
                    "framing": True,
                    "terrain_coverage": True,
                    "terrain_contact": True,
                    "terrain_contact_details": {"ok": True},
                },
                "resolved": {
                    "camera_matrix_world": [1.0] * 16,
                    "wheel_matrix_world": [1.0] * 16,
                    "rover_matrix_world": [1.0] * 16,
                },
                "artifacts": {
                    "rgb": {"path": "rgb/dr_000078.png", "sha256": digest(rgb)},
                    "target_wheel_mask": {"path": "target_wheel_mask/dr_000078.png", "sha256": digest(mask)},
                },
                "render": {"setup_seconds": 0.1, "render_seconds": 1.0, "write_seconds": 0.1, "total_seconds": 1.2},
            }
            validate_artifact_row(run_dir, row, (8, 6))
            failed_gate = copy.deepcopy(row)
            failed_gate["gates"]["framing"] = False
            with self.assertRaises(ValueError):
                validate_artifact_row(run_dir, failed_gate, (8, 6))
            rgb.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                validate_artifact_row(run_dir, row, (8, 6))
            Image.new("RGB", (8, 6), (80, 40, 20)).save(rgb)
            row["artifacts"]["rgb"]["sha256"] = digest(rgb)
            mask_image.putpixel((4, 2), 128)
            mask_image.save(mask)
            row["artifacts"]["target_wheel_mask"]["sha256"] = digest(mask)
            with self.assertRaises(ValueError):
                validate_artifact_row(run_dir, row, (8, 6))
            Image.new("L", (8, 6), 0).save(mask)
            row["artifacts"]["target_wheel_mask"]["sha256"] = digest(mask)
            with self.assertRaises(ValueError):
                validate_artifact_row(run_dir, row, (8, 6))
            mask.unlink()
            with self.assertRaises(ValueError):
                validate_artifact_row(run_dir, row, (8, 6))

    def test_artifact_inventory_rejects_extra_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "rgb").mkdir()
            (run_dir / "target_wheel_mask").mkdir()
            Image.new("RGB", (2, 2)).save(run_dir / "rgb" / "dr_000078.png")
            Image.new("L", (2, 2)).save(run_dir / "target_wheel_mask" / "dr_000078.png")
            validate_artifact_inventory(run_dir, ["dr_000078"])
            Image.new("RGB", (2, 2)).save(run_dir / "rgb" / "orphan.png")
            with self.assertRaises(ValueError):
                validate_artifact_inventory(run_dir, ["dr_000078"])

    def test_manifest_semantics_must_match_plan(self) -> None:
        planned = build_plan(self.batch, self.domain)[0]
        row = {
            "sample_id": planned["sample_id"],
            "semantic_sample_id": planned["sample_id"],
            "sample_index": planned["sample_index"],
            "seeds": {
                "sample": planned["sample_seed"],
                "wear": planned["wear_seed"],
                "camera_attempt": planned["camera_attempt_seed"],
                "pair_lock_id": planned["pair_lock_id"],
            },
            "sampling": {
                "target_wheel": planned["target_wheel"],
                "camera_pose": planned["camera_pose"],
                "lighting_preset": planned["lighting_preset"],
                "surface_wear": planned["surface_wear"],
                "lighting_jitter": planned["lighting_jitter"],
                "healthy_roll_degrees": planned["healthy_roll_degrees"],
            },
        }
        validate_manifest_matches_plan(row, planned)
        row["sampling"]["target_wheel"] = "wheel_front_left" if planned["target_wheel"] != "wheel_front_left" else "wheel_rear_right"
        with self.assertRaises(ValueError):
            validate_manifest_matches_plan(row, planned)

    def test_config_rejects_wrong_schema_version_and_range(self) -> None:
        for mutation in (
            {"schema_version": 2},
            {"range": {"start_index": -1, "count": 24}},
            {"render": {**self.batch["render"], "blender_major_minor": [5, 1]}},
        ):
            candidate = json.loads(json.dumps(self.batch))
            candidate.update(mutation)
            with self.assertRaises(ValueError):
                validate_clean_batch_config(candidate)

    def test_clean_benchmark_contract_is_explicit_and_fail_closed(self) -> None:
        benchmark_path = (
            self.root
            / "configs"
            / "blender"
            / "benchmarks"
            / "clean_24_cache_64_pipeline.json"
        )
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_clean_batch_config(benchmark)["count"], 24)
        incomplete = copy.deepcopy(benchmark)
        del incomplete["benchmark"]["reuse_matrix_audit"]
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            validate_clean_batch_config(incomplete)
        invalid_compression = copy.deepcopy(benchmark)
        invalid_compression["benchmark"]["scene_png_compression"] = 101
        with self.assertRaisesRegex(ValueError, "0..100"):
            validate_clean_batch_config(invalid_compression)
        invalid_queue = copy.deepcopy(self.batch)
        invalid_queue["execution"]["async_postprocess"] = True
        invalid_queue["execution"]["async_queue_depth"] = 0
        with self.assertRaisesRegex(ValueError, "between 1 and 16"):
            validate_clean_batch_config(invalid_queue)

    def test_async_png_finalizer_does_not_load_blender_images(self) -> None:
        renderer = (self.root / "scripts" / "blender" / "render_clean_batch.py").read_text(encoding="utf-8")
        inspector = renderer.split("def _inspect_output", 1)[1].split("def _paeth", 1)[0]
        finalizer = renderer.split("def _finalize_sample_artifacts", 1)[1].split("def render_chunk", 1)[0]
        self.assertNotIn("bpy.", inspector)
        self.assertNotIn("bpy.", finalizer)
        self.assertNotIn("bpy.data.images", inspector)

    def test_sha256_bytes_is_stable(self) -> None:
        self.assertEqual(sha256_bytes(b"clean-batch"), sha256_bytes(b"clean-batch"))


if __name__ == "__main__":
    unittest.main()
