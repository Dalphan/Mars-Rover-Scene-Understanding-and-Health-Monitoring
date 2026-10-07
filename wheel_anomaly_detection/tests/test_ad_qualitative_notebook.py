"""Data, spatial-alignment and resume checks for the standalone Figure 6.1 notebook."""

import ast
import csv
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from PIL import Image


class QualitativeNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.temporary_root = cls.root / "outputs/qualitative_notebook_tests"
        cls.temporary_root.mkdir(parents=True, exist_ok=True)
        cls.notebook = json.loads((cls.root / "notebooks/kaggle_wheel_anomaly_qualitative_comparison.ipynb").read_text("utf-8"))
        cls.cells = {cell["id"]: "".join(cell["source"]) for cell in cls.notebook["cells"]}
        module = types.ModuleType("_wheel_qualitative_test")
        sys.modules[module.__name__] = module
        cls.ns = module.__dict__
        identities = [
            "qualitative-imports", "qualitative-configuration",
            "synthetic-preprocessing-definitions", "synthetic-dataset-definitions",
            "qualitative-selection-definitions", "common-model-interface",
            "patchcore-model-definitions", "efficientad-model-definitions",
            "supersimplenet-model-definitions", "tinyglass-model-definitions",
            "google-drive-checkpoint-download", "strict-efficientad-restore-definitions",
            "strict-supersimplenet-restore-definitions", "strict-tinyglass-restore-definitions",
            "qualitative-checkpoint-definitions", "qualitative-inference-definitions",
            "qualitative-render-definitions",
        ]
        for identity in identities:
            source = cls.cells[identity]
            if identity == "qualitative-imports":
                # Widgets/display are provided by Kaggle; these checks use headless Matplotlib.
                tree = ast.parse(source)
                tree.body = [n for n in tree.body if not (
                    isinstance(n, ast.ImportFrom) and n.module == "IPython.display")]
                source = tree
            exec(compile(source, identity, "exec"), cls.ns)
        torch.set_num_threads(2)

    def test_notebook_code_and_authoritative_copies(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(len(self.cells), len(self.notebook["cells"]))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        pairs = {
            "synthetic-preprocessing-definitions": "data/preprocessing.py",
            "synthetic-dataset-definitions": "data/curiosity_wheel_dataset.py",
            "common-model-interface": "models/base.py",
            "patchcore-model-definitions": "models/patchcore.py",
            "efficientad-model-definitions": "models/trainable.py",
            "supersimplenet-model-definitions": "models/supersimplenet.py",
            "tinyglass-model-definitions": "models/tinyglass.py",
        }
        for identity, relative in pairs.items():
            path = self.root / "src/anomaly_detection" / relative
            source = path.read_text("utf-8")
            original = ast.parse(source)
            embedded = ast.parse(self.cells[identity])
            expected = [node for node in original.body if not (isinstance(node, ast.ImportFrom) and node.level)]
            self.assertEqual(ast.dump(ast.Module(body=expected, type_ignores=[])), ast.dump(embedded), relative)
            self.assertEqual(self.notebook["metadata"]["qualitative_comparison"]["source_sha256"][
                "src/anomaly_detection/" + relative], hashlib.sha256(source.encode()).hexdigest())

    def test_reference_profiles_and_restore_helpers_match_selected_notebooks(self):
        for filename, key, family in (
            ("kaggle_efficientad_efficiency.ipynb", "EAD-S", "efficientad"),
            ("kaggle_supersimplenet_efficiency.ipynb", "SSN", "supersimplenet"),
            ("kaggle_tinyglass_efficiency.ipynb", "TG-384", "tinyglass"),
        ):
            original = json.loads((self.root / "notebooks" / filename).read_text("utf-8"))
            cells = {c.get("id"): "".join(c["source"]) for c in original["cells"]}
            config = next(value for identity, value in cells.items() if identity.endswith("configuration"))
            env = {"Path": Path}
            exec(config, env)
            expected = {k: v for k, v in env["SELECTED_DRIVE_RUN"].items()
                        if k in ("model", "folder_id", "checkpoint_file_id")}
            self.assertEqual(self.ns["MODEL_PROFILES"][key]["drive_run"], expected)
            if family == "tinyglass":
                for profile_key, profile in env["REFERENCE_PROFILES"].items():
                    current = self.ns["MODEL_PROFILES"][profile_key]
                    self.assertEqual(current["input_size"], profile["input_size"])
                    self.assertEqual(current["feature_grid_resolution"], profile["feature_grid_resolution"])
                    self.assertEqual(current["gaussian_sigma"], profile["gaussian_sigma"])
                    self.assertEqual(current["drive_run"]["checkpoint_file_id"], profile["drive_run"]["checkpoint_file_id"])
            identity = f"strict-{family}-restore-definitions"
            function = f"load_frozen_{family}"
            original_node = next(n for n in ast.parse(cells[identity]).body if getattr(n, "name", None) == function)
            embedded_node = next(n for n in ast.parse(self.cells[identity]).body if getattr(n, "name", None) == function)
            self.assertEqual(ast.dump(original_node), ast.dump(embedded_node))

    def test_pc_large_field_of_view_and_heatmap_alignment(self):
        profile = self.ns["MODEL_PROFILES"]["PC-L"]
        self.assertEqual(self.ns["common_coverage_box"](profile), (45, 45, 630, 630))
        values, observed = self.ns["align_to_common"](np.ones((224, 224)), profile)
        self.assertEqual(int(observed.sum()), 630 * 630)
        self.assertTrue(np.isnan(values[:45]).all())
        self.assertTrue(np.isnan(values[:, 675:]).all())
        np.testing.assert_array_equal(values[observed], 1)
        rgb = self.ns["colour_map"](values, observed, {"lower": 0, "upper": 1})
        np.testing.assert_array_equal(rgb[0, 0], [190, 190, 190])
        self.assertFalse(np.array_equal(rgb[100, 100], rgb[0, 0]))
        for key in ("PC-S", "EAD-S", "SSN", "TG-384", "TG-512"):
            _, valid = self.ns["align_to_common"](np.ones((8, 8)), self.ns["MODEL_PROFILES"][key])
            self.assertTrue(valid.all())

    def test_code_hash_detects_changes_inside_no_grad_decorated_methods(self):
        before = self.ns["effective_inference_code_hash"]()

        @torch.no_grad()
        def changed_predict(self, images):
            return images + 1

        with patch.object(self.ns["PatchCore"], "predict_with_raw", changed_predict):
            self.assertNotEqual(before, self.ns["effective_inference_code_hash"]())

    def test_sampling_is_reproducible_and_manual_ids_stay_in_test_subset(self):
        rows = [dict(image_id=f"id_{i}", condition="hole" if i % 2 else "clean") for i in range(40)]
        selected = self.ns["select_test_indices"](rows)
        self.assertEqual(selected, self.ns["select_test_indices"](rows))
        self.assertEqual(len(selected), 20)
        self.assertEqual(len(set(selected)), 20)
        with patch.dict(self.ns, SAMPLE_CONDITION="hole", SELECTED_IMAGE_IDS=["id_1", "id_3"]):
            self.assertEqual(self.ns["select_test_indices"](rows), [1, 3])
            with self.assertRaisesRegex(ValueError, "outside"):
                self.ns["select_test_indices"]([rows[0]])

    def test_tinyglass_uses_smoothed_prediction_not_returned_native_grid(self):
        prediction_type = self.ns["AnomalyPrediction"]

        class FakeTinyGlass:
            def predict_with_raw(self, images):
                b, _, h, w = images.shape
                prediction = prediction_type(torch.full((b,), 0.6), torch.full((b, 1, h, w), 0.25))
                return prediction, torch.full((b,), 0.6), torch.full((b, 1, 2, 2), 0.875)

        profile = dict(family="tinyglass", input_size=(8, 8))
        result = self.ns["capture_prediction"](FakeTinyGlass(), torch.zeros(1, 3, 8, 8), profile)
        np.testing.assert_array_equal(result["anomaly_map"], 0.25)
        np.testing.assert_array_equal(result["returned_raw_map"], 0.875)
        self.assertEqual(result["returned_raw_map"].shape, (1, 1, 2, 2))

    def test_display_retains_original_normalized_response_without_contrast_stretch(self):
        scale = self.ns["display_scale"]()
        values = np.array([[0.48, 0.50, 0.52]], dtype=np.float32)
        result = self.ns["colour_map"](values, np.ones_like(values, dtype=bool), scale)
        expected = (self.ns["matplotlib"].colormaps["magma"](values)[..., :3] * 255).round().astype(np.uint8)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual((scale["lower"], scale["upper"]), (0, 1))
        self.assertFalse(scale["contrast_rescaling"])

    def test_real_patchcore_fit_and_inference_contract_in_embedded_code(self):
        model = self.ns["PatchCore"](pretrained=False, coreset_sampling_ratio=0.5,
            num_neighbors=2, max_patches_per_image=4, max_training_embeddings=8,
            max_memory_bank_size=2, projection_dim=4, calibration_batches=1,
            distance_query_chunk_size=16, distance_bank_chunk_size=2, gaussian_sigma=1)
        images = torch.rand(2, 3, 64, 64)
        model.fit([dict(image=images, label=torch.zeros(2, dtype=torch.long))], device="cpu")
        result = self.ns["capture_prediction"](model, images[:1], dict(input_size=(64, 64)))
        self.assertEqual(model.fit_summary["num_training_images"], 2)
        self.assertEqual(result["anomaly_map"].shape, (1, 1, 64, 64))
        self.assertTrue(np.isfinite(result["returned_raw_map"]).all())
        self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_incompatible_patchcore_checkpoint_is_rejected_before_constructor(self):
        profile = self.ns["MODEL_PROFILES"]["PC-S"]
        config = self.ns["patchcore_config"](profile)
        config["max_memory_bank_size"] = 2_048
        payload = dict(schema_version=self.ns["CHECKPOINT_SCHEMA_VERSION"],
                       model_class="src.anomaly_detection.models.patchcore.PatchCore", model_config=config)
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(self.temporary_root.resolve()))
            path = Path(temporary) / "model.ckpt"
            torch.save(payload, path)
            with patch.dict(self.ns, PatchCore=lambda **kw: self.fail("Invalid checkpoint should not build a model")):
                with self.assertRaisesRegex(ValueError, "settings differ"):
                    self.ns["load_frozen_patchcore"](path, profile)

    @staticmethod
    def make_dataset(root):
        (root / "images").mkdir()
        (root / "masks").mkdir()
        rgb = np.zeros((900, 1200, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(1200, dtype=np.uint16) % 256
        mask = np.zeros((900, 1200), dtype=np.uint8)
        mask[160:175, 320:335] = 255
        Image.fromarray(rgb).save(root / "images/rgb.png")
        Image.fromarray(mask).save(root / "masks/mask.png")
        rows = []
        for split in ("train", "validation", "test"):
            for condition in (("clean",) if split != "test" else ("clean", "hole")):
                for i in range(2):
                    rows.append(dict(image_id=f"{split}_{condition}_{i}", split=split, condition=condition,
                        image_path="images/rgb.png", target_mask_path="masks/mask.png",
                        anomaly_mask_path="masks/mask.png" if condition == "hole" else "",
                        pair_id=f"pair_{i}" if split == "test" else "", camera_pose="A_overhead"))
        with (root / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_prediction_resume_tamper_detection_and_export(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as temporary:
            self.assertTrue(Path(temporary).resolve().is_relative_to(self.temporary_root.resolve()))
            root = Path(temporary)
            self.make_dataset(root)
            output = root / "output"
            output.mkdir()
            manifest_hash = self.ns["sha256_file"](root / "samples.csv")
            settings = dict(DATASET_ROOT=root, OUTPUT_DIR=output, MANIFEST_SHA256=manifest_hash,
                NUM_SAMPLES=2, DEVICE=torch.device("cpu"),
                INFERENCE_CODE_SHA256="test_inference_code", RUNTIME_VERSIONS={"torch": str(torch.__version__)})
            with patch.dict(self.ns, settings):
                dataset_type = self.ns["CuriosityWheelDataset"]
                test = dataset_type(root, "test", self.ns["common_preprocessor"](), ("A_overhead",))
                selection = self.ns["prepare_selection"](test)
                legacy_selection = json.loads(json.dumps(selection))
                legacy_selection["settings"].update(display_quantiles=[0.01, 0.995],
                    display_reference_seed=43, display_references=[dict(image_id="validation_clean_0")])
                self.ns["save_json_atomic"](legacy_selection, output / "selection.json")
                bank = output / "fitted_models/PC-S/model.ckpt"
                bank.parent.mkdir(parents=True)
                bank.write_bytes(b"existing fitted bank")
                self.assertEqual(selection, self.ns["prepare_selection"](test))
                self.assertEqual(bank.read_bytes(), b"existing fitted bank")
                checkpoint = root / "checkpoint.ckpt"
                checkpoint.write_bytes(b"mock checkpoint, not loaded")
                prediction_type = self.ns["AnomalyPrediction"]

                class FakeDetector(torch.nn.Module):
                    def predict_with_raw(self, images):
                        values = images[:, :1].sigmoid()
                        return prediction_type(values.flatten(1).amax(1), values), values.flatten(1).amax(1), values

                def restore(key, profile, path):
                    return FakeDetector(), {"source_manifest_sha256": manifest_hash}

                with patch.dict(self.ns, selection=selection, samples=selection["samples"],
                        SELECTION_SHA256=self.ns["canonical_digest"](selection),
                        resolve_checkpoint=lambda key, profile: (checkpoint, {"source": "test"}),
                        restore_selected_model=restore):
                    first = self.ns["run_one_model"]("PC-S")
                    self.assertEqual(len(first["predictions"]), 2)
                    with patch.dict(self.ns, restore_selected_model=lambda *a: self.fail("Completed maps should be reused")):
                        self.assertEqual(first, self.ns["run_one_model"]("PC-S"))
                    paths = self.ns["prediction_paths"]("PC-S", selection["samples"][0])
                    with np.load(paths["arrays"], allow_pickle=False) as arrays:
                        self.assertEqual(arrays["model_ground_truth"].shape, (384, 512))
                        self.assertEqual(arrays["aligned_anomaly_map"].shape, (720, 720))
                    with paths["arrays"].open("ab") as stream:
                        stream.write(b"changed")
                    self.assertIsNone(self.ns["completed_model"]("PC-S", first["signature"]))
                    self.ns["run_one_model"]("PC-S")
                    manifests = {key: {"signature": "test"} for key in self.ns["MODEL_ORDER"]}
                    sample = selection["samples"][0]
                    for key in self.ns["MODEL_ORDER"]:
                        array, valid = self.ns["align_to_common"](np.ones((8, 8)), self.ns["MODEL_PROFILES"][key])
                        target = self.ns["prediction_paths"](key, sample)["heatmap"]
                        self.ns["save_png"](target, self.ns["colour_map"](array, valid, {"lower": 0, "upper": 1}))
                        np.savez_compressed(self.ns["prediction_paths"](key, sample)["arrays"],
                                            aligned_anomaly_map=array, valid_mask=valid)
                    with patch.dict(self.ns, model_manifests=manifests, FIGURE_DPI=80):
                        figure = self.ns["render_comparison"](sample)
                    self.assertTrue((output / figure["paths"]["pdf"]).is_file())
                    with Image.open(output / figure["paths"]["png"]) as image:
                        self.assertEqual(image.size, (960, 532))


if __name__ == "__main__":
    unittest.main()
