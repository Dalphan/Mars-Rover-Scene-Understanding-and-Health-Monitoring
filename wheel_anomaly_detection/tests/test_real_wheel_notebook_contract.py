import ast
import json
import unittest
from pathlib import Path


class RealWheelNotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        notebook_path = (
            cls.root
            / "notebooks"
            / "kaggle_real_wheel_zero_shot.ipynb"
        )
        cls.notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cls.cells = {cell.get("id"): cell for cell in cls.notebook["cells"]}

    def test_notebook_runs_and_renders_the_real_pilot(self):
        expected_ids = {
            "real-pilot-evaluation-definitions",
            "run-real-inference",
            "save-and-display-real-results",
            "real-results-interpretation",
        }
        self.assertTrue(expected_ids <= set(self.cells))
        cell_ids = [cell.get("id") for cell in self.notebook["cells"]]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in self.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for token in (
            "evaluate_real_wheel_pilot(",
            "plot_real_wheel_predictions(",
            "save_real_wheel_pilot_results(",
            '"image_auroc"',
            '"pixel_average_precision"',
            '"aupro_by_max_fpr"',
            "raw_anomaly_score",
            "normalized_anomaly_score",
        ):
            self.assertIn(token, source)

        namespace = {}
        exec("".join(self.cells["configuration"]["source"]), namespace)
        self.assertEqual(namespace["BATCH_SIZE"], 1)
        self.assertEqual(
            namespace["DATASET_ROOT"],
            Path("/kaggle/input/datasets/dalphan01/real-mars-rover-wheel-ad/real_wheel_pilot_v2"),
        )
        self.assertEqual(namespace["METRIC_HISTOGRAM_BINS"], 2048)
        self.assertEqual(namespace["PRO_MAX_FPR"], 0.30)

    def test_drive_restore_does_not_hash_checkpoints(self):
        source = "".join(
            self.cells["google-drive-checkpoint-download"].get("source", [])
        )
        configuration = "".join(self.cells["configuration"].get("source", []))
        self.assertNotIn("sha256", source.lower())
        self.assertNotIn("checkpoint_sha256", configuration)
        self.assertIn("expected_size", source)
        self.assertIn("checkpoint_file_id", source)

    def test_minimal_dataset_loader_matches_source_module(self):
        notebook_source = "".join(self.cells["real-wheel-loader"]["source"])
        self.assertNotIn("discover_real_dataset_root", notebook_source)
        self.assertNotIn("audit_real_crop_contract", notebook_source)
        notebook_tree = ast.parse(notebook_source)
        module_tree = ast.parse(
            (
                self.root
                / "src/anomaly_detection/data/real_wheel_dataset.py"
            ).read_text(encoding="utf-8")
        )
        for name in ("_read_csv", "load_real_wheel_records", "RealWheelCropDataset"):
            notebook_node = next(
                node for node in notebook_tree.body if getattr(node, "name", None) == name
            )
            module_node = next(
                node for node in module_tree.body if getattr(node, "name", None) == name
            )
            self.assertEqual(
                ast.dump(notebook_node, include_attributes=False),
                ast.dump(module_node, include_attributes=False),
                name,
            )

    def test_notebook_evaluation_definitions_match_source_modules(self):
        notebook_tree = ast.parse(
            "".join(
                self.cells["real-pilot-evaluation-definitions"].get("source", [])
            )
        )
        notebook_nodes = {
            node.name: node
            for node in notebook_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        sources = {
            self.root / "src/anomaly_detection/evaluation/metrics.py": (
                "ExactBinaryMetrics",
                "BinaryHistogramMetrics",
            ),
            self.root / "src/anomaly_detection/evaluation/diagnostics.py": (
                "_connected_components",
                "PerRegionOverlap",
            ),
            self.root / "src/anomaly_detection/evaluation/real_pilot.py": (
                "_batch_metadata_value",
                "evaluate_real_wheel_pilot",
                "plot_real_wheel_predictions",
                "save_real_wheel_pilot_results",
            ),
        }
        for path, names in sources.items():
            module_tree = ast.parse(path.read_text(encoding="utf-8"))
            module_nodes = {
                node.name: node
                for node in module_tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            }
            for name in names:
                self.assertEqual(
                    ast.dump(notebook_nodes[name], include_attributes=False),
                    ast.dump(module_nodes[name], include_attributes=False),
                    name,
                )

    def test_every_code_cell_compiles(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), f"<{cell.get('id')}>", "exec")


if __name__ == "__main__":
    unittest.main()
