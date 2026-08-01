from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from src.blender_audit.validation import validate_output


def write_minimal_png(path, width=800, height=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )


class BlenderAuditValidationTests(unittest.TestCase):
    def test_validate_complete_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports = root / "reports"
            diagnostics = root / "diagnostics"
            logs = root / "logs"
            reports.mkdir()
            diagnostics.mkdir()
            logs.mkdir()
            (reports / "audit.md").write_text("# Audit", encoding="utf-8")
            (diagnostics / "imported_asset.blend").write_bytes(b"BLENDER")
            (logs / "audit.log").write_text("completed", encoding="utf-8")

            rover = []
            for index in range(6):
                relative = f"renders/rover/view_{index}.png"
                write_minimal_png(root / relative)
                rover.append(relative)
            closeups = []
            for index in range(3):
                relative = f"renders/candidates/wheel_01/view_{index}.png"
                write_minimal_png(root / relative)
                closeups.append(relative)
            wireframe = "renders/topology/wireframe.png"
            write_minimal_png(root / wireframe)

            report = {
                "schema_version": "1.0",
                "status": "completed",
                "asset": {
                    "sha256_before": "abc",
                    "sha256_after": "abc",
                    "unchanged": True,
                },
                "environment": {
                    "render_resolution": {"width": 800, "height": 600}
                },
                "wheel_detection": {
                    "candidates": [
                        {
                            "id": "wheel_01",
                            "renders": {
                                "closeups": closeups,
                                "wireframe": wireframe,
                            },
                        }
                    ]
                },
                "classification": {"category": "A"},
                "materials": {"missing_textures": []},
                "artifacts": {
                    "rover_renders": rover,
                    "topology_renders": [wireframe],
                },
            }
            (reports / "audit.json").write_text(
                json.dumps(report), encoding="utf-8"
            )

            result = validate_output(root)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["checked_file_count"], 14)

    def test_validate_detects_incomplete_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reports = root / "reports"
            reports.mkdir(parents=True)
            (reports / "audit.json").write_text("{}", encoding="utf-8")
            result = validate_output(root)
            self.assertFalse(result["valid"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
