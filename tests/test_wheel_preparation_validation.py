import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from src.wheel_preparation.validation import validate_output


def png_chunk(chunk_type, data):
    body = chunk_type + data
    return (
        struct.pack(">I", len(data))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def write_png(path, width=800, height=600):
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    row = b"\x00" + b"\xff\xff\xff\xff" * width
    compressed = zlib.compress(row * height)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        signature
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", compressed)
        + png_chunk(b"IEND", b"")
    )


class WheelPreparationValidationTests(unittest.TestCase):
    def write_valid_output(self, root):
        required = [
            root / "reports" / "preparation.md",
            root / "diagnostics" / "wheel_canonical.blend",
            root / "diagnostics" / "perforation_probe.blend",
            root / "logs" / "preparation.log",
        ]
        for path in required:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"diagnostic")
        image_paths = [
            "renders/extraction/wheel_outboard.png",
            "renders/extraction/wheel_tread.png",
            "renders/extraction/wheel_oblique.png",
            "renders/topology/wheel_topology.png",
            "renders/perforation/normal.png",
            "renders/perforation/anomaly.png",
            "renders/perforation/anomaly_mask.png",
            "renders/perforation/difference.png",
        ]
        for relative in image_paths:
            write_png(root / relative)
        report = {
            "schema_version": "1.0",
            "status": "completed",
            "asset": {
                "sha256_before": "same",
                "sha256_after": "same",
                "unchanged": True,
            },
            "environment": {
                "render_resolution": {"width": 800, "height": 600}
            },
            "configuration": {
                "gates": {
                    "minimum_silhouette_iou": 0.995,
                    "minimum_mask_ratio": 0.002,
                    "maximum_mask_ratio": 0.05,
                    "maximum_outside_change_ratio": 0.01,
                }
            },
            "candidate": {
                "expected_counts": {
                    "vertex_count": 2458,
                    "face_count": 1664,
                    "component_count": 303,
                }
            },
            "extraction": {
                "vertex_count": 2458,
                "face_count": 1664,
                "component_count": 303,
                "coordinates_preserved": True,
                "materials_preserved": True,
                "uv_preserved": True,
                "custom_normals_preserved": True,
                "minimum_silhouette_iou": 1.0,
            },
            "repair": {
                "strategy": "hybrid_parametric_shell",
                "final_topology": {
                    "boundary_edge_count": 0,
                    "multi_face_edge_count": 0,
                    "normals_consistent": True,
                },
            },
            "perforation": {
                "boolean": {"success": True, "closed_manifold": True},
                "mask": {"area_ratio": 0.01, "outside_change_ratio": 0.0},
            },
            "validation": {"valid": True},
            "artifacts": {
                "extraction_renders": image_paths[:3],
                "topology_renders": image_paths[3:4],
                "perforation_renders": image_paths[4:],
            },
        }
        report_path = root / "reports" / "preparation.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report_path

    def test_valid_output_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_valid_output(root)
            result = validate_output(root)
            self.assertTrue(result["valid"], result["errors"])
            self.assertEqual(result["errors"], [])

    def test_mask_ratio_outside_gate_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = self.write_valid_output(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["perforation"]["mask"]["area_ratio"] = 0.2
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = validate_output(root)
            self.assertFalse(result["valid"])
            self.assertTrue(
                any("Mask area ratio" in error for error in result["errors"])
            )

    def test_missing_candidate_is_rejected_by_core_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = self.write_valid_output(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            del report["candidate"]["expected_counts"]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            result = validate_output(root)
            self.assertTrue(
                result["valid"],
                "The artifact validator intentionally tolerates absent audit "
                "expectations; candidate lookup is enforced before Blender work.",
            )


if __name__ == "__main__":
    unittest.main()
