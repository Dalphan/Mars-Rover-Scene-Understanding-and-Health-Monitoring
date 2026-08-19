import copy
import json
import unittest
from pathlib import Path

from src.microterrain.level3_validation import required_level3_renders, validate_level3_report
from src.microterrain.material import validate_level3_config


class MicroterrainMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).parents[1] / "configs" / "blender" / "gale_terrain.json").read_text(encoding="utf-8"))

    def test_material_contract_is_stable(self):
        first = validate_level3_config(self.config)
        second = validate_level3_config(copy.deepcopy(self.config))
        self.assertEqual(first["material_signature_sha256"], second["material_signature_sha256"])
        self.assertEqual(first["validation_distances_m"], [0.05, 0.10])

    def test_rejects_non_subgranular_primary_bump(self):
        invalid = copy.deepcopy(self.config)
        invalid["microterrain"]["material"]["micro_bump_wavelength_m"] = 0.001
        with self.assertRaises(ValueError):
            validate_level3_config(invalid)

    def test_level3_report_gate(self):
        contract = validate_level3_config(self.config)
        report = {
            "level": 3,
            "level2": {"scatter_signature_sha256": "a" * 64},
            "material": {"signature_sha256": contract["material_signature_sha256"], "terrain_material": "GaleTerrainMicroterrain_L3", "terrain_noise_layers": 6, "clast_material_count": 4},
            "geometry": {"level2_modified": False, "scatter_signature_sha256": "a" * 64},
            "ablation": {"level2_materials_preserved": True},
            "scope": {"multi_distance_matrix_generated": False},
            "renders": {name: "x.png" for name in required_level3_renders(self.config)},
        }
        self.assertEqual(validate_level3_report(self.config, report), [])


if __name__ == "__main__":
    unittest.main()
