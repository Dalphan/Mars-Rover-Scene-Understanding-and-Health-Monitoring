from pathlib import Path

from scripts.check_notebook_config_sync import check_sync


def test_notebook_constants_match_python_configs():
    root = Path(__file__).resolve().parents[1]
    assert check_sync(root) == []
