import tomllib
from pathlib import Path

import slm_weight_editing


def test_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8"))
    assert slm_weight_editing.__version__ == pyproject["project"]["version"]
