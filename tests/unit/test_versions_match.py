"""One product, one version: pyproject, version.py and the bundled viewer agree."""

import json
import tomllib
from pathlib import Path

from videoannotator.version import __version__, __version_info__

ROOT = Path(__file__).resolve().parents[2]


def test_version_py_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == __version__
    assert ".".join(str(p) for p in __version_info__[:3]) == __version__


def test_viewer_package_matches():
    viewer = json.loads((ROOT / "viewer" / "package.json").read_text())
    assert viewer["version"] == __version__
