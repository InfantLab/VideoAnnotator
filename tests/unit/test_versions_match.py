"""One product, one version: pyproject, version.py and the bundled viewer agree."""

import json
import tomllib
from pathlib import Path

from packaging.version import Version

from videoannotator.version import __version__, __version_info__

ROOT = Path(__file__).resolve().parents[2]


def test_version_py_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["version"] == __version__
    assert Version(__version__).release == __version_info__[:3]


# npm spells a release candidate 1.6.0-rc1 where Python spells it 1.6.0rc1.
def test_viewer_package_matches():
    viewer = json.loads((ROOT / "viewer" / "package.json").read_text())
    assert Version(viewer["version"]) == Version(__version__)
