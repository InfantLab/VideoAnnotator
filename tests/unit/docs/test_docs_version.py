"""Every docs page names the version it documents (the site build's hook)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from videoannotator.cli import app
from videoannotator.version import __version__

REPO = Path(__file__).resolve().parents[3]


def _hooks():
    spec = importlib.util.spec_from_file_location(
        "mkdocs_hooks", REPO / "scripts" / "mkdocs_hooks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_site_name_and_footer_carry_the_package_version():
    config = {"docs_dir": str(REPO / "docs"), "extra": {}, "site_name": "x"}
    config = _hooks().on_config(config)
    assert config["site_name"] == f"VideoAnnotator {__version__}"
    assert f"VideoAnnotator {__version__}" in config["copyright"]
    assert "videoannotator --version" in config["copyright"]


def test_cli_version_flag_matches():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"VideoAnnotator v{__version__}"
