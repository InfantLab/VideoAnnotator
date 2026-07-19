"""Integration test: extras groups actually isolate their dependencies.

Installs `videoannotator[scene]` into a throwaway virtual environment and
confirms torch/open-clip-torch are importable there while pyannote.audio/
ultralytics/deepface are not (quickstart.md §1, User Story 1).

This is a real network+build operation (pip/uv resolving and downloading
torch et al.), so it's opt-in like the rest of this repo's "_real" /
network-bound suites: set VIDEOANNOTATOR_RUN_EXTRAS_ISOLATION=1 to run it.
It's meant for a dedicated CI job (plan.md), not routine local `pytest tests/`.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_FLAG = "VIDEOANNOTATOR_RUN_EXTRAS_ISOLATION"


def _skip_unless_enabled():
    if os.environ.get(RUN_FLAG) != "1":
        pytest.skip(
            f"Set {RUN_FLAG}=1 to run the (network+build heavy) extras "
            "isolation test — see tests/integration/test_extras_isolation.py"
        )
    if shutil.which("uv") is None:
        pytest.skip("uv is not on PATH; required to install into a clean venv")


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _is_importable(python: Path, module: str) -> bool:
    result = subprocess.run(
        [str(python), "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


class TestExtrasIsolation:
    def test_scene_extra_pulls_only_its_own_deps(self, tmp_path):
        """`pip install videoannotator[scene]` gets torch/open-clip-torch,
        not pyannote/ultralytics/deepface (research.md §1)."""
        _skip_unless_enabled()

        venv_dir = tmp_path / "va-scene"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = _venv_python(venv_dir)

        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                f"{REPO_ROOT}[scene]",
            ],
            check=True,
        )

        assert _is_importable(python, "torch")
        assert _is_importable(python, "open_clip")
        assert _is_importable(python, "scenedetect")

        assert not _is_importable(python, "pyannote.audio")
        assert not _is_importable(python, "ultralytics")
        assert not _is_importable(python, "deepface")

    def test_scene_pipeline_available_face_pipeline_is_not(self, tmp_path):
        """The registry reflects the same isolation: `scene_detection`
        loads, `face_analysis` doesn't, with an actionable reason
        (quickstart.md §1)."""
        _skip_unless_enabled()

        venv_dir = tmp_path / "va-scene-registry"
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
        python = _venv_python(venv_dir)

        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--quiet",
                f"{REPO_ROOT}[scene]",
            ],
            check=True,
        )

        check_script = (
            "from videoannotator.registry.pipeline_loader import get_pipeline_loader\n"
            "classes = get_pipeline_loader().load_all_pipelines()\n"
            "assert 'scene_detection' in classes, classes.keys()\n"
            "assert 'face_analysis' not in classes, classes.keys()\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [str(python), "-c", check_script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "OK" in result.stdout
