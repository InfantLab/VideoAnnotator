"""Unit tests for api/extras_install.py: command selection, dedup tracking,
and the run_install/start_install execution path.

None of these tests invoke a real `pip`/`uv` install -- `subprocess.run` is
always mocked, even in the run_install tests below.
"""

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator.api import extras_install
from videoannotator.database.models import ExtrasInstallJob, ExtrasInstallJobStatus


@pytest.fixture(autouse=True)
def _reset_module_state():
    """extras_install keeps module-level dedup/restart state; isolate tests."""
    extras_install._in_flight.clear()
    extras_install._restart_required = False
    yield
    extras_install._in_flight.clear()
    extras_install._restart_required = False


class TestResolveInstallCommand:
    def test_uses_uv_sync_for_editable_checkout_with_uv_on_path(self):
        fake_root = Path("/workspaces/VideoAnnotator")
        with (
            patch.object(
                extras_install, "_editable_checkout_root", return_value=fake_root
            ),
            patch.object(extras_install.shutil, "which", return_value="/usr/bin/uv"),
        ):
            command, cwd = extras_install.resolve_install_command("scene")
        assert command == ["uv", "sync", "--extra", "scene", "--inexact"]
        assert cwd == fake_root

    def test_falls_back_to_pinned_pip_when_not_editable_checkout(self):
        with (
            patch.object(extras_install, "_editable_checkout_root", return_value=None),
            patch.object(
                extras_install.importlib.metadata, "version", return_value="1.5.0"
            ),
        ):
            command, cwd = extras_install.resolve_install_command("scene")
        assert command == [
            extras_install.sys.executable,
            "-m",
            "pip",
            "install",
            "videoannotator[scene]==1.5.0",
        ]
        assert cwd is None

    def test_falls_back_to_pip_when_editable_checkout_but_no_uv_on_path(self):
        fake_root = Path("/workspaces/VideoAnnotator")
        with (
            patch.object(
                extras_install, "_editable_checkout_root", return_value=fake_root
            ),
            patch.object(extras_install.shutil, "which", return_value=None),
            patch.object(
                extras_install.importlib.metadata, "version", return_value="1.5.0"
            ),
        ):
            command, cwd = extras_install.resolve_install_command("audio")
        assert command[0] == extras_install.sys.executable
        assert "pip" in command
        assert cwd is None


class TestDedupTracking:
    def test_first_caller_registers_and_gets_none_back(self):
        result = extras_install.try_begin_install("face", "job-1")
        assert result is None
        assert extras_install._in_flight["face"] == "job-1"

    def test_second_caller_for_same_extra_gets_existing_job_id(self):
        extras_install.try_begin_install("face", "job-1")
        result = extras_install.try_begin_install("face", "job-2")
        assert result == "job-1"
        # The second, rejected attempt must not overwrite the in-flight entry.
        assert extras_install._in_flight["face"] == "job-1"

    def test_different_extras_do_not_collide(self):
        result_a = extras_install.try_begin_install("face", "job-1")
        result_b = extras_install.try_begin_install("audio", "job-2")
        assert result_a is None
        assert result_b is None

    def test_end_install_clears_entry_allowing_a_new_one(self):
        extras_install.try_begin_install("face", "job-1")
        extras_install._end_install("face")
        result = extras_install.try_begin_install("face", "job-2")
        assert result is None
        assert extras_install._in_flight["face"] == "job-2"


class TestRestartRequiredFlag:
    def test_starts_false(self):
        assert extras_install.restart_required() is False

    def test_mark_restart_required_flips_flag(self):
        extras_install._mark_restart_required()
        assert extras_install.restart_required() is True


class TestEditableCheckoutRootRealFilesystem:
    """Exercises the real filesystem-walking logic (no mocking) -- this test
    suite itself runs from an editable checkout of videoannotator, so the
    real function should find this repo's own project root."""

    def test_finds_this_repos_own_root(self):
        root = extras_install._editable_checkout_root()
        assert root is not None
        assert (root / "pyproject.toml").is_file()
        assert 'name = "videoannotator"' in (root / "pyproject.toml").read_text(
            encoding="utf-8"
        )

    def test_returns_none_when_no_sibling_pyproject(self, tmp_path):
        fake_pkg_file = tmp_path / "src" / "videoannotator" / "__init__.py"
        fake_pkg_file.parent.mkdir(parents=True)
        fake_pkg_file.write_text("")
        with patch.object(
            extras_install._videoannotator_pkg, "__file__", str(fake_pkg_file)
        ):
            assert extras_install._editable_checkout_root() is None

    def test_returns_none_when_pyproject_names_different_project(self, tmp_path):
        fake_pkg_file = tmp_path / "src" / "videoannotator" / "__init__.py"
        fake_pkg_file.parent.mkdir(parents=True)
        fake_pkg_file.write_text("")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
        with patch.object(
            extras_install._videoannotator_pkg, "__file__", str(fake_pkg_file)
        ):
            assert extras_install._editable_checkout_root() is None


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test_run_install.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(extras_install._db_module, "engine", engine)
    monkeypatch.setattr(
        extras_install._db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    extras_install._db_module.Base.metadata.create_all(bind=engine)
    yield engine


@pytest.fixture
def db_session(temp_db):
    session = extras_install._db_module.SessionLocal()
    yield session
    session.close()


class TestRunInstall:
    def test_success_updates_job_and_flips_restart_flag(self, temp_db, db_session):
        job = ExtrasInstallJob(
            extra_name="scene", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        job_id = str(job.id)

        extras_install.try_begin_install("scene", job_id)

        fake_result = MagicMock(
            returncode=0, stdout="Successfully installed", stderr=""
        )
        with (
            patch.object(subprocess, "run", return_value=fake_result) as mock_run,
            patch.object(
                extras_install,
                "resolve_install_command",
                return_value=(["pip", "install", "videoannotator[scene]"], None),
            ),
        ):
            extras_install.run_install(job_id, "scene")

        mock_run.assert_called_once()
        db_session.refresh(job)
        assert job.status == ExtrasInstallJobStatus.COMPLETED
        assert job.command_output == "Successfully installed"
        assert job.started_at is not None
        assert job.finished_at is not None
        assert extras_install.restart_required() is True
        # dedup entry must be cleared once the job resolves.
        assert "scene" not in extras_install._in_flight

    def test_nonzero_exit_marks_failed_without_flipping_restart_flag(
        self, temp_db, db_session
    ):
        job = ExtrasInstallJob(
            extra_name="audio", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        job_id = str(job.id)
        extras_install.try_begin_install("audio", job_id)

        fake_result = MagicMock(returncode=1, stdout="", stderr="ERROR: no match")
        with (
            patch.object(subprocess, "run", return_value=fake_result),
            patch.object(
                extras_install,
                "resolve_install_command",
                return_value=(["pip", "install", "videoannotator[audio]"], None),
            ),
        ):
            extras_install.run_install(job_id, "audio")

        db_session.refresh(job)
        assert job.status == ExtrasInstallJobStatus.FAILED
        assert "ERROR: no match" in job.command_output
        assert extras_install.restart_required() is False
        assert "audio" not in extras_install._in_flight

    def test_oserror_launching_command_marks_failed(self, temp_db, db_session):
        job = ExtrasInstallJob(
            extra_name="person", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        job_id = str(job.id)
        extras_install.try_begin_install("person", job_id)

        with (
            patch.object(subprocess, "run", side_effect=OSError("command not found")),
            patch.object(
                extras_install,
                "resolve_install_command",
                return_value=(["uv", "sync", "--extra", "person"], None),
            ),
        ):
            extras_install.run_install(job_id, "person")

        db_session.refresh(job)
        assert job.status == ExtrasInstallJobStatus.FAILED
        assert "command not found" in job.command_output
        assert "person" not in extras_install._in_flight

    def test_missing_job_row_returns_early_without_error(self, temp_db):
        # Should not raise even though no such job exists.
        extras_install.run_install("does-not-exist", "scene")


class TestStartInstall:
    def test_spawns_a_thread_that_calls_run_install(self):
        called_with = {}
        done = threading.Event()

        def fake_run_install(job_id, extra_name):
            called_with["job_id"] = job_id
            called_with["extra_name"] = extra_name
            done.set()

        with patch.object(extras_install, "run_install", side_effect=fake_run_install):
            extras_install.start_install("job-123", "face")
            assert done.wait(timeout=5)

        assert called_with == {"job_id": "job-123", "extra_name": "face"}
