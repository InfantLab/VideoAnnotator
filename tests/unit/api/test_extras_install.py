"""Unit tests for api/extras_install.py: command selection, dedup tracking,
and the run_install/start_install execution path.

None of these tests invoke a real `pip`/`uv` install -- `subprocess.run` is
always mocked, even in the run_install tests below.
"""

import subprocess
import threading
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
    extras_install._activation.clear()
    yield
    extras_install._in_flight.clear()
    extras_install._restart_required = False
    extras_install._activation.clear()


def _fake_dist(requires):
    dist = MagicMock()
    dist.requires = requires
    return dist


class TestExtraRequirements:
    REQUIRES = [
        "fastapi>=0.115.0",
        'torch==2.6.0; extra == "scene"',
        'scenedetect>=0.6.3; extra == "scene"',
        'torch==2.6.0; extra == "person"',
        'ultralytics>=8.3.0; extra == "person"',
        'pywin32>=300; sys_platform == "nonexistent-os" and extra == "scene"',
        'videoannotator[person,scene]; extra == "all"',
    ]

    def _requirements(self, extra):
        with patch.object(
            extras_install.importlib.metadata,
            "distribution",
            return_value=_fake_dist(self.REQUIRES),
        ):
            return extras_install.extra_requirements(extra)

    def test_returns_only_the_groups_own_requirements_without_markers(self):
        assert self._requirements("scene") == ["torch==2.6.0", "scenedetect>=0.6.3"]

    def test_drops_requirements_whose_platform_marker_does_not_apply(self):
        assert not any("pywin32" in r for r in self._requirements("scene"))

    def test_expands_a_meta_group_and_dedupes(self):
        assert self._requirements("all") == [
            "torch==2.6.0",
            "ultralytics>=8.3.0",
            "scenedetect>=0.6.3",
        ]

    def test_reads_this_installs_real_metadata(self):
        # No mocking: the llm group is declared in pyproject.toml.
        assert any(
            r.startswith("ollama") for r in extras_install.extra_requirements("llm")
        )


class TestResolveInstallCommand:
    def test_uses_uv_pip_against_the_running_interpreter(self):
        with (
            patch.object(
                extras_install, "extra_requirements", return_value=["a>=1", "b"]
            ),
            patch.object(extras_install.shutil, "which", return_value="/usr/bin/uv"),
        ):
            command, cwd = extras_install.resolve_install_command("scene")
        assert command == [
            "uv",
            "pip",
            "install",
            "--python",
            extras_install.sys.executable,
            "a>=1",
            "b",
        ]
        assert cwd is None

    def test_uses_pip_when_uv_is_not_on_path(self):
        with (
            patch.object(extras_install, "extra_requirements", return_value=["a>=1"]),
            patch.object(extras_install.shutil, "which", return_value=None),
        ):
            command, cwd = extras_install.resolve_install_command("scene")
        assert command == [
            extras_install.sys.executable,
            "-m",
            "pip",
            "install",
            "a>=1",
        ]
        assert cwd is None

    def test_never_re_syncs_the_environment(self):
        command, _ = extras_install.resolve_install_command("llm")
        assert "sync" not in command

    def test_falls_back_to_pinned_extra_when_metadata_lists_nothing(self):
        with (
            patch.object(extras_install, "extra_requirements", return_value=[]),
            patch.object(extras_install.shutil, "which", return_value=None),
            patch.object(
                extras_install.importlib.metadata, "version", return_value="1.5.0"
            ),
        ):
            command, _ = extras_install.resolve_install_command("scene")
        assert command[-1] == "videoannotator[scene]==1.5.0"


class TestDecideActivation:
    def test_only_new_distributions_activate_live(self):
        outcome = extras_install.decide_activation(
            before={"numpy": "2.1.0"},
            after={"numpy": "2.1.0", "scenedetect": "0.6.4"},
            modules={"scenedetect": {"scenedetect"}},
            loaded={"numpy"},
        )
        assert outcome == {"activation": "live", "conflicting_distributions": []}

    def test_changed_version_of_an_imported_distribution_needs_restart(self):
        outcome = extras_install.decide_activation(
            before={"numpy": "1.26.4"},
            after={"numpy": "2.1.0"},
            modules={"numpy": {"numpy"}},
            loaded={"numpy"},
        )
        assert outcome["activation"] == "restart_required"
        assert outcome["conflicting_distributions"] == [
            {"name": "numpy", "old_version": "1.26.4", "new_version": "2.1.0"}
        ]

    def test_changed_version_of_an_unimported_distribution_stays_live(self):
        outcome = extras_install.decide_activation(
            before={"pillow": "10.0.0"},
            after={"pillow": "11.0.0"},
            modules={"pillow": {"PIL"}},
            loaded={"numpy"},
        )
        assert outcome["activation"] == "live"

    def test_module_name_differing_from_distribution_name_is_detected(self):
        outcome = extras_install.decide_activation(
            before={"pyyaml": "6.0.1"},
            after={"pyyaml": "6.0.2"},
            modules={"pyyaml": {"yaml", "_yaml"}},
            loaded={"yaml"},
        )
        assert outcome["activation"] == "restart_required"

    def test_removed_imported_distribution_needs_restart(self):
        outcome = extras_install.decide_activation(
            before={"opencv-python-headless": "4.10.0"},
            after={},
            modules={"opencv-python-headless": {"cv2"}},
            loaded={"cv2"},
        )
        assert outcome["conflicting_distributions"][0]["new_version"] is None


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
    def test_success_updates_job_and_activates_live(self, temp_db, db_session):
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
        # Nothing changed on disk (subprocess is mocked): live, no restart.
        assert extras_install.activation_for(job_id)["activation"] == "live"
        assert extras_install.restart_required() is False
        # dedup entry must be cleared once the job resolves.
        assert "scene" not in extras_install._in_flight

    def test_conflicting_install_records_restart_required(self, temp_db, db_session):
        job = ExtrasInstallJob(extra_name="face", status=ExtrasInstallJobStatus.PENDING)
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        job_id = str(job.id)
        extras_install.try_begin_install("face", job_id)

        snapshots = iter(
            [
                ({"numpy": "1.26.4"}, {"numpy": {"numpy"}}),
                ({"numpy": "2.1.0"}, {"numpy": {"numpy"}}),
            ]
        )
        with (
            patch.object(
                subprocess,
                "run",
                return_value=MagicMock(returncode=0, stdout="", stderr=""),
            ),
            patch.object(
                extras_install, "resolve_install_command", return_value=(["pip"], None)
            ),
            patch.object(
                extras_install,
                "_installed_distributions",
                side_effect=lambda: next(snapshots),
            ),
        ):
            extras_install.run_install(job_id, "face")

        outcome = extras_install.activation_for(job_id)
        assert outcome["activation"] == "restart_required"
        assert outcome["conflicting_distributions"][0]["name"] == "numpy"
        assert extras_install.restart_required() is True

    def test_job_reads_completed_only_after_activation_is_settled(
        self, temp_db, db_session
    ):
        # The viewer re-fetches the pipeline list the moment it sees
        # `completed`; activation (cache clearing) must already be done.
        job = ExtrasInstallJob(
            extra_name="scene", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        job_id = str(job.id)
        extras_install.try_begin_install("scene", job_id)

        statuses_seen = []

        def snapshot():
            check = extras_install._db_module.SessionLocal()
            try:
                row = check.query(ExtrasInstallJob).filter_by(id=job.id).first()
                statuses_seen.append(row.status)
            finally:
                check.close()
            return ({}, {})

        with (
            patch.object(
                subprocess,
                "run",
                return_value=MagicMock(returncode=0, stdout="", stderr=""),
            ),
            patch.object(
                extras_install, "resolve_install_command", return_value=(["pip"], None)
            ),
            patch.object(
                extras_install, "_installed_distributions", side_effect=snapshot
            ),
        ):
            extras_install.run_install(job_id, "scene")

        assert len(statuses_seen) == 2
        assert ExtrasInstallJobStatus.COMPLETED not in statuses_seen
        db_session.refresh(job)
        assert job.status == ExtrasInstallJobStatus.COMPLETED

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
