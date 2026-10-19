"""`videoannotator storage-cleanup`: its exit code says whether cleanup worked."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from videoannotator.cli import app
from videoannotator.storage.cleanup import CleanupResult

runner = CliRunner()


def _run(result: CleanupResult):
    with patch("videoannotator.storage.cleanup.cleanup_old_jobs", return_value=result):
        return runner.invoke(app, ["storage-cleanup", "--retention-days", "30"])


def test_a_dry_run_with_nothing_to_do_succeeds():
    outcome = _run(CleanupResult())
    assert outcome.exit_code == 0, outcome.output
    assert "Cleanup failed" not in outcome.output


def test_errors_during_cleanup_exit_1():
    result = CleanupResult()
    result.errors.append("job-1: permission denied")
    outcome = _run(result)
    assert outcome.exit_code == 1
    assert "1 errors occurred" in outcome.output


def test_a_crash_exits_1_with_the_reason():
    with patch(
        "videoannotator.storage.cleanup.cleanup_old_jobs",
        side_effect=RuntimeError("database locked"),
    ):
        outcome = runner.invoke(app, ["storage-cleanup", "--retention-days", "30"])
    assert outcome.exit_code == 1
    assert "database locked" in outcome.output
