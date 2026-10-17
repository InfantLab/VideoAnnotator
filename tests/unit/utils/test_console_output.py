"""What the server's terminal shows, and that API keys never reach the logs."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from videoannotator.api import background_tasks
from videoannotator.api.middleware import request_logging
from videoannotator.utils.logging_config import (
    CONSOLE_LOGGER,
    VERBOSE_ENV,
    ConsoleFilter,
    ConsoleFormatter,
    key_fingerprint,
)

KEY = "va_vT27i4FUXTMArpyt-D3qtZopfklxvE1b8_F2HXGaku4"


def _record(name: str, level: int, msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.delenv(VERBOSE_ENV, raising=False)


@pytest.mark.parametrize(
    ("name", "level", "shown"),
    [
        (CONSOLE_LOGGER, logging.INFO, True),
        ("videoannotator.api", logging.INFO, False),
        ("videoannotator.api", logging.WARNING, True),
        ("videoannotator.api", logging.ERROR, True),
        ("py.warnings", logging.WARNING, False),
        ("torch", logging.WARNING, False),
        ("ultralytics", logging.ERROR, True),
    ],
)
def test_terminal_shows_only_what_matters(name, level, shown):
    assert ConsoleFilter().filter(_record(name, level)) is shown


def test_verbose_shows_everything(monkeypatch):
    monkeypatch.setenv(VERBOSE_ENV, "1")
    assert ConsoleFilter().filter(_record("videoannotator.api", logging.INFO))
    assert ConsoleFilter().filter(_record("py.warnings", logging.WARNING))


def test_format_is_short_and_tagged():
    f = ConsoleFormatter()
    info = f.format(_record(CONSOLE_LOGGER, logging.INFO, "Started clip.mp4"))
    warn = f.format(_record("videoannotator.api", logging.WARNING, "Disk nearly full"))
    assert info.endswith(" Started clip.mp4")
    assert len(info.split(" ")[0]) == 8  # HH:MM:SS
    assert "[WARNING] Disk nearly full" in warn
    assert "videoannotator.api" not in warn


def test_traceback_stays_in_the_log_files():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record("videoannotator.api", logging.ERROR, "Job failed")
        record.exc_info = sys.exc_info()
    line = ConsoleFormatter().format(record)
    assert "Traceback" not in line
    assert "(details in logs/errors.log)" in line


def test_fingerprint_reveals_none_of_the_key():
    fp = key_fingerprint(KEY)
    assert fp.startswith("key:")
    assert KEY[3:11] not in fp
    assert fp == key_fingerprint(KEY)


@pytest.fixture
def app():
    request_logging._auth_hint_last.clear()
    app = FastAPI()
    app.add_middleware(request_logging.RequestLoggingMiddleware)

    @app.get("/api/v1/jobs")
    async def jobs():
        return JSONResponse({"error": "no"}, status_code=401)

    return TestClient(app)


def test_api_key_never_in_log_records(app, caplog):
    with caplog.at_level(logging.DEBUG):
        app.get("/api/v1/jobs", headers={"Authorization": f"Bearer {KEY}"})
    assert caplog.records
    for record in caplog.records:
        assert KEY[:12] not in str(record.__dict__)


def test_missing_key_hint_once_per_client(app, caplog):
    with caplog.at_level(logging.WARNING, logger=CONSOLE_LOGGER):
        for _ in range(3):
            app.get("/api/v1/jobs")
        app.get("/api/v1/jobs", headers={"Authorization": "Bearer va_bogus"})
    hints = [r.getMessage() for r in caplog.records if r.name == CONSOLE_LOGGER]
    assert len(hints) == 2
    assert "without an API key" in hints[0]
    assert "videoannotator generate-token" in hints[0]
    assert "doesn't know" in hints[1]


def test_job_lines_read_well():
    start = datetime(2026, 10, 17, 15, 0, 0)
    job = SimpleNamespace(
        job_id="1a2b3c4d5e6f",
        video_path="/videos/clip.mp4",
        selected_pipelines=["face_analysis", "speech_recognition"],
        started_at=start,
        completed_at=start + timedelta(minutes=1, seconds=12),
        error_message="CUDA out of memory\nTraceback...",
    )
    assert background_tasks._job_label(job) == "clip.mp4 [1a2b3c4d]"
    assert background_tasks._pipelines_label(job) == "face_analysis, speech_recognition"
    assert background_tasks._took(job) == " in 1m 12s"
    assert background_tasks._reason(job) == "CUDA out of memory"
