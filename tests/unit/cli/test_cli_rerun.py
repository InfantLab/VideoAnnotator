"""`videoannotator job rerun` (spec 019), and the job commands' API key."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from videoannotator.cli import app

runner = CliRunner()


def _response(status, body):
    resp = MagicMock(status_code=status, ok=status < 400, content=b"x", text=str(body))
    resp.json.return_value = body
    return resp


def test_rerun_posts_overrides_with_the_key(tmp_path):
    settings = tmp_path / "s.yaml"
    settings.write_text("scene_detection:\n  threshold: 35\n")
    new = {"id": "j2", "status": "pending"}
    with patch("requests.request", return_value=_response(201, new)) as req:
        result = runner.invoke(
            app,
            ["job", "rerun", "j1", "--pipelines", "scene_detection",
             "--config", str(settings), "--api-key", "va_k"],
        )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Job j2 runs j1 again" in result.output
    method, url = req.call_args.args
    assert (method, url) == ("POST", "http://127.0.0.1:18011/api/v1/jobs/j1/rerun")
    assert req.call_args.kwargs["json"] == {
        "selected_pipelines": ["scene_detection"],
        "config": {"scene_detection": {"threshold": 35}},
    }
    assert req.call_args.kwargs["headers"] == {"Authorization": "Bearer va_k"}


def test_rerun_explains_why_not():
    body = {
        "error": {"message": "Cannot run job j1 again: its video is no longer stored"}
    }
    with patch("requests.request", return_value=_response(409, body)):
        result = runner.invoke(app, ["job", "rerun", "j1"])
    assert result.exit_code == 1
    assert "no longer stored" in result.output


def test_job_status_sends_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_API_KEY", "va_env")
    body = {"id": "j1", "status": "completed", "created_at": "x"}
    with patch("requests.get", return_value=_response(200, body)) as get:
        runner.invoke(app, ["job", "status", "j1"])
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer va_env"}
