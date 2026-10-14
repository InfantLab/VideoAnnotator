"""`videoannotator dataset ...`: the Datasets page's actions from a terminal (spec 018)."""

import copy
import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from videoannotator.cli import app

runner = CliRunner()

DATASET = {
    "id": "d1",
    "name": "Pilot",
    "owner_user_id": "u1",
    "owner_name": "ada",
    "video_manifest": [
        {"filename": "a.mp4", "size_bytes": 10, "relative_path": "p/a.mp4"}
    ],
    "server_folder": None,
    "server_folder_recursive": False,
    "created_at": "2026-10-04T00:00:00Z",
}


def _response(status=200, body=None):
    resp = MagicMock(status_code=status, ok=status < 400)
    resp.content = b"x" if body is not None else b""
    resp.json.side_effect = lambda: copy.deepcopy(body)
    resp.text = json.dumps(body)
    return resp


def test_list_shows_name_count_source_and_owner():
    with patch(
        "requests.request",
        return_value=_response(body={"datasets": [DATASET], "total": 1}),
    ):
        result = runner.invoke(app, ["dataset", "list"])
    assert result.exit_code == 0, result.output
    assert "Pilot  (1 videos, uploaded, saved by ada)" in result.output


def test_export_then_import_round_trips(tmp_path):
    out = tmp_path / "pilot.json"
    with patch("requests.request", return_value=_response(body=dict(DATASET))) as req:
        assert (
            runner.invoke(app, ["dataset", "export", "d1", "-o", str(out)]).exit_code
            == 0
        )
        exported = json.loads(out.read_text())
        assert "id" not in exported and "owner_user_id" not in exported
        result = runner.invoke(
            app, ["dataset", "import", str(out), "--api-key", "va_x"]
        )
    assert result.exit_code == 0, result.output
    method, url = req.call_args.args
    assert (method, url.endswith("/api/v1/datasets/")) == ("POST", True)
    assert req.call_args.kwargs["json"]["name"] == "Pilot"
    assert req.call_args.kwargs["headers"] == {"Authorization": "Bearer va_x"}


def test_a_name_clash_is_reported_with_the_servers_message(tmp_path):
    clash = _response(
        409, {"error": {"message": "You already have a dataset named 'Pilot'."}}
    )
    exported = tmp_path / "d.json"
    exported.write_text(json.dumps(DATASET))
    with patch("requests.request", return_value=clash):
        result = runner.invoke(app, ["dataset", "import", str(exported)])
    assert result.exit_code == 1
    assert "already have a dataset named 'Pilot'" in result.output


def test_delete_asks_first():
    with patch("requests.request", return_value=_response(204)) as req:
        result = runner.invoke(app, ["dataset", "delete", "d1"], input="n\n")
        assert result.exit_code == 1
        req.assert_not_called()
        assert runner.invoke(app, ["dataset", "delete", "d1", "--yes"]).exit_code == 0
    assert req.call_args.args[0] == "DELETE"


def test_unauthenticated_says_how_to_get_a_key():
    with patch("requests.request", return_value=_response(401, {})):
        result = runner.invoke(app, ["dataset", "list"])
    assert result.exit_code == 1
    assert "generate-token" in result.output
