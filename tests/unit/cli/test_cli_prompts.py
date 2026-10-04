"""`videoannotator prompts ...` and `vlm preview` (spec 020)."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from videoannotator.cli import app

runner = CliRunner()


def _response(body):
    resp = MagicMock(status_code=200, ok=True, content=b"x")
    resp.json.return_value = body
    return resp


def _prompt(sha, text, **kw):
    return {"sha256": sha, "text": text, "name": None, "starred": False, "use_count": 1,
            "models": ["gemma4:e4b"], "job_ids": [], "first_used_at": "t", "last_used_at": "t", **kw}  # fmt: skip


def test_list_marks_starred_and_shows_models():
    body = {
        "prompts": [_prompt("a" * 64, "Touch?\nAnswer TOUCH", starred=True)],
        "total": 1,
    }
    with patch("requests.request", return_value=_response(body)):
        result = runner.invoke(app, ["prompts", "list"])
    assert result.exit_code == 0, result.output
    assert "* aaaaaaaaaaaa  Touch?  (1 uses; gemma4:e4b)" in result.output


def test_diff_shows_changed_words_and_names_whitespace_only():
    a, b = (
        _prompt("a" * 64, "Is the adult touching?"),
        _prompt("b" * 64, "Is the parent touching?"),
    )
    with patch("requests.request", side_effect=[_response(a), _response(b)]):
        out = runner.invoke(app, ["prompts", "diff", "a", "b"]).output
    assert "- adult" in out and "+ parent" in out
    c = _prompt("c" * 64, "Is the adult  touching?\n")
    with patch("requests.request", side_effect=[_response(a), _response(c)]):
        assert (
            "only in whitespace"
            in runner.invoke(app, ["prompts", "diff", "a", "c"]).output
        )


def test_vlm_preview_sends_the_absolute_path_and_prompt(tmp_path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("Touch?")
    result_body = {"label": "TOUCH", "reasoning": "hands", "total_time": 1.2,
                   "frames": [{"frame_number": 30}]}  # fmt: skip
    with patch("requests.request", return_value=_response(result_body)) as req:
        result = runner.invoke(
            app,
            ["vlm", "preview", "clip.mp4", "--at", "1", "--model", "gemma4:e4b",
             "--prompt-file", str(prompt), "--burst"],
        )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Label: TOUCH" in result.output
    data = req.call_args.kwargs["data"]
    assert data["prompt"] == "Touch?" and data["sampling_mode"] == "frame_burst"
    assert data["video_path"].endswith("clip.mp4") and data["video_path"].startswith(
        "/"
    )
