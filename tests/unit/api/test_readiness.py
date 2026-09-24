"""Unit tests for api/readiness.py (specs/011-pipeline-readiness, contract §1-2)."""

import time
from unittest.mock import patch

import pytest

from videoannotator.api import extras_install, readiness
from videoannotator.registry import pipeline_loader
from videoannotator.registry.pipeline_registry import (
    PipelineConfigField,
    PipelineMetadata,
    PipelineOutputFormat,
    SetupRequirement,
    WeightSpec,
)

HF_SECRET = SetupRequirement(
    kind="secret",
    name="HF_AUTH_TOKEN",
    aliases=["HUGGINGFACE_TOKEN"],
    description="A Hugging Face access token",
    help_url="https://huggingface.co/settings/tokens",
)
LICENCE = SetupRequirement(
    kind="licence",
    name="pyannote/speaker-diarization-3.1",
    description="Accept the licence.",
    help_url="https://huggingface.co/pyannote/speaker-diarization-3.1",
)
OLLAMA = SetupRequirement(kind="service", name="ollama")


def _meta(
    name="speaker_diarization", extras=("audio",), setup=(), weights=(), config=None
):
    return PipelineMetadata(
        name=name,
        display_name=name,
        description="",
        outputs=[PipelineOutputFormat(format="JSON", types=[])],
        config_schema=config or {},
        version=1,
        requires_extras=list(extras),
        module_path="x:y",
        requires_setup=list(setup),
        weights=list(weights),
    )


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("HF_AUTH_TOKEN", "HUGGINGFACE_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    extras_install._in_flight.clear()
    extras_install._restart_pending_extras.clear()
    pipeline_loader.clear_import_errors()
    readiness._ollama_cache.clear()
    readiness._ollama_refreshing.clear()
    with patch.object(readiness, "extras_available", return_value=True):
        yield
    extras_install._in_flight.clear()
    extras_install._restart_pending_extras.clear()
    pipeline_loader.clear_import_errors()


class TestStates:
    def test_installing_wins_over_everything(self):
        extras_install._in_flight["audio"] = "job-1"
        with patch.object(readiness, "extras_available", return_value=False):
            r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET]))
        assert (r["state"], r["next_action"], r["install_job_id"]) == (
            "installing",
            "wait",
            "job-1",
        )

    def test_not_installed(self):
        with patch.object(readiness, "extras_available", return_value=False):
            r = readiness.pipeline_readiness(_meta())
        assert (r["state"], r["next_action"], r["extras_group"]) == (
            "not_installed",
            "install",
            "audio",
        )

    def test_restart_required_after_a_conflicting_install(self):
        extras_install._restart_pending_extras.add("audio")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET]))
        assert (r["state"], r["next_action"]) == ("restart_required", "restart")
        assert r["blockers"] == []

    def test_missing_secret_needs_setup_and_says_where_to_set_it(self):
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert (r["state"], r["next_action"]) == ("needs_setup", "setup")
        [blocker] = r["blockers"]
        assert blocker["kind"] == "secret"
        assert blocker["name"] == "HF_AUTH_TOKEN"
        assert "server's environment" in blocker["message"]
        assert blocker["help_url"] == "https://huggingface.co/settings/tokens"

    def test_secret_value_never_appears(self, monkeypatch):
        monkeypatch.setenv("HF_AUTH_TOKEN", "hf_sentinel_value")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert "hf_sentinel_value" not in repr(r)

    def test_alias_counts_as_set(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "x")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert r["state"] == "ready"
        assert [n["kind"] for n in r["notes"]] == ["licence"]

    def test_licence_is_a_note_never_a_blocker(self, monkeypatch):
        monkeypatch.setenv("HF_AUTH_TOKEN", "x")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert r["state"] == "ready" and r["next_action"] == "none"
        assert r["notes"][0]["help_url"].endswith("speaker-diarization-3.1")

    def test_import_error_blocks(self):
        pipeline_loader._import_errors["speaker_diarization"] = (
            "ImportError: libsndfile"
        )
        r = readiness.pipeline_readiness(_meta())
        assert r["state"] == "needs_setup"
        assert r["blockers"][0]["kind"] == "import_error"
        assert "libsndfile" in r["blockers"][0]["message"]

    def test_uncached_weights_are_a_note(self):
        weight = WeightSpec(id="base", approx_mb=140, cache="whisper")
        with patch.object(readiness, "_weights_cached", return_value=False):
            r = readiness.pipeline_readiness(_meta(weights=[weight]))
        assert r["state"] == "ready"
        assert r["notes"] == [
            {
                "kind": "weights_not_cached",
                "name": "base",
                "message": "The first run downloads about 140 MB of model weights (base).",
                "help_url": None,
                "approx_mb": 140,
            }
        ]

    def test_cached_weights_say_nothing(self):
        weight = WeightSpec(id="base", approx_mb=140, cache="whisper")
        with patch.object(readiness, "_weights_cached", return_value=True):
            r = readiness.pipeline_readiness(_meta(weights=[weight]))
        assert r["notes"] == []


class TestOllama:
    def _vlm(self):
        return _meta(
            name="vlm_annotation",
            extras=("llm",),
            setup=[OLLAMA],
            config={
                "base_url": PipelineConfigField(type="string", default="http://h:1")
            },
        )

    def test_unreachable_blocks_with_the_configured_url(self):
        status = {"ollama_reachable": False, "base_url": "http://h:1", "models": []}
        with patch.object(readiness, "_ollama_status", return_value=status) as check:
            r = readiness.pipeline_readiness(self._vlm())
        check.assert_called_once_with("http://h:1")
        assert r["state"] == "needs_setup"
        assert "isn't reachable at http://h:1" in r["blockers"][0]["message"]

    def test_reachable_without_models_blocks(self):
        status = {"ollama_reachable": True, "base_url": "http://h:1", "models": []}
        with patch.object(readiness, "_ollama_status", return_value=status):
            r = readiness.pipeline_readiness(self._vlm())
        assert "no models pulled" in r["blockers"][0]["message"]

    def test_reachable_with_models_is_ready(self):
        status = {"ollama_reachable": True, "base_url": "http://h:1", "models": ["m"]}
        with patch.object(readiness, "_ollama_status", return_value=status):
            assert readiness.pipeline_readiness(self._vlm())["state"] == "ready"

    def test_checks_are_cached_and_stale_ones_refresh_in_the_background(self):
        calls = []

        def diagnose(base_url, timeout):
            calls.append(base_url)
            return {"ollama_reachable": False, "base_url": base_url, "models": []}

        with patch(
            "videoannotator.diagnostics.ollama.diagnose_ollama", side_effect=diagnose
        ):
            readiness._ollama_status("http://h:1")
            readiness._ollama_status("http://h:1")
            assert len(calls) == 1  # cached

            # Age the entry past the TTL: the stale value comes back at once,
            # and a background refresh runs.
            stamp, status = readiness._ollama_cache["http://h:1"]
            readiness._ollama_cache["http://h:1"] = (
                stamp - readiness._OLLAMA_TTL_S - 1,
                status,
            )
            assert readiness._ollama_status("http://h:1") is status
            deadline = time.monotonic() + 5
            while len(calls) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        assert len(calls) == 2


class TestExtrasGroups:
    def test_lists_only_groups_that_enable_pipelines_in_declared_order(self):
        metas = [
            _meta(name="speech_recognition", extras=("audio",)),
            _meta(name="speaker_diarization", extras=("audio",)),
            _meta(name="scene_detection", extras=("scene",)),
        ]
        groups = readiness.extras_groups(metas)
        names = [g["name"] for g in groups]
        assert set(names) == {"audio", "scene"}
        assert names.index("audio") < names.index("scene")  # pyproject.toml order
        audio = groups[names.index("audio")]
        assert audio["pipelines"] == ["speech_recognition", "speaker_diarization"]
        assert audio["installed"] is True
        assert audio["install_job_id"] is None

    def test_reports_an_in_flight_install(self):
        extras_install._in_flight["scene"] = "job-9"
        [group] = readiness.extras_groups(
            [_meta(name="scene_detection", extras=("scene",))]
        )
        assert group["install_job_id"] == "job-9"

    def test_size_includes_torch_only_when_torch_is_missing(self):
        with patch.object(readiness, "_extra_requires_torch", return_value=True):
            with patch.object(readiness, "_torch_installed", return_value=True):
                with_torch = readiness.approx_download_mb("scene")
            with patch.object(readiness, "_torch_installed", return_value=False):
                without_torch = readiness.approx_download_mb("scene")
        assert without_torch == with_torch + readiness._TORCH_MB

    def test_unknown_group_has_no_size(self):
        assert readiness.approx_download_mb("not-a-group") is None
