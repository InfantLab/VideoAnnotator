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
    name="HUGGINGFACE_TOKEN",
    aliases=["HF_AUTH_TOKEN", "HF_TOKEN"],
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


_REAL_HF_CACHED = readiness._hf_cached


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("HUGGINGFACE_TOKEN", "HF_AUTH_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    extras_install._in_flight.clear()
    extras_install._restart_pending_extras.clear()
    pipeline_loader.clear_import_errors()
    readiness._ollama_cache.clear()
    readiness._ollama_refreshing.clear()
    readiness._hf_cache.clear()
    readiness._hf_refreshing.clear()
    # Never reach the real Hub: by default it looks unreachable.
    with (
        patch.object(readiness, "extras_available", return_value=True),
        patch.object(readiness, "_hf_get", return_value=(None, b"")),
        patch.object(readiness, "_hf_cached", return_value=False),
    ):
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
        assert blocker["name"] == "HUGGINGFACE_TOKEN"
        assert "Set HUGGINGFACE_TOKEN" in blocker["message"]
        assert "server's environment" in blocker["message"]
        assert blocker["help_url"] == "https://huggingface.co/settings/tokens"

    def test_secret_value_never_appears(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_sentinel_value")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert "hf_sentinel_value" not in repr(r)

    @pytest.mark.parametrize("legacy_name", ["HF_AUTH_TOKEN", "HF_TOKEN"])
    def test_alias_counts_as_set(self, monkeypatch, legacy_name):
        monkeypatch.setenv(legacy_name, "x")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert r["state"] == "ready"
        assert [n["kind"] for n in r["notes"]] == ["licence"]

    def test_unverifiable_licence_is_a_note(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "x")
        r = readiness.pipeline_readiness(_meta(setup=[HF_SECRET, LICENCE]))
        assert r["state"] == "ready" and r["next_action"] == "none"
        assert r["notes"][0]["help_url"].endswith("speaker-diarization-3.1")
        assert "HUGGINGFACE_TOKEN" in r["notes"][0]["message"]

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


SEGMENTATION = SetupRequirement(
    kind="licence",
    name="pyannote/segmentation-3.0",
    help_url="https://huggingface.co/pyannote/segmentation-3.0",
)


def _hub(whoami=200, accepted=()):
    def get(path, token):
        if path == "/api/whoami-v2":
            return whoami, b'{"name": "someone"}'
        repo = path.removeprefix("/api/models/").removesuffix("/auth-check")
        return (200 if repo in accepted else 403), b""

    return get


class TestHuggingFaceLicences:
    @pytest.fixture(autouse=True)
    def _token(self, monkeypatch):
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_x")

    def _readiness(self):
        return readiness.pipeline_readiness(
            _meta(setup=[HF_SECRET, LICENCE, SEGMENTATION])
        )

    def test_rejected_token_blocks_and_says_so(self):
        with patch.object(readiness, "_hf_get", side_effect=_hub(whoami=401)):
            r = self._readiness()
        assert r["state"] == "needs_setup"
        [blocker] = r["blockers"]
        assert blocker["kind"] == "secret"
        assert "rejects it" in blocker["message"]
        assert r["notes"] == []

    def test_unaccepted_licence_blocks_naming_the_tokens_account(self):
        accepted = {"pyannote/speaker-diarization-3.1"}
        with patch.object(readiness, "_hf_get", side_effect=_hub(accepted=accepted)):
            r = self._readiness()
        assert r["state"] == "needs_setup"
        [blocker] = r["blockers"]
        assert blocker["kind"] == "licence"
        assert blocker["name"] == "pyannote/segmentation-3.0"
        assert "'someone'" in blocker["message"]
        assert "HUGGINGFACE_TOKEN" in blocker["message"]

    def test_accepted_licences_say_nothing(self):
        accepted = {"pyannote/speaker-diarization-3.1", "pyannote/segmentation-3.0"}
        with patch.object(readiness, "_hf_get", side_effect=_hub(accepted=accepted)):
            r = self._readiness()
        assert r["state"] == "ready"
        assert r["blockers"] == [] and r["notes"] == []

    def test_cached_models_skip_the_network(self):
        with (
            patch.object(readiness, "_hf_cached", return_value=True),
            patch.object(readiness, "_hf_get") as get,
        ):
            r = self._readiness()
        get.assert_not_called()
        assert r["state"] == "ready" and r["notes"] == []

    def test_checks_are_cached(self):
        with patch.object(readiness, "_hf_get", side_effect=_hub()) as get:
            self._readiness()
            calls = get.call_count
            self._readiness()
        assert get.call_count == calls

    def test_no_token_keeps_licence_reminders_next_to_the_secret_blocker(
        self, monkeypatch
    ):
        monkeypatch.delenv("HUGGINGFACE_TOKEN")
        with patch.object(readiness, "_hf_get") as get:
            r = self._readiness()
        get.assert_not_called()
        assert [b["kind"] for b in r["blockers"]] == ["secret"]
        assert [n["kind"] for n in r["notes"]] == ["licence", "licence"]


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


class TestWeightCacheLocations:
    def test_pyannote_models_found_in_pyannotes_own_cache(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PYANNOTE_CACHE", str(tmp_path))
        snapshot = (
            tmp_path / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "abc"
        )
        snapshot.mkdir(parents=True)
        with patch.object(readiness, "_hf_cached", _REAL_HF_CACHED):
            assert readiness._hf_cached("pyannote/speaker-diarization-3.1")
            assert not readiness._hf_cached("pyannote/segmentation-3.0")

    def test_whisper_found_in_the_pipelines_models_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        monkeypatch.chdir(tmp_path)
        assert not readiness._whisper_cached("base")
        (tmp_path / "models" / "whisper").mkdir(parents=True)
        (tmp_path / "models" / "whisper" / "base.pt").write_bytes(b"")
        assert readiness._whisper_cached("base")


class TestDeepFaceWeights:
    def test_deepface_weights_checked_in_deepface_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DEEPFACE_HOME", str(tmp_path))
        weight = WeightSpec(id="age_model_weights.h5", approx_mb=540, cache="deepface")
        assert not readiness._weights_cached(weight)

        weights_dir = tmp_path / ".deepface" / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "age_model_weights.h5").write_bytes(b"")
        assert readiness._weights_cached(weight)

    def test_face_analysis_declares_its_deepface_weights(self):
        from videoannotator.registry.pipeline_registry import get_registry

        registry = get_registry()
        registry.load()
        meta = next(m for m in registry.list() if m.name == "face_analysis")
        assert {w.cache for w in meta.weights} == {"deepface"}
        assert sum(w.approx_mb for w in meta.weights) > 1000


class TestOllamaBaseUrl:
    """The e2e run: Ollama running on the host, the server in a devcontainer
    checking the hardcoded 127.0.0.1 -- the container's own loopback."""

    def test_empty_schema_default_uses_server_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        meta = _meta(
            name="vlm_annotation",
            config={"base_url": PipelineConfigField(type="string", default="")},
        )
        assert readiness._ollama_base_url(meta) == "http://host.docker.internal:11434"

    def test_unset_env_falls_back_to_loopback(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        meta = _meta(name="vlm_annotation", config={})
        assert readiness._ollama_base_url(meta) == "http://127.0.0.1:11434"

    def test_shipped_metadata_defers_to_server_default(self):
        from videoannotator.registry.pipeline_registry import get_registry

        registry = get_registry()
        registry.load()
        meta = next(m for m in registry.list() if m.name == "vlm_annotation")
        assert not meta.config_schema["base_url"].default

    def test_unreachable_loopback_in_container_points_at_host(self):
        from videoannotator.diagnostics import ollama

        with patch.object(ollama, "_in_container", return_value=True):
            hint = ollama.unreachable_hint("http://127.0.0.1:11434")
        assert "OLLAMA_BASE_URL=http://host.docker.internal:11434" in hint

    def test_unreachable_outside_container_says_start_it(self):
        from videoannotator.diagnostics import ollama

        with patch.object(ollama, "_in_container", return_value=False):
            hint = ollama.unreachable_hint("http://127.0.0.1:11434")
        assert hint == "Start it with 'ollama serve'."


class TestHuggingFaceToken:
    @pytest.fixture(autouse=True)
    def _unset(self, monkeypatch):
        for var in ("HUGGINGFACE_TOKEN", "HF_AUTH_TOKEN", "HF_TOKEN"):
            monkeypatch.delenv(var, raising=False)

    def test_primary_name_wins(self, monkeypatch):
        from videoannotator.config_env import huggingface_token

        monkeypatch.setenv("HUGGINGFACE_TOKEN", "primary")
        monkeypatch.setenv("HF_AUTH_TOKEN", "legacy")
        assert huggingface_token() == "primary"

    def test_legacy_name_still_read(self, monkeypatch):
        from videoannotator.config_env import huggingface_token

        monkeypatch.setenv("HF_AUTH_TOKEN", "legacy")
        assert huggingface_token() == "legacy"

    def test_unset_or_blank_is_none(self, monkeypatch):
        from videoannotator.config_env import huggingface_token

        monkeypatch.setenv("HUGGINGFACE_TOKEN", "  ")
        assert huggingface_token() is None

    def test_shipped_metadata_names_huggingface_token(self):
        from videoannotator.registry.pipeline_registry import get_registry

        registry = get_registry()
        registry.load()
        for name in ("speaker_diarization", "audio_processing"):
            meta = next(m for m in registry.list() if m.name == name)
            [secret] = [r for r in meta.requires_setup if r.kind == "secret"]
            assert secret.name == "HUGGINGFACE_TOKEN"
            assert "HF_AUTH_TOKEN" in secret.aliases
