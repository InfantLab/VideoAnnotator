"""A methods paragraph, with BibTeX, from what made the results (spec 025).

Built only from the provenance records stamped in the output files (spec 017)
and each pipeline's metadata (its plain description, default settings and
citations). Nothing is filled in from the running install: a pipeline whose
files carry no record is described as "model version not recorded".

The text names no video, path, job id or server address (FR-029): video
names often carry participant codes, so it counts videos instead.

Contract: specs/025-results-out/contracts/elan-and-methods.md.
"""

from __future__ import annotations

import ipaddress
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..registry.pipeline_registry import PipelineMetadata, get_registry
from .citation import KEY as VIDEOANNOTATOR_KEY
from .citation import videoannotator_bibtex
from .readers import NOT_RECORDED
from .results_set import ResultsSet

# What each shipped pipeline does, in a methods section's words.
PHRASES = {
    "person_tracking": "People were detected, their body pose estimated and tracked "
    "across frames",
    # Age and gender only when switched on, which the settings list then says.
    "face_analysis": "Faces were detected and their emotional expression estimated",
    "face_openface3_embedding": "Faces were detected and facial action units, head "
    "pose and gaze estimated",
    "scene_detection": "Scene changes were detected and each scene classified",
    "speaker_diarization": "Speaker turns were identified",
    "speech_recognition": "Speech was transcribed",
    "audio_processing": "Speech was transcribed and speaker turns identified",
    "vlm_annotation": "Sampled frames were labelled by a vision-language model",
}

# Settings that change where or how fast a pipeline runs, not what it finds.
_NOT_RESULT_KEYS = {
    "device",
    "cache_dir",
    "model_path",
    "base_url",
    "use_auth_token",
    "batch_size",
    "keep_alive",
    "request_timeout_sec",
    "verbose",
    "prompt",  # identified by its fingerprint, with the full text alongside
}
_NOT_RESULT_PATTERN = re.compile(r"(_dir|_path|_url|timeout|_token)$")

OLLAMA_KEY = "ollama"

_REVISION_WORDS = {"sha256": "sha256", "git": "revision", "ollama-digest": "digest"}


@dataclass
class MethodsDescription:
    prose: str
    bibtex: str
    revisions: list[str] = field(default_factory=list)
    prompts: list[dict[str, str]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prose": self.prose,
            "bibtex": self.bibtex,
            "revisions": self.revisions,
            "prompts": self.prompts,
            "summary": self.summary,
        }

    def to_markdown(self) -> str:
        parts = ["# Methods", "", self.prose, ""]
        if self.revisions:
            parts += ["## Model revisions", ""]
            parts += [f"- {r}" for r in self.revisions]
            parts.append("")
        parts += ["## References (BibTeX)", "", "```bibtex", self.bibtex, "```", ""]
        for prompt in self.prompts:
            parts += [
                f"## Prompt {prompt['sha256'][:12]}",
                "",
                f"sha256 `{prompt['sha256']}`",
                "",
                "```text",
                prompt["text"],
                "```",
                "",
            ]
        return "\n".join(parts)


@dataclass(frozen=True)
class _Model:
    name: str
    source: str
    revision: str | None
    kind: str
    quantization: str | None = None


@dataclass(frozen=True)
class _Use:
    """How one pipeline ran on one video: the parts that decide its results."""

    pipeline: str
    recorded: bool
    version: str | None
    models: tuple[_Model, ...]
    settings: tuple[tuple[str, str], ...]
    deterministic: bool | None
    prompt_sha256: str | None
    quantization: str | None
    where: str | None = None
    # A VLM run's model server kind (spec 033): Ollama is cited only when it ran.
    backend: str | None = None


# Where a model ran, without naming the server (FR-029).
_THIS_COMPUTER_HOSTS = {"localhost", "host.docker.internal"}


def _where(vlm: dict[str, Any]) -> str | None:
    cls = vlm.get("endpoint_class")
    if cls == "remote":
        return "on a remote server"
    if cls != "local":
        return None
    host = str(vlm.get("endpoint_host") or "").strip("[]").lower()
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host in _THIS_COMPUTER_HOSTS
    return "on this computer" if loopback else "on the local network"


def _short_name(model: _Model) -> str:
    name = model.name
    # A local weights file ("models/yolo/yolo11n-pose.pt"), not a hub id
    # ("pyannote/speaker-diarization-3.1"); its directory is this machine's.
    if "/" in name and Path(name).suffix and model.source != "huggingface":
        return Path(name).name
    return name


def _model_text(model: _Model) -> str:
    if not model.revision:
        text = f"{_short_name(model)} (revision not recorded)"
    else:
        word = _REVISION_WORDS.get(model.kind, "revision")
        text = f"{_short_name(model)} ({word} {model.revision[:12]})"
    return f"{text} at {model.quantization}" if model.quantization else text


def _setting_value(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _changed_settings(
    settings: dict[str, Any], meta: PipelineMetadata | None
) -> tuple[tuple[str, str], ...]:
    if meta is None:
        return ()
    changed = []
    for key, value in settings.items():
        spec = meta.config_schema.get(key)
        if spec is None or key in _NOT_RESULT_KEYS or _NOT_RESULT_PATTERN.search(key):
            continue
        if value != spec.default:
            changed.append((key, _setting_value(value)))
    return tuple(sorted(changed))


def _use(
    pipeline: str, record: dict[str, Any] | None, meta: PipelineMetadata | None
) -> _Use:
    if not record:
        return _Use(pipeline, False, None, (), (), None, None, None)
    models = tuple(
        _Model(
            str(m.get("name")),
            str(m.get("source") or ""),
            m.get("revision"),
            str(m.get("revision_kind") or "unknown"),
            m.get("quantization"),
        )
        for m in record.get("models") or []
        if isinstance(m, dict) and m.get("name")
    )
    determinism = record.get("determinism") or {}
    vlm = record.get("vlm") or {}
    raw_settings = record.get("settings")
    settings: dict[str, Any] = raw_settings if isinstance(raw_settings, dict) else {}
    return _Use(
        pipeline=pipeline,
        recorded=True,
        version=record.get("videoannotator_version"),
        models=models,
        settings=_changed_settings(settings, meta),
        deterministic=determinism.get("deterministic") if determinism else None,
        prompt_sha256=vlm.get("prompt_sha256"),
        quantization=vlm.get("quantization"),
        where=_where(vlm),
        backend=vlm.get("backend"),
    )


def _prompt_text(record: dict[str, Any] | None, path: Path) -> str | None:
    settings = (record or {}).get("settings") or {}
    if isinstance(settings, dict) and isinstance(settings.get("prompt"), str):
        return settings["prompt"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for annotation in data.get("annotations") or []:
            if isinstance(annotation, dict) and annotation.get("prompt"):
                return str(annotation["prompt"])
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _cite(keys: list[str]) -> str:
    return (
        f" [{'; '.join('@' + k for k in keys)}]" if keys else " (no citation recorded)"
    )


class _Writer:
    def __init__(self) -> None:
        self.registry = get_registry()
        self.cited: list[str] = []
        self.entries: dict[str, str] = {}

    def meta(self, pipeline: str) -> PipelineMetadata | None:
        return self.registry.get(pipeline)

    def cite(self, pipeline: str, backend: str | None = None) -> str:
        meta = self.meta(pipeline)
        keys = []
        for c in meta.citations if meta else []:
            if c.key == OLLAMA_KEY and backend not in (None, "ollama"):
                continue
            keys.append(c.key)
            if c.key not in self.entries:
                self.entries[c.key] = c.bibtex
                self.cited.append(c.key)
        return _cite(keys)

    def phrase(self, pipeline: str) -> str:
        meta = self.meta(pipeline)
        if pipeline in PHRASES:
            return PHRASES[pipeline]
        return f"{meta.display_name if meta else pipeline} was run"

    def sentence(self, use: _Use) -> str:
        text = self.phrase(use.pipeline)
        if not use.recorded:
            return f"{text} (model version not recorded){self.cite(use.pipeline, use.backend)}."
        if use.models:
            text += " with " + ", ".join(_model_text(m) for m in use.models)
        if use.quantization:
            text += f", quantised {use.quantization}"
        if use.where:
            text += f", running {use.where}"
        if use.prompt_sha256:
            text += f", using prompt {use.prompt_sha256[:12]} (full text below)"
        if use.settings:
            text += "; settings: " + ", ".join(f"{k} = {v}" for k, v in use.settings)
        return f"{text}{self.cite(use.pipeline, use.backend)}."


def _version_text(versions: set[str | None]) -> str:
    known = sorted(v for v in versions if v)
    if not known:
        return "VideoAnnotator (version not recorded)"
    return "VideoAnnotator " + " and ".join(known)


def _determinism_text(uses: list[_Use]) -> str | None:
    values = {
        u.deterministic for u in uses if u.recorded and u.deterministic is not None
    }
    if values == {True}:
        return "GPU deterministic mode was on."
    if values == {False}:
        return "GPU deterministic mode was off."
    return None


def describe(results: ResultsSet) -> MethodsDescription:
    """The methods description of everything in `results`."""
    writer = _Writer()
    prompts: dict[str, str] = {}
    videos: list[tuple[_Use, ...]] = []
    not_recorded: set[str] = set()

    for video in results.videos:
        by_pipeline: dict[str, _Use] = {}
        for result in video.files:
            pipeline = result.pipeline
            use = _use(pipeline, result.provenance, writer.meta(pipeline))
            # A pipeline with several files (audio_processing) is one use; a
            # recorded file wins over an unrecorded one.
            if pipeline not in by_pipeline or (
                use.recorded and not by_pipeline[pipeline].recorded
            ):
                by_pipeline[pipeline] = use
            if not use.recorded:
                not_recorded.add(pipeline)
            if use.prompt_sha256 and use.prompt_sha256 not in prompts:
                text = _prompt_text(result.provenance, result.path)
                if text is not None:
                    prompts[use.prompt_sha256] = text
        if not video.files:
            continue
        if by_pipeline:
            videos.append(tuple(by_pipeline[p] for p in sorted(by_pipeline)))
    if not videos:
        raise ValueError("no results to describe")

    groups = Counter(videos).most_common()
    main, main_count = groups[0]
    total = len(videos)
    versions = {u.version for u in main if u.recorded}
    intro = (
        f"{'The video was' if total == 1 else f'All {total} videos were'}"
        if len(groups) == 1
        else f"Of {total} videos, {main_count} {'was' if main_count == 1 else 'were'}"
    )
    sentences = [
        f"{intro} annotated automatically with {_version_text(versions)} "
        f"[@{VIDEOANNOTATOR_KEY}]."
    ]
    sentences += [writer.sentence(u) for u in main]
    determinism = _determinism_text(list(main))
    if determinism:
        sentences.append(determinism)
    if any(u.settings for u in main) or any(u.recorded for u in main):
        sentences.append("Settings not listed were the pipelines' defaults.")

    main_by_pipeline = {u.pipeline: u for u in main}
    for uses, count in groups[1:]:
        clauses = []
        theirs = {u.pipeline: u for u in uses}
        other_versions = {u.version for u in uses if u.recorded}
        if other_versions != versions:
            clauses.append(f"{_version_text(other_versions)} was used")
        for pipeline in sorted(set(theirs) | set(main_by_pipeline)):
            if pipeline not in theirs:
                clauses.append(f"{pipeline} was not run")
            elif theirs[pipeline] != main_by_pipeline.get(pipeline):
                described = writer.sentence(theirs[pipeline]).rstrip(".")
                clauses.append(described[0].lower() + described[1:])
        which = (
            ("the other video" if count == 1 else f"the other {count} videos")
            if len(groups) == 2
            else ("another video" if count == 1 else f"another {count} videos")
        )
        sentences.append(f"For {which}, " + "; ".join(clauses) + ".")

    revisions = []
    seen: set[_Model] = set()
    for uses in dict.fromkeys(videos):
        for use in uses:
            for model in use.models:
                if model not in seen:
                    seen.add(model)
                    revision = (
                        f"{model.kind} {model.revision}"
                        if model.revision
                        else NOT_RECORDED
                    )
                    revisions.append(
                        f"{model.name} ({model.source or 'unknown source'}): {revision}"
                    )

    main_version = sorted(v for v in versions if v)
    bibtex = [videoannotator_bibtex(main_version[-1] if main_version else None)]
    bibtex += [writer.entries[k] for k in writer.cited]
    return MethodsDescription(
        prose=" ".join(sentences),
        bibtex="\n\n".join(bibtex),
        revisions=revisions,
        prompts=[{"sha256": sha, "text": text} for sha, text in prompts.items()],
        summary={
            "videos": total,
            "groups": len(groups),
            "not_recorded": sorted(not_recorded),
        },
    )
