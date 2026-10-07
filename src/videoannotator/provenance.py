"""What made an output file, recorded with it (spec 017).

A record names the pipeline, VideoAnnotator's version, each model and its
weight revision, the effective settings (secrets redacted), the numerical
settings in force, when the file was made, the job and the input video. The
job runner builds one per pipeline run and stamps it into every file the
pipeline wrote, so the record travels with a file that leaves the job:

- JSON (COCO and the others): a top-level ``provenance`` key.
- WebVTT: a ``NOTE videoannotator-provenance <json>`` block after the header.
- RTTM: a companion ``<file>.provenance.json`` (no comment syntax every RTTM
  loader skips).

Contract: specs/017-output-provenance/contracts/provenance-in-files.md.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .version import __version__

SCHEMA_VERSION = 1
VTT_MARKER = "NOTE videoannotator-provenance "
COMPANION_SUFFIX = ".provenance.json"
REDACTED = "<redacted>"

_SECRET_KEY = re.compile(
    r"token|secret|password|passwd|api_?key|(^|_)auth($|_)", re.IGNORECASE
)


@dataclass
class ModelRef:
    """One model a pipeline used, and which weights exactly."""

    name: str
    source: str
    revision: str | None
    revision_kind: str  # sha256 | git | ollama-digest | unknown
    revision_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """As stored in a record; `revision_note` only when there is one."""
        data = asdict(self)
        if data["revision_note"] is None:
            del data["revision_note"]
        return data


def weights_ref(name: str, source: str, path: str | Path | None) -> ModelRef:
    """A ModelRef identified by the sha256 of its weights file."""
    if path is None or not Path(path).is_file():
        return ModelRef(name, source, None, "unknown", "weights file not found")
    return ModelRef(name, source, file_sha256(path), "sha256")


def build_record(
    pipeline_name: str,
    *,
    models: list[ModelRef] | None = None,
    settings: dict[str, Any] | None = None,
    determinism: dict[str, Any] | None = None,
    job_id: str | None = None,
    input_name: str | None = None,
    input_sha256: str | None = None,
    vlm: dict[str, Any] | None = None,
    sub_pipeline: str | None = None,
) -> dict[str, Any]:
    """A provenance record (schema version 1); see data-model.md."""
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": {"name": pipeline_name, "sub_pipeline": sub_pipeline},
        "videoannotator_version": __version__,
        "models": [m.to_dict() for m in models or []],
        "settings": redact(settings or {}),
        "determinism": determinism or {},
        "created_at": datetime.now(UTC).isoformat(),
        "job_id": job_id,
        "input": {"name": input_name, "sha256": input_sha256},
    }
    if vlm is not None:
        record["vlm"] = vlm
    return record


def redact(value: Any) -> Any:
    """`value` with every secret-looking key's value replaced, at any depth."""
    if isinstance(value, dict):
        return {
            # Strings only: a flag like use_auth_token=True is a setting, not a secret.
            k: REDACTED
            if isinstance(k, str) and _SECRET_KEY.search(k) and isinstance(v, str) and v
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


_hash_cache: dict[tuple[str, int, float], str] = {}


def file_sha256(path: str | Path) -> str:
    """sha256 of a file, cached per process by (path, size, mtime)."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime)
    if key not in _hash_cache:
        digest = hashlib.sha256()
        with open(resolved, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        _hash_cache[key] = digest.hexdigest()
    return _hash_cache[key]


def companion_path(path: str | Path) -> Path:
    """Where the record of a file that can't hold one is written."""
    path = Path(path)
    return path.with_name(path.name + COMPANION_SUFFIX)


def stamp_file(path: str | Path, record: dict[str, Any]) -> None:
    """Record `record` in or beside the output file at `path`."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            companion_path(path).write_text(json.dumps(record, indent=2))
            return
        data["provenance"] = record
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    elif suffix == ".vtt":
        _stamp_webvtt(path, record)
    else:
        companion_path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")


def _stamp_webvtt(path: Path, record: dict[str, Any]) -> None:
    # A NOTE ends at a blank line and may not contain "-->": one line of JSON,
    # with "-->" written as an escape JSON readers decode back.
    note = VTT_MARKER + json.dumps(record, separators=(",", ":")).replace(
        "-->", "--\\u003e"
    )
    blocks = path.read_text(encoding="utf-8").split("\n\n")
    blocks = [b for b in blocks if not b.startswith(VTT_MARKER)]
    blocks.insert(1, note)
    path.write_text("\n\n".join(blocks), encoding="utf-8")


def read_record(path: str | Path) -> dict[str, Any] | None:
    """The record stamped in or beside `path`, or None."""
    path = Path(path)
    companion = companion_path(path)
    if companion.is_file():
        return json.loads(companion.read_text(encoding="utf-8"))
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("provenance") if isinstance(data, dict) else None
    if suffix == ".vtt":
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            if block.startswith(VTT_MARKER):
                return json.loads(block[len(VTT_MARKER) :])
    return None


def hub_ref(repo_id: str, source: str = "huggingface") -> ModelRef:
    """A Hugging Face Hub model, identified by the commit of its cached snapshot."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, "config.yaml")
        if not isinstance(cached, str):
            cached = try_to_load_from_cache(repo_id, "config.json")
    except Exception as e:  # huggingface_hub missing or cache unreadable
        return ModelRef(repo_id, source, None, "unknown", f"cache not readable: {e}")
    if not isinstance(cached, str):
        return ModelRef(repo_id, source, None, "unknown", "not found in the Hub cache")
    # .../models--org--name/snapshots/<commit>/config.yaml
    return ModelRef(repo_id, source, Path(cached).parent.name, "git")


def whisper_ref(whisper_module: Any, size: str) -> ModelRef:
    """An openai-whisper checkpoint. Its download URL embeds the checkpoint's
    sha256, which whisper verifies on load, so no hashing is needed."""
    models = getattr(whisper_module, "_MODELS", None)
    if models is None:  # the pipeline passes a patchable wrapper, not the module
        try:
            import whisper as openai_whisper

            models = openai_whisper._MODELS
        except Exception:
            models = {}
    url = models.get(size)
    if not url:
        return ModelRef(
            f"whisper {size}", "openai-whisper", None, "unknown", "not a known size"
        )
    return ModelRef(f"whisper {size}", "openai-whisper", url.split("/")[-2], "sha256")


def clip_ref(model: str, pretrained: str) -> ModelRef:
    """An open_clip model and pretrained tag, identified by its checkpoint."""
    name = f"{model} ({pretrained})"
    try:
        import open_clip

        cfg = open_clip.get_pretrained_cfg(model, pretrained)
        path = open_clip.download_pretrained(cfg)  # cached: already loaded
    except Exception as e:
        return ModelRef(
            name, "open_clip", None, "unknown", f"checkpoint not found: {e}"
        )
    return weights_ref(name, "open_clip", path)
