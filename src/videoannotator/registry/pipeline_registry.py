"""Pipeline Registry - Minimal v1.2.1 Implementation.

Single Source of Truth (SSOT) for pipeline metadata.

Design Goals (v1.2.1 slice):
- Lightweight YAML-backed metadata loading.
- Basic validation (required fields, type checks for simple primitives).
- Graceful degradation: missing/invalid files logged as warnings, not fatal.
- Extensibility: reserved placeholders for future (v1.3.0) capabilities & resources.

NOT in v1.2.1 scope:
- Dynamic capability inference
- Resource scheduling logic
- Plugin sandbox / loading
- Multi-modal correlation metadata

Future fields (documented only): resources, capabilities, modalities, supports_streaming.
"""

from __future__ import annotations

import builtins
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger("videoannotator.pipelines")

# Directory containing YAML metadata files (relative to project root)
DEFAULT_METADATA_DIR = Path(__file__).parent / "metadata"

REQUIRED_TOP_LEVEL_FIELDS = [
    "name",
    "display_name",
    "description",
    "outputs",
    "config_schema",
    "version",
]
VALID_OUTPUT_FORMATS = {"COCO", "RTTM", "WebVTT", "JSON"}


@dataclass
class PipelineOutputFormat:
    format: str
    types: list[str] = field(default_factory=list)


@dataclass
class PipelineConfigField:
    type: str
    default: Any = None
    description: str = ""


@dataclass
class PipelineMetadata:
    """Structured metadata describing a single pipeline."""

    name: str
    display_name: str
    description: str
    outputs: list[PipelineOutputFormat]
    config_schema: dict[str, PipelineConfigField]
    version: int
    # Extended taxonomy fields (all optional)
    pipeline_family: str | None = None
    variant: str | None = None
    tasks: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    backends: list[str] = field(default_factory=list)
    stability: str | None = None
    examples: list[dict[str, Any]] = field(default_factory=list)


class PipelineRegistry:
    """Minimal registry to load & provide pipeline metadata."""

    def __init__(self, metadata_dir: Path | None = None):
        """Initialize the registry and set the metadata directory."""
        self.metadata_dir = metadata_dir or DEFAULT_METADATA_DIR
        self._pipelines: dict[str, PipelineMetadata] = {}
        self._loaded = False

    def load(self, force: bool = False) -> None:
        """Load metadata files from disk, optionally forcing a reload."""
        if self._loaded and not force:
            return
        self._pipelines.clear()
        if not self.metadata_dir.exists():
            LOGGER.warning(
                "Pipeline metadata directory not found: %s", self.metadata_dir
            )
            self._loaded = True
            return
        for path in sorted(self.metadata_dir.glob("*.yaml")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                meta = self._parse_metadata(raw, source=path)
                if meta:
                    if meta.name in self._pipelines:
                        LOGGER.warning(
                            "Duplicate pipeline name '%s' in %s (ignoring)",
                            meta.name,
                            path,
                        )
                    else:
                        self._pipelines[meta.name] = meta
            except Exception as e:  # broad by design: metadata issues should not crash
                LOGGER.warning("Failed to load pipeline metadata %s: %s", path, e)
        self._loaded = True
        LOGGER.info("Loaded %d pipeline metadata file(s)", len(self._pipelines))

    def _parse_metadata(
        self, raw: dict[str, Any], source: Path
    ) -> PipelineMetadata | None:
        missing = [f for f in REQUIRED_TOP_LEVEL_FIELDS if f not in raw]
        if missing:
            LOGGER.warning(
                "Metadata %s missing required fields: %s",
                source.name,
                ", ".join(missing),
            )
            return None
        outputs: list[PipelineOutputFormat] = []
        for o in raw.get("outputs", []):
            fmt = o.get("format")
            if fmt not in VALID_OUTPUT_FORMATS:
                LOGGER.warning(
                    "Metadata %s has unsupported output format '%s'", source.name, fmt
                )
                continue
            outputs.append(
                PipelineOutputFormat(format=fmt, types=o.get("types", []) or [])
            )
        if not outputs:
            LOGGER.warning("Metadata %s has no valid outputs; skipping", source.name)
            return None
        config_schema: dict[str, PipelineConfigField] = {}
        raw_schema = raw.get("config_schema", {})
        if not isinstance(raw_schema, dict):
            LOGGER.warning("Metadata %s config_schema is not a dict", source.name)
            return None
        for key, val in raw_schema.items():
            if not isinstance(val, dict) or "type" not in val:
                LOGGER.warning(
                    "Metadata %s field '%s' invalid config entry", source.name, key
                )
                continue
            config_schema[key] = PipelineConfigField(
                type=str(val.get("type")),
                default=val.get("default"),
                description=str(val.get("description", "")),
            )
        if not config_schema:
            LOGGER.warning("Metadata %s has empty config_schema", source.name)
        examples = raw.get("examples") or []
        if not isinstance(examples, list):
            LOGGER.warning("Metadata %s examples not a list", source.name)
            examples = []

        # Optional extended fields (ignored if wrong types)
        def _list_field(key: str) -> list[str]:
            val = raw.get(key)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                return [x.strip() for x in val if x and isinstance(x, str)]
            return []

        return PipelineMetadata(
            name=str(raw["name"]),
            display_name=str(raw["display_name"]),
            description=str(raw["description"]),
            outputs=outputs,
            config_schema=config_schema,
            version=int(raw["version"]),
            pipeline_family=str(raw.get("pipeline_family"))
            if raw.get("pipeline_family")
            else None,
            variant=str(raw.get("variant")) if raw.get("variant") else None,
            tasks=_list_field("tasks"),
            modalities=_list_field("modalities"),
            capabilities=_list_field("capabilities"),
            backends=_list_field("backends"),
            stability=str(raw.get("stability")) if raw.get("stability") else None,
            examples=examples,
        )

    def list(self) -> builtins.list[PipelineMetadata]:
        """Return all loaded pipeline metadata entries."""
        if not self._loaded:
            self.load()
        return list(self._pipelines.values())

    def get(self, name: str) -> PipelineMetadata | None:
        """Return the metadata for a pipeline by name if available."""
        if not self._loaded:
            self.load()
        return self._pipelines.get(name)


# Singleton-style accessor
_registry_instance: PipelineRegistry | None = None


def get_registry() -> PipelineRegistry:
    """Return the shared PipelineRegistry singleton instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = PipelineRegistry()
    return _registry_instance


__all__ = [
    "PipelineMetadata",
    "PipelineRegistry",
    "get_registry",
]
