"""Dynamic pipeline class loader using registry metadata.

This module provides centralized pipeline class loading to eliminate
hardcoded pipeline mappings in JobProcessor and BatchOrchestrator.
"""

import functools
import importlib
import importlib.metadata
import logging

from packaging.requirements import Requirement

from .pipeline_registry import PipelineMetadata, get_registry

LOGGER = logging.getLogger("videoannotator.registry")

_DISTRIBUTION_NAME = "videoannotator"


@functools.cache
def _packages_for_extra(extra: str) -> tuple[str, ...]:
    """Return the pip distribution names declared under `[extra]`.

    Read from this package's own installed metadata (Requires-Dist entries
    with an `extra == "<name>"` marker) rather than a hand-maintained
    mapping, so it can't drift from `pyproject.toml`
    (specs/004-extras-based-install/research.md §3).
    """
    try:
        dist = importlib.metadata.distribution(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return ()
    packages: list[str] = []
    for req_str in dist.requires or []:
        req = Requirement(req_str)
        if req.marker and req.marker.evaluate({"extra": extra}):
            packages.append(req.name)
    return tuple(packages)


@functools.cache
def _is_distribution_installed(name: str) -> bool:
    try:
        importlib.metadata.distribution(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def extras_available(requires_extras: list[str]) -> bool:
    """Return whether every package declared under `requires_extras` is
    installed. An empty list is vacuously available (research.md §2) — it
    means "no extra needed", not "unknown"."""
    for extra in requires_extras:
        for package in _packages_for_extra(extra):
            if not _is_distribution_installed(package):
                return False
    return True


def missing_extras(requires_extras: list[str]) -> list[str]:
    """Return the subset of `requires_extras` that have at least one
    package not currently installed."""
    return [extra for extra in requires_extras if not extras_available([extra])]


def install_hint(requires_extras: list[str]) -> str:
    """Return the exact `pip install videoannotator[...]` command for the
    given extras groups (contracts/unavailable-pipeline-error.md)."""
    groups = ",".join(requires_extras) if requires_extras else ""
    return f"pip install {_DISTRIBUTION_NAME}[{groups}]"


# Pipelines that were installed by default in v1.4.4 but were demoted to a
# non-default extras group in v1.5.0 (data-model.md's migration message
# record). Requesting one of these without its extras gets a distinct
# "no longer installed by default" message instead of the generic
# unavailable-pipeline text (research.md §4).
_V144_DEMOTED_PIPELINES: dict[str, str] = {
    "face_laion_clip": "face-laion",
    "laion_voice": "audio-laion",
    "face_openface3_embedding": "face-openface3",
}


def migration_note(pipeline_name: str) -> str | None:
    """Return the v1.4.4->v1.5.0 migration note for a pipeline demoted out
    of the default install, or None if it was never in the default install.
    """
    extra = _V144_DEMOTED_PIPELINES.get(pipeline_name)
    if extra is None:
        return None
    return (
        f"As of v1.5.0, pipelines requiring the '{extra}' extras group are "
        "no longer installed by default."
    )


class PipelineLoader:
    """Load pipeline classes dynamically from registry metadata."""

    def __init__(self):
        """Initialize the pipeline loader."""
        self._class_cache: dict[str, type] = {}
        self._registry = get_registry()

    def load_all_pipelines(self) -> dict[str, type]:
        """Load all available pipeline classes from registry.

        Returns:
            Dictionary mapping pipeline names to their classes.
            Includes both primary names and any aliases.
        """
        self._registry.load()
        pipeline_classes = {}
        # (stability_rank, name, class) candidates per family, used to pick which
        # pipeline the short family alias (e.g. 'face') should resolve to.
        stability_rank = {"stable": 0, "beta": 1, "experimental": 2}
        family_candidates: dict[str, list[tuple[int, str, type]]] = {}

        for meta in self._registry.list():
            pipeline_class = self._load_pipeline_class(meta)
            if pipeline_class:
                # Add primary name
                pipeline_classes[meta.name] = pipeline_class

                if meta.pipeline_family:
                    family_candidates.setdefault(meta.pipeline_family, []).append(
                        (
                            stability_rank.get(meta.stability or "", 3),
                            meta.name,
                            pipeline_class,
                        )
                    )

                    # Add variant-based alias if applicable
                    if meta.variant:
                        variant_name = f"{meta.pipeline_family}_{meta.variant}"
                        if variant_name != meta.name:
                            pipeline_classes[variant_name] = pipeline_class

        # Assign each family's short alias (e.g. 'face') to its most-stable
        # loadable pipeline, rather than whichever happened to load first —
        # otherwise an experimental variant can silently shadow a stable one
        # just because metadata files are read in alphabetical order.
        for family, candidates in family_candidates.items():
            candidates.sort(key=lambda c: c[0])
            best_rank, best_name, best_class = candidates[0]
            pipeline_classes[family] = best_class
            if len(candidates) > 1:
                LOGGER.debug(
                    f"Family alias '{family}' -> '{best_name}' "
                    f"(candidates: {[c[1] for c in candidates]})"
                )

        LOGGER.info(
            f"Loaded {len(set(pipeline_classes.values()))} unique pipeline classes "
            f"with {len(pipeline_classes)} total names/aliases"
        )

        return pipeline_classes

    def _load_pipeline_class(self, meta: PipelineMetadata) -> type | None:
        """Load a single pipeline class from metadata.

        Args:
            meta: Pipeline metadata containing module path

        Returns:
            Pipeline class or None if loading fails
        """
        # Check cache first
        if meta.name in self._class_cache:
            return self._class_cache[meta.name]

        module_path = meta.module_path
        if not module_path:
            # The registry itself already skips YAML files missing
            # module_path (pipeline_registry.py._parse_metadata), so this is
            # defence-in-depth rather than the expected path.
            LOGGER.warning(
                f"Cannot load pipeline '{meta.name}': no module_path in metadata"
            )
            return None

        if not extras_available(meta.requires_extras):
            LOGGER.info(
                f"Skipping pipeline '{meta.name}': requires extras "
                f"{meta.requires_extras} not installed "
                f"({install_hint(meta.requires_extras)})"
            )
            return None

        try:
            # Parse module_path format: "module.path:ClassName"
            if ":" not in module_path:
                LOGGER.error(
                    f"Invalid module_path format for '{meta.name}': {module_path}. "
                    "Expected 'module.path:ClassName'"
                )
                return None

            module_name, class_name = module_path.split(":", 1)

            # Dynamically import the module
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)

            # Cache the result
            self._class_cache[meta.name] = pipeline_class

            LOGGER.debug(f"Loaded pipeline class: {meta.name} -> {pipeline_class}")
            return pipeline_class

        except ImportError as e:
            LOGGER.warning(
                f"Failed to import pipeline '{meta.name}' from {module_path}: {e}"
            )
            return None
        except AttributeError as e:
            LOGGER.error(f"Pipeline class not found in module for '{meta.name}': {e}")
            return None
        except Exception as e:
            LOGGER.error(
                f"Unexpected error loading pipeline '{meta.name}': {e}",
                exc_info=True,
            )
            return None


# Singleton instance
_loader_instance: PipelineLoader | None = None


def get_pipeline_loader() -> PipelineLoader:
    """Get the shared PipelineLoader singleton instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = PipelineLoader()
    return _loader_instance


__all__ = [
    "PipelineLoader",
    "get_pipeline_loader",
    "extras_available",
    "missing_extras",
    "install_hint",
    "migration_note",
]
