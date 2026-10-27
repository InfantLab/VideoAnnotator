"""The settings a new run starts from (spec 029 R5, R6, FR-012).

A project remembers its current settings (`saved_datasets.current_settings`):
set whenever a run starts on it, and when someone edits them. Each user's
last run is remembered too (`user_preferences`, `last_run_settings`). New run
starts from, in order: the chosen workflow, the project's current settings,
the settings of the project's newest run, the caller's last run, the defaults.
Recorded where runs are created, so the CLI and API keep them current too.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

LAST_RUN = "last_run_settings"
ANONYMOUS = "anonymous"


def user_key(user: dict[str, Any] | None) -> str:
    return str(user["id"]) if user and user.get("id") else ANONYMOUS


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def settings(
    selected_pipelines: list[str] | None,
    config: dict[str, Any] | None,
    source: dict[str, Any],
    user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "selected_pipelines": list(selected_pipelines or []),
        "config": dict(config or {}),
        "source": source,
        "updated_at": _now(),
        "updated_by": user_key(user),
    }


def _set_preference(db: Any, key: str, name: str, value: dict[str, Any]) -> None:
    from .database.models import UserPreference

    row = db.get(UserPreference, (key, name))
    if row is None:
        row = UserPreference(user_key=key, name=name, value=value)
        db.add(row)
    else:
        row.value = value
    row.updated_at = datetime.now(UTC).replace(tzinfo=None)


def record_run(
    db: Any,
    *,
    user: dict[str, Any] | None,
    dataset_id: str | None,
    selected_pipelines: list[str] | None,
    config: dict[str, Any] | None,
    run_id: str,
    run_name: str | None,
) -> None:
    """Remember a run's settings as the caller's last, and, when it runs on a
    project, as the project's current settings. Never raises: a run that
    started must not fail because its settings couldn't be remembered."""
    from .database.crud import SavedDatasetCRUD

    source = {"kind": "run", "id": run_id, "name": run_name}
    value = settings(selected_pipelines, config, source, user)
    try:
        _set_preference(db, user_key(user), LAST_RUN, value)
        if dataset_id:
            dataset = SavedDatasetCRUD.get_by_id(db, dataset_id)
            if dataset is not None:
                dataset.current_settings = value
        db.commit()
    except Exception as e:
        logger.warning(f"Could not remember the settings of run {run_id}: {e}")
        db.rollback()


def last_used(db: Any, user: dict[str, Any] | None) -> dict[str, Any] | None:
    from .database.models import UserPreference

    row = db.get(UserPreference, (user_key(user), LAST_RUN))
    return row.value if row is not None and isinstance(row.value, dict) else None


def _newest_run_settings(storage: Any, dataset_id: str) -> dict[str, Any] | None:
    job_ids = storage.list_jobs_by_dataset(dataset_id)
    for job_id in reversed(job_ids):
        try:
            job = storage.load_job_metadata(job_id)
        except Exception:
            continue
        if job is not None and job.selected_pipelines:
            return {
                "selected_pipelines": list(job.selected_pipelines),
                "config": dict(job.config or {}),
                "updated_at": job.created_at.isoformat() if job.created_at else None,
                "source": {
                    "kind": "run",
                    "id": job.batch_id or job.job_id,
                    "name": job.batch_name,
                },
            }
    return None


def prefill(
    db: Any,
    *,
    user: dict[str, Any] | None,
    project_id: str | None = None,
    workflow_id: str | None = None,
    storage: Any | None = None,
) -> dict[str, Any]:
    """`{selected_pipelines, config, source}` for New run, in FR-012's order.

    `source.kind` is `workflow`, `project`, `last_used` or `defaults`. With
    `defaults`, `selected_pipelines` is empty: the viewer selects what it
    offers by default.
    """
    from .database.crud import SavedDatasetCRUD, SavedPipelinePresetCRUD

    if workflow_id:
        preset = SavedPipelinePresetCRUD.get_by_id(db, workflow_id)
        if preset is not None:
            return {
                "selected_pipelines": list(preset.selected_pipelines or []),
                "config": dict(preset.config or {}),
                "source": {
                    "kind": "workflow",
                    "id": str(preset.id),
                    "name": preset.name,
                    "changed_at": _iso(preset.updated_at or preset.created_at),
                },
            }

    dataset = SavedDatasetCRUD.get_by_id(db, project_id) if project_id else None
    if dataset is not None:
        current = dataset.current_settings
        if not isinstance(current, dict) and storage is not None:
            current = _newest_run_settings(storage, str(dataset.id))
        if isinstance(current, dict):
            return {
                "selected_pipelines": list(current.get("selected_pipelines") or []),
                "config": dict(current.get("config") or {}),
                "source": {
                    "kind": "project",
                    "id": str(dataset.id),
                    "name": dataset.name,
                    "changed_at": current.get("updated_at"),
                    "from": current.get("source"),
                },
            }

    last = last_used(db, user)
    if last is not None and dataset is not None:
        # A project with none of its own starts from the defaults, never
        # silently from the last run elsewhere; that is offered (UAT #44, P6).
        return {
            "selected_pipelines": [],
            "config": {},
            "source": {"kind": "defaults"},
            "last_run": _offer(last),
        }
    if last is not None:
        return {
            "selected_pipelines": list(last.get("selected_pipelines") or []),
            "config": dict(last.get("config") or {}),
            "source": {
                "kind": "last_used",
                "name": (last.get("source") or {}).get("name"),
                "changed_at": last.get("updated_at"),
            },
        }

    return {"selected_pipelines": [], "config": {}, "source": {"kind": "defaults"}}


def _offer(last: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_pipelines": list(last.get("selected_pipelines") or []),
        "config": dict(last.get("config") or {}),
        "name": (last.get("source") or {}).get("name"),
        "changed_at": last.get("updated_at"),
    }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
