"""Server-Sent Events endpoints for VideoAnnotator API."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..database import get_storage_backend

router = APIRouter()

logger = logging.getLogger(__name__)

# How often to re-check job state for changes (spec 008, FR-006). Much
# shorter than the heartbeat interval below -- this is what makes the
# stream feel "real-time" without needing a callback/pub-sub hook into
# job_execution.py's executor-thread writes.
_POLL_INTERVAL_SECONDS = 2.0
_HEARTBEAT_INTERVAL_SECONDS = 30.0


async def event_stream() -> AsyncGenerator[str, None]:
    """Generate server-sent events stream.

    Sends `job_status_changed` events whenever a job's (status,
    progress_percentage) changes, alongside the original periodic
    heartbeat (unchanged). Detection is poll-and-diff against storage --
    the single source of truth -- rather than a push/callback from the
    executor thread that runs job pipelines: simpler, can't miss or
    duplicate an event, and matches the spec's own Assumption that the
    stream is additive, never authoritative (a client that never connects
    must still see correct state purely through polling `GET /jobs`/
    `GET /batches/{id}`, which this same storage read backs).
    """
    storage = get_storage_backend()
    # job_id -> (status, progress_percentage) last emitted for that job.
    last_seen: dict[str, tuple[str, float]] = {}
    last_heartbeat = asyncio.get_event_loop().time()

    try:
        yield f"data: {json.dumps({'type': 'connected', 'timestamp': str(asyncio.get_event_loop().time())})}\n\n"

        while True:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

            try:
                for job_id in storage.list_jobs():
                    job = storage.load_job_metadata(job_id)
                    if job is None:
                        continue
                    current = (job.status.value, job.progress_percentage)
                    if last_seen.get(job_id) == current:
                        continue
                    last_seen[job_id] = current
                    event_data = {
                        "type": "job_status_changed",
                        "job_id": job_id,
                        "batch_id": job.batch_id,
                        "status": job.status.value,
                        "progress_percentage": job.progress_percentage,
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
            except Exception as e:
                # A transient storage hiccup shouldn't kill the connection --
                # the client falls back to polling until the next tick
                # (or reconnects), per the stream's additive-not-authoritative
                # design.
                logger.warning(f"Error polling job state for SSE stream: {e}")

            now = asyncio.get_event_loop().time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL_SECONDS:
                last_heartbeat = now
                heartbeat_data = {
                    "type": "heartbeat",
                    "timestamp": str(now),
                    "message": "Server is alive",
                }
                yield f"data: {json.dumps(heartbeat_data)}\n\n"

    except asyncio.CancelledError:
        logger.info("SSE stream cancelled by client")
        return
    except Exception as e:
        logger.error(f"Error in SSE stream: {e}")
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.get("/stream")
async def events_stream():
    """Server-Sent Events endpoint for real-time updates.

    Emits `job_status_changed` events as job state actually changes (spec
    008), plus a periodic heartbeat -- both additive to, never a
    replacement for, polling `GET /jobs`/`GET /batches/{id}`.

    Returns:
        StreamingResponse: SSE stream of job-status-change and heartbeat events
    """
    logger.info("SSE client connected to /api/v1/events/stream")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )
