"""The video list stays fast at one lab's scale (spec 029 R11): 5,000 jobs over
2,000 video files, with reruns and batches, indexed and listed in under 1 s."""

from __future__ import annotations

import os
import time
import uuid
from datetime import timedelta

from sqlalchemy import insert

from tests.fixtures.video_index import T0
from videoannotator.storage.models import Job, PipelineResult
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend
from videoannotator.video_index import build_index

VIDEOS = 2_000
JOBS = 5_000
TARGET_SEC = 1.0


def _fill(tmp_path, storage: SQLiteStorageBackend) -> None:
    folders = [tmp_path / "data" / f"site_{n}" for n in range(20)]
    for folder in folders:
        folder.mkdir(parents=True)
    videos = []
    for n in range(VIDEOS):
        path = folders[n % len(folders)] / f"p{n:04d}.mp4"
        path.write_bytes(b"v")
        videos.append(path)

    uploads = tmp_path / "storage"
    jobs, results = [], []
    for n in range(JOBS):
        job_id = str(uuid.uuid4())
        video = videos[n % VIDEOS]
        storage_path = None
        if n % 10 == 0:  # an upload kept in its job's folder, hard-linked
            folder = uploads / job_id
            folder.mkdir(parents=True)
            stored = folder / video.name
            os.link(video, stored)
            video, storage_path = stored, str(folder)
        jobs.append(
            {
                "id": job_id,
                "video_path": str(video),
                "status": "completed",
                "selected_pipelines": ["face_analysis", "scene_detection"],
                "config": {},
                "created_at": T0 + timedelta(seconds=n),
                "completed_at": T0 + timedelta(seconds=n + 30),
                "batch_id": f"batch-{n // 50}",
                "storage_path": storage_path,
                "progress_percentage": 100,
            }
        )
        for name in ("face_analysis", "scene_detection"):
            results.append(
                {
                    "job_id": job_id,
                    "pipeline_name": name,
                    "status": "completed",
                    "provenance": {"input": {"sha256": f"{n % VIDEOS:064x}"}},
                }
            )
    with storage.engine.begin() as conn:
        conn.execute(insert(Job), jobs)
        conn.execute(insert(PipelineResult), results)


def test_five_thousand_jobs_index_and_list_in_under_a_second(tmp_path):
    storage = SQLiteStorageBackend(tmp_path / "jobs.db")
    _fill(tmp_path, storage)

    started = time.perf_counter()
    index = build_index(storage.job_summaries(), uploads_root=tmp_path / "storage")
    videos = index.videos()
    elapsed = time.perf_counter() - started

    print(f"\n{JOBS} jobs, {len(videos)} videos: {elapsed:.3f} s")
    assert len(videos) == VIDEOS
    assert sum(len(v.jobs) for v in videos) == JOBS
    assert elapsed < TARGET_SEC, f"{elapsed:.2f} s"
