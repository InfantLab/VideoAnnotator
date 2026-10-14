"""A pipeline that only writes one small file: CI's launcher end-to-end test
runs a real job with it in the slim image (tests/launcher/Dockerfile.e2e)."""

import json
import time
from pathlib import Path
from typing import Any

from videoannotator.pipelines.base_pipeline import BasePipeline


class LauncherE2EPipeline(BasePipeline):
    def initialize(self) -> None:
        self.is_initialized = True

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        # Slow enough, when asked, that jobs are still queued at a restart.
        time.sleep(float(self.config.get("seconds", 0)))
        video = Path(video_path)
        result = [{"video": video.name, "bytes": video.stat().st_size}]
        if output_dir:
            out = Path(output_dir) / f"{Path(video_path).stem}_launcher_e2e.json"
            out.write_text(json.dumps(result))
        return result

    def cleanup(self) -> None:
        pass

    def get_schema(self) -> dict[str, Any]:
        return {"type": "array"}
