"""Manual smoke test for vlm_annotation — Option A from
docs/development/vlm_annotation_pipeline.md ("direct Python smoke test").

Talks straight to a local Ollama server, no API server, no database, no auth.
Requires Ollama running with the target model pulled (`ollama pull <model>`)
and VideoAnnotator installed with the `llm` extra (`uv pip install -e ".[llm]"`).

Usage:
    python tests/manual/vlm_annotation/smoke_test.py
    python tests/manual/vlm_annotation/smoke_test.py --model qwen3.5:9b --sampling-mode frame_burst

Running from inside a devcontainer/Docker container with Ollama on the host
machine: pass --base-url http://host.docker.internal:11434 and make sure
Ollama is listening on all interfaces, not just loopback (OLLAMA_HOST=0.0.0.0
before starting it) — 127.0.0.1 on the host is not reachable from a container.
"""

import argparse
from pathlib import Path

from videoannotator.pipelines.vlm_annotation import VLMAnnotationPipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO = REPO_ROOT / "temp" / "lw002.mp4"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument(
        "--sampling-mode",
        default="single_frame",
        choices=["single_frame", "frame_burst"],
    )
    parser.add_argument("--frame-interval-sec", type=float, default=5.0)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL — use http://host.docker.internal:11434 from a container",
    )
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"Video not found: {args.video}")

    pipeline = VLMAnnotationPipeline(
        {
            "model": args.model,
            "sampling_mode": args.sampling_mode,
            "frame_interval_sec": args.frame_interval_sec,
            "base_url": args.base_url,
        }
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annotations = pipeline.process(
        video_path=str(args.video),
        output_dir=str(OUTPUT_DIR),
    )
    for ann in annotations:
        print(ann["timestamp_sec"], ann["label"], ann["raw_response"])
    pipeline.cleanup()

    print(f"\n{len(annotations)} annotations written under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
