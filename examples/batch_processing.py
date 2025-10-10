"""Example: Batch Video Processing.

This example demonstrates how to process multiple videos in batch using
the modernized VideoAnnotator pipeline system.

Usage:
    python examples/batch_processing.py --input_dir /path/to/videos --output_dir /path/to/outputs"""

import argparse
import concurrent.futures
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.audio_processing import AudioPipeline, AudioPipelineConfig
from src.pipelines.face_analysis import FacePipeline, FacePipelineConfig
from src.pipelines.person_tracking import PersonPipeline, PersonPipelineConfig

# Import the modernized pipelines
from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_logging(log_level: str = "INFO", log_file: str = "batch_processing.log"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )


def find_video_files(input_dir: Path, extensions: list[str] = None) -> list[Path]:
    """Find all video files in the input directory."""
    if extensions is None:
        extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]

    video_files = []
    for ext in extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))

    return sorted(video_files)


def process_single_video(
    video_path: Path, output_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """
    Process a single video file through the annotation pipeline.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save output files
        config: Configuration dictionary

    Returns:
        Dictionary containing processing results and metadata
    """
    logger = logging.getLogger(__name__)

    # Create video-specific output directory
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now()

    # Initialize results dictionary
    results = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "output_dir": str(video_output_dir),
        "start_time": start_time.isoformat(),
        "processing_results": {},
        "status": "started",
    }

    logger.info(f"Processing video: {video_path.name}")

    try:
        # Initialize pipelines
        pipelines = {}

        if config.get("scene_detection", {}).get("enabled", True):
            scene_config = ScenePipelineConfig(**config.get("scene_detection", {}))
            pipelines["scene"] = ScenePipeline(scene_config)

        if config.get("person_tracking", {}).get("enabled", True):
            person_config = PersonPipelineConfig(**config.get("person_tracking", {}))
            pipelines["person"] = PersonPipeline(person_config)

        if config.get("face_analysis", {}).get("enabled", True):
            face_config = FacePipelineConfig(**config.get("face_analysis", {}))
            pipelines["face"] = FacePipeline(face_config)

        if config.get("audio_processing", {}).get("enabled", True):
            audio_config = AudioPipelineConfig(**config.get("audio_processing", {}))
            pipelines["audio"] = AudioPipeline(audio_config)

        # Process each pipeline
        for pipeline_name, pipeline in pipelines.items():
            try:
                logger.info(f"Running {pipeline_name} pipeline for {video_path.name}")

                if (
                    pipeline_name == "scene"
                    or pipeline_name == "person"
                    or pipeline_name == "face"
                ):
                    pipeline_results = pipeline.process_video(video_path)
                elif pipeline_name == "audio":
                    # For audio, we need to extract audio first
                    audio_path = video_output_dir / "extracted_audio.wav"
                    if audio_path.exists():
                        pipeline_results = pipeline.process_audio(audio_path)
                    else:
                        pipeline_results = {"error": "Audio file not found"}

                results["processing_results"][pipeline_name] = pipeline_results

                # Save individual pipeline results
                with open(video_output_dir / f"{pipeline_name}_results.json", "w") as f:
                    json.dump(pipeline_results, f, indent=2, default=str)

                logger.info(f"Completed {pipeline_name} pipeline for {video_path.name}")

            except Exception as e:
                logger.error(
                    f"Error in {pipeline_name} pipeline for {video_path.name}: {e}"
                )
                results["processing_results"][pipeline_name] = {"error": str(e)}

        results["status"] = "completed"

    except Exception as e:
        logger.error(f"Error processing {video_path.name}: {e}")
        results["status"] = "failed"
        results["error"] = str(e)

    # Record completion time
    end_time = datetime.now()
    results["end_time"] = end_time.isoformat()
    results["duration"] = (end_time - start_time).total_seconds()

    # Save complete results
    with open(video_output_dir / "complete_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Finished processing {video_path.name} ({results['status']})")
    return results


def process_videos_batch(
    video_files: list[Path],
    output_dir: Path,
    config: dict[str, Any],
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """
    Process multiple videos in parallel.

    Args:
        video_files: List of video file paths
        output_dir: Directory to save output files
        config: Configuration dictionary
        max_workers: Maximum number of parallel workers

    Returns:
        List of processing results for each video
    """
    logger = logging.getLogger(__name__)

    batch_results = []

    # Process videos in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_video = {
            executor.submit(
                process_single_video, video_path, output_dir, config
            ): video_path
            for video_path in video_files
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                batch_results.append(result)
                logger.info(f"Completed processing: {video_path.name}")
            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {e}")
                batch_results.append(
                    {
                        "video_path": str(video_path),
                        "video_name": video_path.name,
                        "status": "failed",
                        "error": str(e),
                    }
                )

    return batch_results


def generate_batch_report(batch_results: list[dict[str, Any]], output_dir: Path):
    """Generate a summary report for the batch processing."""

    # Calculate statistics
    total_videos = len(batch_results)
    successful = sum(1 for r in batch_results if r.get("status") == "completed")
    failed = sum(1 for r in batch_results if r.get("status") == "failed")

    total_duration = sum(r.get("duration", 0) for r in batch_results if "duration" in r)
    avg_duration = total_duration / max(successful, 1)

    # Create summary report
    report = {
        "batch_summary": {
            "total_videos": total_videos,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total_videos if total_videos > 0 else 0,
            "total_duration": total_duration,
            "average_duration": avg_duration,
        },
        "detailed_results": batch_results,
    }

    # Save report
    with open(output_dir / "batch_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Create CSV summary
    import csv

    with open(output_dir / "batch_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video Name", "Status", "Duration (s)", "Error"])

        for result in batch_results:
            writer.writerow(
                [
                    result.get("video_name", ""),
                    result.get("status", ""),
                    result.get("duration", ""),
                    result.get("error", ""),
                ]
            )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos through VideoAnnotator pipeline"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save output files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".avi", ".mov"],
        help="Video file extensions to process",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level, "batch_processing.log")
    logger = logging.getLogger(__name__)

    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    # Load configuration
    config = load_config(args.config)

    # Find video files
    video_files = find_video_files(input_dir, args.extensions)

    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return 1

    logger.info(f"Found {len(video_files)} video files to process")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration: {config_path}")
    logger.info(f"Max workers: {args.max_workers}")

    # Process videos
    try:
        batch_start_time = datetime.now()
        batch_results = process_videos_batch(
            video_files, output_dir, config, args.max_workers
        )
        batch_end_time = datetime.now()

        # Generate report
        report = generate_batch_report(batch_results, output_dir)

        # Print summary
        logger.info("Batch processing completed!")
        logger.info(f"Total videos: {report['batch_summary']['total_videos']}")
        logger.info(f"Successful: {report['batch_summary']['successful']}")
        logger.info(f"Failed: {report['batch_summary']['failed']}")
        logger.info(f"Success rate: {report['batch_summary']['success_rate']:.2%}")
        logger.info(f"Total duration: {batch_end_time - batch_start_time}")
        logger.info(f"Report saved to: {output_dir / 'batch_report.json'}")

        return 0

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
