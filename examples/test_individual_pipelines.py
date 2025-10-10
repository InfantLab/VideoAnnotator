"""
Example: Individual Pipeline Testing

This example demonstrates how to test individual pipelines in isolation,
useful for debugging and development.

Usage:
    python examples/test_individual_pipelines.py --pipeline scene --video_path /path/to/video.mp4
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.audio_processing import AudioPipeline, AudioPipelineConfig
from src.pipelines.face_analysis import FacePipeline, FacePipelineConfig
from src.pipelines.person_tracking import PersonPipeline, PersonPipelineConfig

# Import the modernized pipelines
from src.pipelines.scene_detection import ScenePipeline, ScenePipelineConfig


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_scene_pipeline(video_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Test the scene detection pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Testing Scene Detection Pipeline")
    logger.info("=" * 50)

    # Create pipeline with config
    scene_config = ScenePipelineConfig(**config.get("scene_detection", {}))
    pipeline = ScenePipeline(scene_config)

    # Get pipeline info
    pipeline_info = pipeline.get_pipeline_info()
    logger.info(f"Pipeline Info: {json.dumps(pipeline_info, indent=2)}")

    # Process video
    logger.info(f"Processing video: {video_path}")
    results = pipeline.process_video(video_path)

    # Display results summary
    logger.info("Results Summary:")
    logger.info(f"  - Total scenes: {len(results.get('scenes', []))}")
    logger.info(f"  - Total duration: {results.get('total_duration', 0):.2f}s")

    if results.get("scenes"):
        logger.info("  - Scene breakdown:")
        for i, scene in enumerate(results["scenes"][:5]):  # Show first 5 scenes
            logger.info(
                f"    Scene {i + 1}: {scene.get('start_time', 0):.2f}s - {scene.get('end_time', 0):.2f}s"
            )

    return results


def test_person_pipeline(video_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Test the person tracking pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Testing Person Tracking Pipeline")
    logger.info("=" * 50)

    # Create pipeline with config
    person_config = PersonPipelineConfig(**config.get("person_tracking", {}))
    pipeline = PersonPipeline(person_config)

    # Get pipeline info
    pipeline_info = pipeline.get_pipeline_info()
    logger.info(f"Pipeline Info: {json.dumps(pipeline_info, indent=2)}")

    # Process video
    logger.info(f"Processing video: {video_path}")
    results = pipeline.process_video(video_path)

    # Display results summary
    logger.info("Results Summary:")
    logger.info(f"  - Total tracks: {len(results.get('tracks', []))}")
    logger.info(f"  - Total detections: {len(results.get('detections', []))}")

    if results.get("tracks"):
        logger.info("  - Track breakdown:")
        for i, track in enumerate(results["tracks"][:5]):  # Show first 5 tracks
            logger.info(f"    Track {i + 1}: {track.get('duration', 0):.2f}s duration")

    return results


def test_face_pipeline(video_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Test the face analysis pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Testing Face Analysis Pipeline")
    logger.info("=" * 50)

    # Create pipeline with config
    face_config = FacePipelineConfig(**config.get("face_analysis", {}))
    pipeline = FacePipeline(face_config)

    # Get pipeline info
    pipeline_info = pipeline.get_pipeline_info()
    logger.info(f"Pipeline Info: {json.dumps(pipeline_info, indent=2)}")

    # Process video
    logger.info(f"Processing video: {video_path}")
    results = pipeline.process_video(video_path)

    # Display results summary
    logger.info("Results Summary:")
    logger.info(f"  - Total faces: {len(results.get('faces', []))}")
    logger.info(f"  - Total face tracks: {len(results.get('face_tracks', []))}")

    if results.get("faces"):
        logger.info("  - Face breakdown:")
        for i, face in enumerate(results["faces"][:5]):  # Show first 5 faces
            logger.info(f"    Face {i + 1}: confidence {face.get('confidence', 0):.2f}")

    return results


def test_audio_pipeline(audio_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Test the audio processing pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Testing Audio Processing Pipeline")
    logger.info("=" * 50)

    # Create pipeline with config
    audio_config = AudioPipelineConfig(**config.get("audio_processing", {}))
    pipeline = AudioPipeline(audio_config)

    # Get pipeline info
    pipeline_info = pipeline.get_pipeline_info()
    logger.info(f"Pipeline Info: {json.dumps(pipeline_info, indent=2)}")

    # Process audio
    logger.info(f"Processing audio: {audio_path}")
    results = pipeline.process_audio(audio_path)

    # Display results summary
    logger.info("Results Summary:")
    logger.info(f"  - Duration: {results.get('duration', 0):.2f}s")
    logger.info(f"  - Sample rate: {results.get('sample_rate', 0)}Hz")
    logger.info(f"  - Audio segments: {len(results.get('segments', []))}")

    if results.get("speech_transcription"):
        logger.info(
            f"  - Speech transcription: {results['speech_transcription']['text'][:100]}..."
        )

    if results.get("speaker_diarization"):
        logger.info(
            f"  - Number of speakers: {results['speaker_diarization']['num_speakers']}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test individual VideoAnnotator pipelines"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        choices=["scene", "person", "face", "audio"],
        help="Pipeline to test",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--audio_path", type=str, help="Path to input audio file (for audio pipeline)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_file", type=str, help="Path to save results (optional)"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate inputs
    if args.pipeline in ["scene", "person", "face"]:
        if not args.video_path:
            logger.error(f"--video_path is required for {args.pipeline} pipeline")
            return 1

        video_path = Path(args.video_path)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return 1

    elif args.pipeline == "audio":
        if not args.audio_path:
            logger.error("--audio_path is required for audio pipeline")
            return 1

        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return 1

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return 1

    config = load_config(args.config)

    # Test the specified pipeline
    try:
        if args.pipeline == "scene":
            results = test_scene_pipeline(video_path, config)
        elif args.pipeline == "person":
            results = test_person_pipeline(video_path, config)
        elif args.pipeline == "face":
            results = test_face_pipeline(video_path, config)
        elif args.pipeline == "audio":
            results = test_audio_pipeline(audio_path, config)

        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to: {output_path}")

        logger.info("Pipeline test completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
